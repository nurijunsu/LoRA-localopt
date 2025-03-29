import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import transformers
from transformers import get_scheduler
from tqdm import tqdm
import os
import wandb
import json
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from fine_tuner import FineTuningTrainer, LoRALayer
import argparse
from dataset import CustomDataset
from models import Model_Pretrained
import multiprocessing as mp
from multiprocessing import Process, Event, Queue, Barrier
import copy
import time

class MultiTrainer:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        models: list,
        base_seed: int = 42,
        tuning_weights: str = "all",
        rank: int = 16,
        rank_star: int = 4,
        lmbda: float = 0.01,
        L2_reg: bool = False,
        num_epochs: int = 100,
        learning_rate: float = 5e-3,
        batch_size: int = 128,
        grad_clip: float = 5.0,
        project_name: str = None,
        run_name: str = None,
        log_dir: str = None,
        run_wandb: bool = True,
        optimizer: str = "SGD",
        lr_scheduler: str = None,
        proximal_gradient: bool = False
    ):
        """
        Initialize MultiTrainer that trains 3 models in parallel on different GPUs.
        """
        self.base_seed = base_seed
        self.num_models = len(models)
        self.trainers = []
        self.devices = self._get_available_gpus()
        self.num_epochs = num_epochs
        self.project_name = project_name
        self.run_wandb = run_wandb
        self.run_name = run_name
        self.global_step = 0
        
        if len(self.devices) < self.num_models:
            raise ValueError(f"Not enough GPUs available. Need {self.num_models}, got {len(self.devices)}")

        # Initialize trainers for each model
        for i in range(self.num_models):
            # Set random seed for this model
            torch.manual_seed(base_seed + i)
            np.random.seed(base_seed + i)
            # Create model instance
            model = models[i]
            print(f"Initializing model {i} on {self.devices[i]}")
            # Create trainer
            trainer = FineTuningTrainer(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model,
                tuning_weights=tuning_weights,
                rank=rank,
                rank_star=rank_star,
                lmbda=lmbda,
                L2_reg=L2_reg,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                grad_clip=grad_clip,
                device=self.devices[i],
                project_name=project_name,
                run_name=f"{run_name}_model{i+1}" if run_name else None,
                log_dir=f"{log_dir}" if log_dir else None,
                run_wandb=False,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                proximal_gradient=proximal_gradient
            )
            self.trainers.append(trainer)

        if self.run_wandb:
            self.config = {
                'Task': train_dataset.task_name,
                'Tuning Weights': tuning_weights,
                'LoRA rank': rank,
                'base_seed': base_seed,
                'lambda': lmbda,
                'lr': learning_rate,
                'batch_size': batch_size,
                'epochs': num_epochs,
                'optimizer': optimizer,
                'scheduler': lr_scheduler,
            }
            wandb.init(project=project_name, config=self.config, name=run_name)

    def _train_model_process(self, model_idx, barrier, metrics_queue, state_dict_queue):
        """Train a single model in a separate process."""
        # Set process name for better identification
        mp.current_process().name = f"Model-{model_idx+1}"
        process_seed = self.base_seed + model_idx
        torch.manual_seed(process_seed)
        np.random.seed(process_seed)
        torch.cuda.manual_seed(process_seed)
        
        trainer = self.trainers[model_idx]
        device = trainer.device
        
        for epoch in range(self.num_epochs):            
            print(f"Model {model_idx+1} on {device} - Epoch {epoch+1}/{self.num_epochs}")
            # Create a scaler for mixed precision training
            scaler = GradScaler()
            
            # Train for one epoch
            trainer.train_one_epoch(epoch, scaler)
            
            # Collect metrics
            val_metrics = trainer.evaluate()
            metrics = {
                f"model{model_idx+1}_val_{key}": value for key, value in val_metrics.items()
            }
            metrics[f"model{model_idx+1}_train_loss"] = trainer.train_loss
            
            print(f"Model {model_idx+1} - Loss: {trainer.train_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save model periodically
            if (epoch + 1) % self.save_epoch == 0:
                trainer.save_model(epoch + 1)
            
            # Put metrics in the queue for the main process
            try:
                metrics_queue.put((model_idx, metrics), timeout=60)
            except Exception as e:
                print(f"Model {model_idx+1} - Error putting metrics in queue: {e}")
                continue
            
            # Move model to CPU and get state_dict for parameter difference computation
            try:
                # Add explicit CUDA synchronization before CPU transfer
                torch.cuda.synchronize(device)
                
                trainer.model = trainer.model.cpu()
                
                # Clone tensors and ensure they're on CPU
                cpu_state_dict = {}
                for name, param in trainer.model.state_dict().items():
                    if param.is_cuda and "lora" in name:
                        cpu_state_dict[name] = param.detach().cpu().clone()
                    else:
                        cpu_state_dict[name] = param.detach().clone()
                
                state_dict_queue.put((model_idx, cpu_state_dict), timeout=120)
                
                # Return model to its device after copying state dict
                trainer.model = trainer.model.to(device)
            except Exception as e:
                print(f"Model {model_idx+1} - Error processing state_dict: {e}")
                # Try to ensure model is back on the right device
                if not next(trainer.model.parameters()).is_cuda:
                    trainer.model = trainer.model.to(device)
                
                # Put None in the queue to signal an error occurred
                try:
                    state_dict_queue.put((model_idx, None), timeout=10)
                except:
                    print(f"Model {model_idx+1} - Failed to put None in state_dict_queue")
            
            try:
                # Wait for all processes to reach this point before continuing
                barrier.wait(timeout=300)
            except Exception as e:
                print(f"Model {model_idx+1} - Barrier timeout: {e}")
        
        # Final clean up
        torch.cuda.empty_cache()
        print(f"Model {model_idx+1} training completed")

    def train(self):
        """Train all models in parallel."""
        print("Starting parallel training with multiprocessing")
        
        # Set start method to spawn to ensure clean process isolation
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Method already set, which is fine
            pass
        
        # Create multiprocessing objects for synchronization
        barrier = mp.Barrier(self.num_models)
        metrics_queue = mp.Queue()
        state_dict_queue = mp.Queue()
        
        # Create processes for each model
        processes = []
        for i in range(self.num_models):
            p = Process(
                target=self._train_model_process, 
                args=(i, barrier, metrics_queue, state_dict_queue)
            )
            p.daemon = True  # Ensure processes terminate when main process ends
            processes.append(p)
            p.start()
            # Small delay to ensure processes start cleanly one after another
            time.sleep(0.1)
        
        # Main process monitors and collects results after each epoch
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - Main process collecting results")
            
            # Collect metrics from all models
            epoch_metrics = {}
            model_state_dicts = {}
            
            # Collect metrics from all models with timeout
            metrics_collected = 0
            try:
                for _ in range(self.num_models):
                    model_idx, metrics = metrics_queue.get(timeout=300)
                    epoch_metrics.update(metrics)
                    metrics_collected += 1
            except Exception as e:
                print(f"Error collecting metrics: {e}. Got {metrics_collected}/{self.num_models}")
            
            # Collect state dicts for parameter difference computation
            states_collected = 0
            try:
                for _ in range(self.num_models):
                    model_idx, state_dict = state_dict_queue.get(timeout=300)
                    if state_dict is not None:  # Handle case where process sent None due to error
                        model_state_dicts[model_idx] = state_dict
                        states_collected += 1
            except Exception as e:
                print(f"Error collecting state dicts: {e}. Got {states_collected}/{self.num_models}")
            
            # Compute parameter differences only if we have all required state dicts
            differences = {}
            if len(model_state_dicts) >= 2:  # Need at least 2 models to compare
                try:
                    differences = self._compute_parameter_differences_from_state_dicts(model_state_dicts)
                except Exception as e:
                    print(f"Error computing parameter differences: {e}")
            else:
                print(f"Too few state dicts collected ({len(model_state_dicts)}) to compute differences")
            
            # Clean up state dictionaries to free memory
            del model_state_dicts
            torch.cuda.empty_cache()
            
            # Log metrics and differences to wandb
            if self.run_wandb:
                wandb.log({
                    **differences,
                    **epoch_metrics,
                    "epoch": epoch
                }, step=self.global_step)
                self.global_step += 1
            
            print(f"Completed epoch {epoch+1}/{self.num_epochs}")
            
            # Check if processes are still alive
            alive_processes = [p.is_alive() for p in processes]
            if not all(alive_processes):
                print(f"Warning: Some processes have died. Alive: {sum(alive_processes)}/{len(processes)}")
        
        # Join all processes
        for p in processes:
            if p.is_alive():
                p.join(timeout=10)  # Don't hang forever on join
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        if self.run_wandb:
            wandb.finish()

    def _get_available_gpus(self):
        """Get list of available GPU devices."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(num_gpus)]

    def _compute_parameter_differences_from_state_dicts(self, model_state_dicts):
        """
        Compute L2 norm of parameter differences between LoRA effective weights (B@A).
        Instead of comparing individual parameters, we compute lora_B @ lora_A for each
        layer and then compare the differences between models.
        """
        differences = {}
        
        # Debug received model_state_dicts
        print(f"Computing differences between {len(model_state_dicts)} models")
        for model_idx, state_dict in model_state_dicts.items():
            lora_count = sum(1 for name in state_dict if "lora" in name)
            print(f"Model {model_idx+1} state dict has {len(state_dict)} keys, {lora_count} LoRA params")
        
        # Group parameters by layer and type (A or B)
        # The structure will be: {layer_name: {'A': {'model0': tensor, 'model1': tensor, ...}, 
        #                                      'B': {'model0': tensor, 'model1': tensor, ...}}}
        grouped_params = {}
        
        # First, collect all lora parameters and group them
        for model_idx, state_dict in model_state_dicts.items():
            for name, param in state_dict.items():
                if "lora" in name:
                    # Extract the layer name (everything before .lora_A or .lora_B)
                    if ".lora_A" in name:
                        layer_name = name.split(".lora_A")[0]
                        param_type = "A"
                    elif ".lora_B" in name:
                        layer_name = name.split(".lora_B")[0]
                        param_type = "B"
                    else:
                        # Skip if not lora_A or lora_B
                        continue
                    
                    # Initialize layer entry if it doesn't exist
                    if layer_name not in grouped_params:
                        grouped_params[layer_name] = {"A": {}, "B": {}}
                        
                    grouped_params[layer_name][param_type][model_idx] = param

        # Now compute B@A for each layer and compare between models
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                # Skip if either model's data is missing
                if i not in model_state_dicts or j not in model_state_dicts:
                    print(f"Skipping comparison between models {i+1} and {j+1} due to missing data")
                    continue
                
                try:
                    diff_norm = 0
                    norm = 0
                    layer_count = 0
                    layer_diffs = {}
                    layer_diff_ratios = {}
                    
                    for layer_name, layer_params in grouped_params.items():
                        # Check if we have all parameters for both models
                        has_all_params = (i in layer_params.get("A", {}) and 
                                        j in layer_params.get("A", {}) and
                                        i in layer_params.get("B", {}) and 
                                        j in layer_params.get("B", {}))
                        
                        if not has_all_params:
                            print(f"Missing parameters for layer {layer_name} in comparison between models {i+1} and {j+1}")
                            continue
                        
                        # Get A and B parameters for both models
                        A_i = layer_params["A"][i]
                        B_i = layer_params["B"][i]
                        A_j = layer_params["A"][j]
                        B_j = layer_params["B"][j]
                          
                        # Compute effective weights (B@A)
                        effective_weight_i = B_i @ A_i
                        effective_weight_j = B_j @ A_j
                        
                        # Compute difference and add to total norm
                        diff = effective_weight_i - effective_weight_j
                        layer_norm = torch.norm(effective_weight_i, p=2).item()
                        layer_diff_norm = torch.norm(diff, p=2).item()
                        diff_norm += layer_diff_norm
                        norm += layer_norm
                        layer_diffs[layer_name] = layer_diff_norm
                        layer_count += 1
                    
                    
                    differences[f"model{i+1}/model{i+1}_model{j+1}_diff"] = diff_norm
                    differences[f"model{i+1}/model{i+1}_model{j+1}_diff_ratio"] = diff_norm / norm
                except Exception as e:
                    print(f"Error computing difference between models {i+1} and {j+1}: {e}")
        
        return differences

    def evaluate(self, test_dataset=None):
        """Evaluate all models and return their metrics."""
        metrics = {}
        for i, trainer in enumerate(self.trainers):
            model_metrics = trainer.evaluate(test_dataset)
            for key, value in model_metrics.items():
                metrics[f"model{i+1}/model{i+1}_{key}"] = value
        return metrics

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task_name', default='sst2', type=str, help='fine tuning task')
    parse.add_argument('--lmbda', default=0.01, type=float, help='weight decay parameter')
    parse.add_argument('--tuning_weights', default='all', type=str, help='finetuning type')
    parse.add_argument('--learning_rate', default=0.001, type=float, help ='learning rate')
    parse.add_argument('--rank', default=8, type=int, help ='rank')
    parse.add_argument('--local_init', default="True", type=str, help ='local initialization')
    parse.add_argument('--Scheduler', type=str, default = 'CosineDecay',help ='Learning Rate Scheduler')
    parse.add_argument('--sample_ratio', type=float, default=1.0, help ='Fraction of dataset to use (default: 0.1)')
    args = parse.parse_args()
    return args

if __name__ == "__main__":

    #python multi_trainer.py --task_name sst2 --lmbda 0.01 --tuning_weights all --learning_rate 0.005 --rank 8 --local_init True --Scheduler CosineDecay
    args = parse_args()
    task_name = args.task_name
    lmbda = args.lmbda
    tuning_weights = args.tuning_weights
    learning_rate = args.learning_rate
    rank = args.rank
    local_init = args.local_init
    Scheduler = args.Scheduler
    sample_ratio = args.sample_ratio

    base_seed = 42
    num_models = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # Create custom datasets with 10% sampling
    train_dataset = CustomDataset(task_name=task_name,split="train", sample_ratio=sample_ratio, random_seed=base_seed)

    test_dataset = CustomDataset(task_name=task_name, split="test", sample_ratio=sample_ratio, random_seed=base_seed)

    print("Datasets Created")

    models = []

    
    for i in range(num_models):
        model_seed = base_seed + i
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        model_loader = Model_Pretrained("roberta",task_name, fine_tuned=True, rank=rank, tuning_weights=tuning_weights, local_init = local_init)  
        model = model_loader.get_model()
        models.append(model)
    
    print("Model Loaded")

    project_name= f'Rebuttal_Sameglobalmin'


    trainer = MultiTrainer(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        models = models,
        base_seed=base_seed,
        tuning_weights=tuning_weights,
        rank= rank,
        lmbda=lmbda,
        L2_reg=False,
        num_epochs=500,  
        save_epoch=100,
        learning_rate=learning_rate,
        batch_size=256,
        grad_clip=5.0,
        project_name=project_name,
        run_name=f"tryout3_fasttrain",
        log_dir = f'../logs/{train_dataset.task_name}/tuned={tuning_weights}_LoRA={rank}/lambda_{lmbda}',
        optimizer="SGD",
        lr_scheduler=Scheduler
    )

    trainer.train()