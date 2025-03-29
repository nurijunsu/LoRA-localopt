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
import multiprocessing as mp

class LoRALayer(nn.Linear):
    #LoRA implementation for a linear layer
    def __init__(
            self,
            in_features: int,
            out_features: int, 
            r: int,
            lora_alpha: int =4,
            local_init: str = "True",
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.lora_alpha = lora_alpha
        self.r = r
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        else :
            self.delta = nn.Parameter(self.weight.new_zeros(in_features, out_features))
        self.scaling = self.lora_alpha / self.r if self.r!=0 else 0
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.local_init = local_init

        self.reset_parameters()

    def reset_parameters(self):
        print("resetting parameters")
        nn.Linear.reset_parameters(self) 
        if hasattr(self, 'lora_A'):
            if self.local_init=="True":
                # initialize B the same way as the default for nn.Linear and A to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif self.local_init == "LargeRandom":
                nn.init.normal_(self.lora_A, mean=0, std=1/4)
                nn.init.normal_(self.lora_B, mean=0, std=1/4)
                # with torch.no_grad():
                #     self.lora_A[0, 0] = 10 
                #     self.lora_B[1, 1] = -10
                print('initialized with large random')
            elif self.local_init == "Ortho":
                with torch.no_grad():
                    nn.init.normal_(self.lora_A, mean=-1/20, std=1/20)
                    self.lora_B.data.copy_(self.lora_A.T) 
            # nn.init.constant_(self.lora_A, 10)  # Large positive constant
                # nn.init.constant_(self.lora_B, -10)  # Large negative constant
                # with torch.no_grad():
                #     self.lora_A.zero_()
                #     self.lora_B.zero_()
                #     self.lora_A[0, 0] = 50  # Only one large value in A
                #     self.lora_B[0, 0] = -50  # Only one large value in B`
                print('orthogonally initialized')
            elif self.local_init == "SingleValue":
                nn.init.zeros_(self.lora_A)
                nn.init.zeros_(self.lora_B)
                with torch.no_grad():
                    self.lora_A[0, 0] = 35 
                    self.lora_B[0, 0] = 35

    
    def forward(self, x:torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.r > 0:
            result += (x @ self.lora_A.transpose(0,1) @ self.lora_B.transpose(0,1)) * self.scaling
        else:
            result += x @ self.delta
        return result
    


###############################################################################
# Model Fine-Tuning Trainer
###############################################################################

class FineTuningTrainer:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model: nn.Module, 
        tuning_weights: str = "all",    # one, last, or all
        rank: int = 16,                  # full fine tuning if rank=0
        rank_star: int = 4,               # rank of global min    
        lmbda: float = 0.01,            # Weight decay OR nuclear-norm coefficient
        L2_reg: bool = False,
        local_initialization: bool = True,
        num_epochs: int = 100,
        save_epoch: int = 10,
        learning_rate: float = 5e-3,
        batch_size:int = 128,
        grad_clip: float = 10,
        device: str = "cuda:0",
        project_name: str = None, 
        run_name: str = None,
        log_dir : str = None, 
        run_wandb: bool = True,
        optimizer: str = "SGD",  #SGD  #Adam
        lr_scheduler: str = None, #ReduceLROnPlateu, CosineAnnealing, CosineDecay, LinearWarmup
        proximal_gradient: bool = False # Only for full fine tuning (rank=0)
    ):
        """
        Args:
            model: A preloaded (and partially frozen) model, such as a RobertaForSequenceClassification or ViTForImageClassification.
            tuning_weights: 'one' (only Q of last layer), 'last' (Q & V of last layer), or 'all' (Q & V of all layers).
            lora: If True, apply LoRA to the specified layers; if False, do standard full fine-tuning with nuclear norm regularization.
            rank: Rank used in LoRA; ignored if lora=False.
            lmbda: Regularization coefficient. If lora=True, this acts as weight decay; if lora=False, it is nuclear-norm coefficient.
            local_initialization: If True, standard LoRA init; if False, "exploding" init.
            num_epochs: Number of epochs to train.
            learning_rate: Learning rate for the AdamW optimizer.
            device: Specific CUDA device (e.g., 'cuda:0', 'cuda:1') or 'cpu'.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tuning_weights = tuning_weights
        self.rank = rank
        self.rank_star = rank_star
        self.lmbda = lmbda 
        self.L2_reg = L2_reg
        self.num_epochs = num_epochs
        self.save_epoch = save_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        
        # Validate and set device
        if device.startswith('cuda:'):
            gpu_idx = int(device.split(':')[1])
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if gpu_idx >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_idx} is not available. Only {torch.cuda.device_count()} GPUs found.")
        elif device != 'cpu':
            raise ValueError("Device must be either a specific CUDA device (e.g., 'cuda:0') or 'cpu'")
        self.device = device
        
        self.project_name = project_name
        self.run_wandb = run_wandb
        self.run_name = run_name
        self.proximal_gradient = proximal_gradient                                                                

        # 1. Unfreeze or transform the relevant layers
        self.model = self.model.to(self.device)
        # 2. Build optimizer (and optionally a scheduler)
        if optimizer == "SGD":
            self.optimizer = SGD(self._get_trainable_params(), lr=self.learning_rate, momentum = 0.0 if (L2_reg or proximal_gradient or self.rank>0) else 0.9)
        elif optimizer =="Adam":
            self.optimizer = Adam(self._get_trainable_params(), lr=self.learning_rate)

        if lr_scheduler == "ReduceLROnPlateu":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, threshold = 0.0005, patience=3, min_lr=5e-7) 
        elif lr_scheduler == "CosineAnnealing":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                    self.optimizer,
                                    T_0=20,  # Number of epochs for the first restart
                                    T_mult = 2, #  A factor by which T_i increases after a restart
                                    eta_min=1e-8,  # Minimum learning rate at the end of each cycle
                                )
        elif lr_scheduler == "CosineDecay":
            self.lr_scheduler = get_scheduler(
                                        name="cosine",
                                        optimizer=self.optimizer,
                                        num_warmup_steps= round(0.01 * self.num_epochs),
                                        num_training_steps= self.num_epochs
                                    )               
        elif lr_scheduler == "LinearWarmup":
            warmup_steps = 0.05 * self.num_epochs
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))
        elif lr_scheduler == "LinearDecay":
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: 1- step/self.num_epochs)
        else:
            self.lr_scheduler = None
                                                   
        self.train_loss = 0

        # Wandb logging
        self.project_name = project_name
        if run_name is None: 
            run_name = f'tuned={self.tuning_weights}_LoRA={self.rank}_lambda={self.lmbda}'
        self.global_step = 0                                                       
        if log_dir is not None:
            self.log_dir = log_dir
        else:             
            self.log_dir = f'../logs/{train_dataset.task_name}/tuned={self.tuning_weights}_LoRA={self.rank}/lambda_{self.lmbda}'
        self.config = {
            'Task' : train_dataset.task_name,
            'Tuning Weights' : tuning_weights,                                                                                    
            'LoRA rank' : rank,                        
            'local_initialization' : local_initialization,                                                                                                                               
            'lambda': lmbda,
            'lr' : learning_rate,
            'batch_size' : batch_size,
            'epochs' : num_epochs,
            'optimizer' : self.optimizer.__class__.__name__,
            'scheduler' : lr_scheduler,
            'proximal_gradient' : self.proximal_gradient
        }
        self.save_config()
        if run_wandb:
            wandb.init(project=project_name, config=self.config, name=run_name)

    def save_config(self):
        os.makedirs(f"{self.log_dir}/{self.project_name}/", exist_ok=True)
        config_path = os.path.join(f"{self.log_dir}/{self.project_name}/", f'{self.run_name}_config.json')
        with open(config_path, 'w') as json_file:
            json.dump(self.config, json_file,   indent=4)

    def save_model(self, epochs):
        if self.project_name is not None:
            save_path = f"{self.log_dir}/{self.project_name}/{self.run_name}_epoch_{epochs}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    def _get_trainable_params(self):
        """
        Return the list of all parameters that are trainable. 
        In LoRA mode, that will be the LoRA parameters; 
        In full fine-tune mode, that will be Q/V params we marked as requires_grad=True.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def _compute_nuclear_norm(self, params_list):
        """
        Sums the nuclear norms (sum of singular values) of each parameter in params_list.
        This is used if LoRA == False for nuclear norm regularization.
        """
        total_nuc_norm = 0.0
        if self.rank == 0:
            total_nuc_norm = sum(torch.norm(X, p='nuc') for X in params_list)
        else:
            total_nuc_norm = sum(torch.norm(B @ A, p='nuc') for A, B in zip(params_list[::2], params_list[1::2]))
        return total_nuc_norm

    def _compute_rank_of_deltas(self, params_list):
        """
        At the end of each epoch, print rank(\Delta W).
        """
        # We'll sum up "effective" ranks or just print them individually.
        # For demonstration, let's sum rank across all trainable params.
        # The rank is the number of singular values above a threshold, e.g. 1e-5.
        rank = []
        sigma_r = []
        sigma_r_star = []
        if self.rank == 0:
            for param in params_list:
                if param.ndim < 2:
                    continue
                # Reshape and detach the parameter for SVD computation
                mat = param.view(param.shape[0], -1).detach()
                # Perform SVD
                _, S, _ = torch.linalg.svd(mat, full_matrices=False)
                # Compute rank based on the threshold
                threshold = 1e-3 * min(S[0],1)
                rank.append(torch.sum(S > threshold).item())
                sigma_r.append(S[self.rank-1])
                sigma_r_star.append(S[self.rank_star-1])
        else:
            for i in range(0, len(params_list), 2):
                A = params_list[i]     # Get A matrix
                B = params_list[i+1]   # Get B matrix
                mat = B @ A  # Compute the matrix product
                # Perform SVD
                _, S, _ = torch.linalg.svd(mat, full_matrices=False)
                threshold = 1e-3 * min(S[0],1)

                # Compute rank based on the threshold
                rank.append(torch.sum(S > threshold).item())
                sigma_r.append(S[self.rank-1])
                sigma_r_star.append(S[self.rank_star-1])

        return sigma_r, sigma_r_star, rank

    
    
    def train(self):
        scaler = GradScaler()

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch, scaler)
        
        if self.project_name is not None:
            wandb.finish()
            self.save_model(self.num_epochs)

    def train_one_epoch(self, epoch, scaler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_reg = 0.0

        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch], dim=0)
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch], dim=0)
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        if hasattr(self.model, 'roberta'):
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,  # Disable multiprocessing for DataLoader
                pin_memory=True
            )
        else: 
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Disable multiprocessing for DataLoader
                pin_memory=True
            )

        # Get the process ID for multiprocessing environments
        try:
            process_id = mp.current_process().name
            is_main_process = ("MainProcess" in process_id) or ("Model-1" in process_id)
        except:
            is_main_process = True  # Default to showing progress bar if not in multiprocessing

        # Create a progress bar with a description that includes the process information
        desc = f"Epoch {epoch+1}" if is_main_process else f"Process {process_id} - Epoch {epoch+1}"
        for batch in tqdm(train_loader, desc=desc, disable=not is_main_process):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            # forward pass
            with autocast():
                outputs = self.model(**batch)
                vanilla_loss = outputs.loss

                trainable_params = self._get_trainable_params()
                # If not using LoRA, apply nuclear norm regularization
                if self.rank == 0:
                    # Only apply nuclear norm to the Q/V weights we are training
                    if self.proximal_gradient:
                        reg_loss = 0
                    else:
                        nuc_norm = self._compute_nuclear_norm(trainable_params)
                        reg_loss = self.lmbda * nuc_norm                        
                else:
                    if self.L2_reg:
                        L2_norm =  sum(torch.norm(B @ A, p='fro')**2 for A, B in zip(trainable_params[::2], trainable_params[1::2]))
                    else: 
                        L2_norm = sum((matrix ** 2).sum() for matrix in trainable_params)
                    reg_loss = self.lmbda * L2_norm/2 
                loss = vanilla_loss + reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
            learning_rate = self.optimizer.param_groups[0]['lr']
            if self.proximal_gradient:
                nuc_norm = 0
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if "delta" in name and self.lmbda >= 1e-8: #Skip this if there is no weight decay (lmbda= 0)                       
                        with torch.no_grad():
                            u,s,v = torch.linalg.svd(param, full_matrices= False)
                            s = torch.nn.Threshold(0, 0)(s- learning_rate * self.lmbda)  #Soft-thresholding operator 
                            param.data = (u @ torch.diag(s)) @ v
                            nuc_norm += s.sum()
                            if self.run_wandb:
                                wandb.log({
                                    f"train/rank_{i}": torch.sum(s> 1e-8).item()
                                }, step = self.global_step)   
            total_loss += vanilla_loss.item()
            total_reg += reg_loss.item()

            if self.run_wandb:
                wandb.log({
                    "train/vanilla_loss": vanilla_loss,
                    "train/reg": nuc_norm if self.rank ==0 else L2_norm/2,
                    "train/loss": loss.item(),
                    "train/learning_rate": learning_rate,
                    "train/grad_norm": grad_norm,
                    "epoch": epoch
                }, step = self.global_step)

            self.global_step +=1

        # Validation step
        val_acc = self.evaluate()['accuracy']

        # Compute rank(\Delta W) at end of epoch
        trainable_params = self._get_trainable_params()
        nuc_norm = self._compute_nuclear_norm(trainable_params)
        self.train_loss = total_loss/len(train_loader)
        reg_loss = total_reg/len(train_loader)
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.train_loss)  # Pass train_loss for ReduceLROnPlateau
            else:
                self.lr_scheduler.step()  # Call step() for other schedulers
        
        print(f"Epoch [{epoch+1}/{self.num_epochs}], "
              f"Train Loss: {self.train_loss:.4f}, "
              f"Val Accuracy: {val_acc:.4f}, "
              )
        
        if self.run_wandb:
            if (epoch+1)% self.save_epoch ==0:
                self.save_model(epoch+1)

            wandb.log({
                    "test/total_train_loss": self.train_loss,
                    "test/val_acc": val_acc,
                    "test/reg_loss": reg_loss,
                    "test/nuc_norm": nuc_norm,
                    "epoch": epoch  
                }, step = self.global_step)

    def evaluate(self, test_dataset = None):
        dataset = test_dataset if test_dataset is not None else self.test_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        self.model.eval()

        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch], dim=0)
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch], dim=0)
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        if isinstance(self.model, transformers.RobertaForSequenceClassification):
            eval_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
        else:
            eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits
                all_predictions.append(predictions)
                all_targets.append(batch["labels"])

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if self.model.num_labels == 1:  # Regression
            targets = targets.float().unsqueeze(-1)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            metrics = {"mse": loss.item()}
        else:
            loss = torch.nn.functional.cross_entropy(predictions, targets)
            pred_classes = predictions.argmax(dim=-1)
            accuracy = (pred_classes == targets).float().mean()
            metrics = {"loss": loss.item(), "accuracy": accuracy.item()}

        return metrics