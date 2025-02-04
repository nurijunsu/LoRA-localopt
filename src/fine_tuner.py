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

class LoRALayer(nn.Linear):
    #LoRA implementation for a linear layer
    def __init__(
            self,
            in_features: int,
            out_features: int, 
            r: int,
            lora_alpha: int =0,
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
        nn.Linear.reset_parameters(self) 
        if hasattr(self, 'lora_A'):
            if self.local_init=="True":
                # initialize B the same way as the default for nn.Linear and A to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                print('locally initialized')
            elif self.local_init == "LargeRandom":
                nn.init.normal_(self.lora_A, mean=0, std=1/5)
                nn.init.normal_(self.lora_B, mean=0, std=1/5)
                print('initialized with large random')
            elif self.local_init == "Ortho":
                with torch.no_grad():
                    nn.init.normal_(self.lora_A, mean=0, std=1/20)
                    self.lora_B.data.copy_(self.lora_A.T) 
            # nn.init.constant_(self.lora_A, 10)  # Large positive constant
                # nn.init.constant_(self.lora_B, -10)  # Large negative constant
                # with torch.no_grad():
                #     self.lora_A.zero_()
                #     self.lora_B.zero_()
                #     self.lora_A[0, 0] = 50  # Only one large value in A
                #     self.lora_B[0, 0] = -50  # Only one large value in B`
                print('orthogonally initialized')

    
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
        rank_star: int = 5,               # rank of global min    
        lmbda: float = 0.01,            # Weight decay OR nuclear-norm coefficient
        L2_reg: bool = False,
        local_initialization: bool = True,
        num_epochs: int = 100,
        learning_rate: float = 5e-3,
        batch_size:int = 128,
        grad_clip: float = 50.0,
        device: str = "cuda",
        project_name: str = None, 
        run_name: str = None,
        log_dir : str = None, 
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
            local_initialization: If True, standard LoRA init; if False, “exploding” init.
            num_epochs: Number of epochs to train.
            learning_rate: Learning rate for the AdamW optimizer.
            device: 'cuda' or 'cpu'.
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = device
        self.project_name = project_name
        self.proximal_gradient = proximal_gradient                                                                

        # 1. Unfreeze or transform the relevant layers
        self.model = self.model.to(device)
        # 2. Build optimizer (and optionally a scheduler)
        if optimizer == "SGD":
            self.optimizer = SGD(self._get_trainable_params(), lr=self.learning_rate, momentum = 0.0 if (L2_reg or proximal_gradient or self.rank>0) else 0.9)
        elif optimizer =="Adam":
            self.optimizer = Adam(self._get_trainable_params(), lr=self.learning_rate)

        if lr_scheduler == "ReduceLROnPlateu":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, threshold = 0.002, patience=5, min_lr=5e-7) 
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
                                        num_warmup_steps= round(0.05 * self.num_epochs),
                                        num_training_steps= self.num_epochs
                                    )   
        elif lr_scheduler == "LinearWarmup":
            warmup_steps = 0.05 * self.num_epochs
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))
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
        if project_name is not None:
            wandb.init(project=project_name, config=self.config, name=run_name)

    def save_config(self):
        os.makedirs(f"{self.log_dir}", exist_ok=True)
        config_path = os.path.join(f"{self.log_dir}", 'config.json')
        with open(config_path, 'w') as json_file:
            json.dump(self.config, json_file,   indent=4)

    def save_model(self, epochs):
        save_path = f"{self.log_dir}/lambda_{self.lmbda}_epoch_{epochs}_lr.pth"
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
                sigma_r.append(S[rank[-1]-1])
                sigma_r_star.append(S[self.rank_star])
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
                sigma_r.append(S[rank[-1]-1])
                sigma_r_star.append(S[self.rank_star])

        return sigma_r, sigma_r_star, rank

    
    
    def train(self):

        # def hook_fn(module, input, output):
        #     print(f"[Hook] Module: {module.__class__.__name__}")
        #     print(f"Input Shapes: {[i.shape for i in input if isinstance(i, torch.Tensor)]}")
        #     print(f"Output Shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")

        # # Register hooks for all modules in the model
        # for name, module in self.model.named_modules():
        #     module.register_forward_hook(hook_fn)
        scaler = GradScaler()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            total_reg = 0.0

            def collate_fn(batch):
                # Each item is a dict with 'input_ids', 'attention_mask', 'labels', etc.
                # They are all the same length (since we used padding="max_length").
                input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch], dim=0)
                attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch], dim=0)
                labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)

                return {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                    "labels": labels.to(self.device),
                }
            
            if hasattr(self.model, 'roberta'):
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn
                )
            else: 
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                )

            for batch in tqdm(train_loader):
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
                                if self.project_name is not None:
                                    wandb.log({
                                        f"train/rank_{i}": torch.sum(s> 1e-8).item()
                                    }, step = self.global_step)   
                total_loss += vanilla_loss.item()
                total_reg += reg_loss.item()

                if self.project_name is not None:
                    wandb.log({
                        "train/vanilla_loss": vanilla_loss,
                        "train/reg": nuc_norm if self.rank ==0 else L2_norm/2,
                        "train/loss": loss.item(),
                        "train/learning_rate": learning_rate,
                        "train/grad_norm": grad_norm,
                        "epoch": epoch
                    }, step = self.global_step)

                self.global_step +=1

            # Validation step (optional)
            val_acc = self.evaluate()['accuracy']

            # Compute rank(\Delta W) at end of epoch
            trainable_params = self._get_trainable_params()
            sigma_r , sigma_r_star, rank_deltaW = self._compute_rank_of_deltas(trainable_params)
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
                  f"Rank(ΔW): {rank_deltaW}")
            
            if self.project_name is not None:
                if (epoch+1)% 50 ==0:
                    self.save_model(epoch+1)

                wandb.log({
                        "test/total_train_loss": self.train_loss,
                        "test/val_acc": val_acc,
                        "test/reg_loss": reg_loss,
                        "test/nuc_norm": nuc_norm,
                        **{f"test/rank_{i}": rank_deltaW[i] for i in range(len(rank_deltaW))}, 
                        **{f"test/sigma_r_{i}": sigma_r[i] for i in range(len(sigma_r))}, 
                        **{f"test/sigma_r_star_{i}": sigma_r_star[i] for i in range(len(sigma_r_star))}, 
                        "test/sigma_ratio(weight 1)": sigma_r_star[1]/sigma_r[1],
                        "epoch": epoch  
                    }, step = self.global_step)
        
        if self.project_name is not None:
            wandb.finish()
            self.save_model(self.num_epochs)
            

    def evaluate(self, test_dataset = None):
        dataset = test_dataset if test_dataset is not None else self.test_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        self.model.eval()

        def collate_fn(batch):
            # batch is a list of items from dataset[i]
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch], dim=0)
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch], dim=0)
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
            return {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "labels": labels.to(self.device),
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