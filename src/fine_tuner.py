import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import SGD
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm
import os
import wandb
import json

class LoRALayer(nn.Linear):
    #LoRA implementation for a linear layer
    def __init__(
            self,
            in_features: int,
            out_features: int, 
            r: int,
            lora_alpha: int =0,
            local_init: bool =True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.lora_alpha = lora_alpha
        self.r = r
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        else :
            self.delta = nn.Parameter(self.weight.new_zeros(in_features, out_features)
                                      )
        self.scaling = self.lora_alpha / self.r if self.r!=0 else 0
        self.weight.requires_grad = False
        self.local_init = local_init

        self.reset_parameters()
        

    def reset_parameters(self):
        nn.Linear.reset_parameters(self) 
        if hasattr(self, 'lora_A'):
            if self.local_init:
                # initialize B the same way as the default for nn.Linear and A to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else: 
                nn.init.kaiming_uniform_(self.lora_A, a = 1000)
                nn.init.kaiming_uniform_(self.lora_B, a = 1000)
        if hasattr(self, 'delta'):
            nn.init.kaiming_uniform(self.delta, a=2)
    
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
        tuning_weights: str = "last",    # one, last, or all
        rank: int = 4,
        lmbda: float = 0.01,            # Weight decay OR nuclear-norm coefficient
        local_initialization: bool = True,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size:int = 32,
        device: str = "cuda",
        project_name: str = None,
        log_dir : str = None
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
        self.lmbda = lmbda if rank == 0 else 0.0
        self.local_initialization = local_initialization
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.project_name = project_name                                                                

        # 1. Unfreeze or transform the relevant layers
        self.configure_layers()
        self.model = self.model.to(device)
        # 2. Build optimizer (and optionally a scheduler)
        self.optimizer = SGD(self._get_trainable_params(), lr=self.learning_rate, weight_decay=self.lmbda if self.rank > 0 else 0.0)
        self.lr_scheduler = None  # if you want, define a scheduler

        # Wandb logging
        self.project_name = project_name
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
            'epochs' : num_epochs
        }
        self.save_config()
        if project_name is not None:
            wandb.init(project=project_name, config=self.config, name=f'tuned={self.tuning_weights}_LoRA={self.rank}_lambda={self.lmbda}')

    def save_config(self):
        os.makedirs(f"{self.log_dir}", exist_ok=True)
        config_path = os.path.join(f"{self.log_dir}", 'config.json')
        with open(config_path, 'w') as json_file:
            json.dump(self.config, json_file, indent=4)

    def save_model(self):
        save_path = f"{self.log_dir}/lambda_{self.lmbda}_epoch_{self.num_epochs}_lr.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    def configure_layers(self):
        """
        Unfreezes the relevant Q and/or V weights (depending on `tuning_weights`).
        If LoRA is True, we replace them by LoRALayer. Otherwise, we simply unfreeze them 
        for standard training (and will apply nuclear-norm reg in the loss).
        
        We assume:
            - For a RoBERTa model: model.roberta.encoder.layer[...] 
                Q -> self_attn.k_proj.weight, 
                V -> self_attn.v_proj.weight
              or sometimes: q_proj, v_proj. The exact naming can vary. 
            
            - For a ViT model: model.vit.encoder.layer[...].attention.attention.query, etc.
            
        Modify below logic to match the exact submodule names in your architecture.
        """
        # Identify the relevant layers
        # Example: for demonstration, let's assume the naming in huggingface's 
        #  Roberta: model.roberta.encoder.layer[i].attention.self
        #  ViT:     model.vit.encoder.layer[i].layernorm_after or .attention.attention...
        # In practice, you must adapt these to the real submodule structure.

        if hasattr(self.model, "roberta"):
            # We have a RoBERTa model
            encoder_layers = self.model.roberta.encoder.layer
        elif hasattr(self.model, "vit"):
            # We have a ViT model
            encoder_layers = self.model.vit.encoder.layer
        else:
            raise ValueError("Model structure not recognized for Q/V projection heads.")

        # Decide which layers to modify
        if self.tuning_weights == "one":
            # Only the query matrix of the last attention layer
            # => We do not modify value
            layers_to_modify = [len(encoder_layers) - 1]
            q_or_v = ["q"]  # indicates we'll only do Q in the last layer
        elif self.tuning_weights == "last":
            # Both Q and V of the last attention layer
            layers_to_modify = [len(encoder_layers) - 1]
            q_or_v = ["q", "v"]
        elif self.tuning_weights == "all":
            # Q and V of all layers
            layers_to_modify = list(range(len(encoder_layers)))
            q_or_v = ["q", "v"]
        else:
            raise ValueError("tuning_weights must be one of ['one','last','all']")

        for layer_idx in layers_to_modify:
            attn_module = encoder_layers[layer_idx].attention
            if hasattr(attn_module, "attention"):  # For ViT
                submodule = attn_module.attention
            elif hasattr(attn_module, "self"):  # For RoBERTa
                submodule = attn_module.self

            print(submodule)

            if hasattr(submodule, "query") and hasattr(submodule, "value"):  # Roberta, vit
                q_name = "query"    
                v_name = "value"
            elif hasattr(submodule, "q_proj") and hasattr(submodule, "v_proj"):  # Some other models
                q_name = "q_proj"
                v_name = "v_proj"
            # For each name (q_proj or v_proj), we freeze/unfreeze or replace with LoRA
            if "q" in q_or_v:
                self._replace_with_lora(submodule, q_name)
            if "v" in q_or_v and self.tuning_weights!='one':
                self._replace_with_lora(submodule, v_name)

    def _replace_with_lora(self, parent_module: nn.Module, attr_name: str):
        """
        replace the parent's submodule (a Linear) with LoRALayer. 
        """
        if not hasattr(parent_module, attr_name):
            print(f"Warning: {attr_name} not found in {parent_module}. Skipping.")
            return

        linear_layer = getattr(parent_module, attr_name)
        if not isinstance(linear_layer, nn.Linear):
            raise TypeError(f"{attr_name} is not nn.Linear but {type(linear_layer)}")
        # Replace or unfreeze
        # Convert this linear to LoRALayer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Create LoRALayer
        lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            r=self.rank,
            lora_alpha=self.rank,  # you may want a different alpha
            local_init=self.local_initialization,
            bias=(linear_layer.bias is not None)
        )
        # Copy the pretrained weight & bias (LoRA layer's base weight is kept frozen, but we copy it over)
        with torch.no_grad():
            lora_layer.weight.copy_(linear_layer.weight)
            if linear_layer.bias is not None:
                lora_layer.bias.copy_(linear_layer.bias)

        # Finally replace
        setattr(parent_module, attr_name, lora_layer)


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
        for param in params_list:
            if param.ndim < 2:
                # 1D or scalar param => skip or treat as vector norm
                continue
            # Flatten the 2D (or more) into 2D for SVD
            mat = param.view(param.shape[0], -1).detach()
            # Avoid computing large SVD on huge dims if not needed, 
            # but let's do it for demonstration
            _, S, _ = torch.svd(mat)  # or torch.linalg.svd in newer PyTorch
            nuc_norm = S.sum()
            total_nuc_norm += nuc_norm
        return total_nuc_norm

    def _compute_rank_of_deltas(self, params_list):
        """
        At the end of each epoch, print rank(\Delta W).
        """
        # We'll sum up "effective" ranks or just print them individually.
        # For demonstration, let's sum rank across all trainable params.
        # The rank is the number of singular values above a threshold, e.g. 1e-5.
        rank = []
        threshold = 1e-8
        for param in params_list:
            if param.ndim < 2:
                continue
            mat = param.view(param.shape[0], -1).detach()
            _, R = torch.linalg.qr(mat, mode="r")
            rank.append(torch.sum(torch.abs(torch.diag(R)) > threshold).item())
        return rank

    
    
    def train(self):
        # def hook_fn(module, input, output):
        #     print(f"[Hook] Module: {module.__class__.__name__}")
        #     print(f"Input Shapes: {[i.shape for i in input if isinstance(i, torch.Tensor)]}")
        #     print(f"Output Shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")

        # # Register hooks for all modules in the model
        # for name, module in self.model.named_modules():
        #     module.register_forward_hook(hook_fn)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

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

            trainable_params = []
            all_param = 0
            for name, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params.append((name, param.shape))
                    print(f"Training: {name}, Shape: {param.shape}")

            for batch in tqdm(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                # forward pass
                outputs = self.model(**batch)
                vanilla_loss = outputs.loss

                # If not using LoRA, apply nuclear norm regularization
                if self.rank == 0:
                    # Only apply nuclear norm to the Q/V weights we are training
                    trainable_params = self._get_trainable_params()
                    nuc_norm = self._compute_nuclear_norm(trainable_params)
                    reg_loss = self.lmbda * nuc_norm
                    loss = vanilla_loss + reg_loss

                else:
                    loss = vanilla_loss

                loss.backward()
                self.optimizer.step()
                # if self.lr_scheduler is not None:
                #     self.lr_scheduler.step()

                total_loss += loss.item()

                current_lr = self.optimizer.param_groups[0]['lr']
                if self.project_name is not None:
                    wandb.log({
                        "train/vanilla_loss": vanilla_loss,
                        "train/reg_loss": reg_loss if self.rank == 0 else None,
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "epoch": epoch
                    }, step = self.global_step)

                self.global_step +=1

            # Validation step (optional)
            val_acc = self.evaluate()['accuracy']

            # Compute rank(\Delta W) at end of epoch
            if self.rank == 0:
                trainable_params = self._get_trainable_params()
                rank_deltaW = self._compute_rank_of_deltas(trainable_params)
            else:
                rank_deltaW = 0

            print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                  f"Train Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Accuracy: {val_acc:.4f}, "
                  f"Rank(ΔW): {rank_deltaW}")
            
            if self.project_name is not None:
                wandb.log({
                        "test/val_acc": val_acc,
                        "test/rank": rank_deltaW,
                        "epoch": epoch
                    }, step = self.global_step)
        
        if self.project_name is not None:
            wandb.finish()
            self.save_model()
            

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
