from fine_tuner import LoRALayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
from models import Model_Pretrained
from dataset import CustomDataset
from tqdm import tqdm
import pandas as pd 
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

class RSCM_checker:
    def __init__(self, 
                 pretrained_model, 
                 tuning_weights, 
                 train_dataset,
                 epsilon = 1,
                 rank=4, 
                 num_samples=1000,):  # Add batch_size parameter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = pretrained_model.to(self.device)
        self.tuning_weights = tuning_weights
        self.epsilon = epsilon
        self.rank = rank
        self.num_samples = num_samples
        self.weight_list = self._configure_weights()
        self.train_dataset = train_dataset
        self.batch_size = 256

        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch], dim=0)
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch], dim=0)
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
            return {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "labels": labels.to(self.device),
            }
        
        if hasattr(self.model, "roberta"):
            self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,  # Use smaller batch size
                        shuffle=False,
                        collate_fn=collate_fn
                    )
        elif hasattr(self.model, "vit"):
            self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,  # Use smaller batch size
                        shuffle=False,
                    )

        for param in self.model.parameters():
            param.requires_grad = False
        for (submodule, param_name) in self.weight_list:
            tuning_layer = getattr(submodule, param_name)
            tuning_layer.delta.requires_grad = True

    def _configure_weights(self):
        weight_list = []
        if hasattr(self.model, "roberta"):
            encoder_layers = self.model.roberta.encoder.layer
        elif hasattr(self.model, "vit"):
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

            if hasattr(submodule, "query") and hasattr(submodule, "value"):  # Roberta, vit
                q_name = "query"    
                v_name = "value"
            elif hasattr(submodule, "q_proj") and hasattr(submodule, "v_proj"):  # Some other models
                q_name = "q_proj"
                v_name = "v_proj"
            # For each name (q_proj or v_proj), we freeze/unfreeze or replace with LoRA

            if "q" in q_or_v:
                weight_list.append((submodule, q_name))
            if "v" in q_or_v:
                weight_list.append((submodule, v_name))
        return weight_list
    
    def generate_local_rank_r(self, matrix):
        m, n = matrix.shape
        A = torch.randn(m, self.rank, device = self.device)
        B = torch.randn(self.rank, n, device = self.device)
        X = A @ B
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        scale_factor = S[self.rank-1] 
        if scale_factor > 0:
            X = X * torch.rand(1).item() * self.epsilon / scale_factor
            
        return X

    def compute_gradient(self, weight_config, delta=None):
        parent_module, attr_name = weight_config
        lora_layer = getattr(parent_module, attr_name)

        if delta is not None:
            original_data = lora_layer.delta.data.clone()
            lora_layer.delta.data.copy_(delta)

        # Initialize accumulated gradient
        accumulated_gradient = None
        num_batches = len(self.train_loader)
        
        for batch in tqdm(self.train_loader):
            self.model.zero_grad()     
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model(**batch).loss
            loss = loss / num_batches  # Scale the loss to get proper average
            loss.backward()

            # Accumulate gradients
            if accumulated_gradient is None:
                accumulated_gradient = lora_layer.delta.grad
            else:
                accumulated_gradient += lora_layer.delta.grad
        if delta is not None:
            lora_layer.delta.data.copy_(original_data)

        return accumulated_gradient

    def compute_Hessian_vector_prod(self, weight_config, delta: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Compute the scalar value of direction^T H direction, where H is the Hessian
        of the loss function with respect to delta.
        Args:
            delta: The perturbation tensor.
            direction: The vector to compute the Hessian product with.
        Returns:
            hvp_scalar: The scalar value of direction^T H direction.
        """
        # Ensure delta requires gradient
        parent_module, attr_name = weight_config
        lora_layer = getattr(parent_module, attr_name)
        if not isinstance(lora_layer, LoRALayer):
            raise TypeError(f"{attr_name} is not nn.Linear but {type(lora_layer)}")
    
        original_data = lora_layer.delta.data.clone()
        delta.requires_grad_(True)

        subset_dataset = torch.utils.data.Subset(self.train_dataset, range(100))  # Use a subset
        subset_loader = DataLoader(
            subset_dataset, 
            batch_size = self.batch_size,
            shuffle = False, 
            collate_fn=self.train_loader.collate_fn
        )

        lora_layer.delta = nn.Parameter(delta.clone(), requires_grad=True)

        loss=0.0
        # Compute the loss
        for batch in tqdm(subset_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast():
                loss += self.model(**batch).loss

        loss= loss / len(subset_loader)
        self.model.zero_grad()
        loss.backward(create_graph = True)
        # Compute the first gradient (grad of loss w.r.t. delta)
        with autocast():
            grad = lora_layer.delta.grad
        # Compute the dot product of grad and direction
        grad_dot_direction = torch.dot(grad.view(-1), direction.view(-1))
        
        grad_dot_direction.backward()
        
        # Compute the second gradient (Hessian-vector product)
        hessian_vector = lora_layer.delta.grad
        
        # Compute the scalar value of direction^T H direction
        hvp_scalar = torch.dot(hessian_vector.view(-1), direction.view(-1))

        lora_layer.delta.data.copy_(original_data)
        
        return hvp_scalar
    
    def RSC(self) -> List[float]:
        """
        Check restricted strong convexity by computing:
        <∇L(delta), delta - delta*> / ||delta - delta*||_F^2
        for sample_number random deltas with rank ≤ test_rank.
        
        Returns:
            the whole value list
        """
        values_dict = defaultdict(list)
        for parent_module, attr_name in self.weight_list:
            lora_layer = getattr(parent_module, attr_name)
            delta_star = lora_layer.delta.data
            star_gradient = self.compute_gradient(weight_config=(parent_module, attr_name))
            column_name = f"{parent_module}_{attr_name}"
            for _ in range(self.num_samples):
                delta = self.generate_local_rank_r(delta_star)
            
                # Compute gradient at delta
                gradient_diff = self.compute_gradient(weight_config = (parent_module, attr_name), delta= delta.to(self.device)) - star_gradient
                
                # Compute difference
                diff = delta - delta_star
                
                # Compute inner product
                numerator = torch.sum(gradient_diff * diff)
                denominator = torch.norm(diff, p='fro') ** 2
                
                values_dict[column_name].append((numerator / denominator).item())
            
        return pd.DataFrame(values_dict)
    
    def RSM(self) -> List[float]:
        """
        Compute β_local(X) for matrices X of rank ≤ test_rank within epsilon ball of delta_star.
        Uses random sampling followed by rank projection and validation.
        """
        values_dict = defaultdict(list)
        for parent_module, attr_name in self.weight_list:
            lora_layer = getattr(parent_module, attr_name)
            delta_star = lora_layer.delta.data
            column_name = f"{parent_module}_{attr_name}"
        
            for _ in range(self.num_samples):
                # Generate X with correct rank within epsilon ball
                X = self.generate_local_rank_r(delta_star)
                m, n = X.shape
                
                u1 = torch.rand(m, 1, device=self.device)  # m×1 vector
                u2 = torch.rand(1, m, device=self.device)  # 1×m vector
                U = u1 @ u2
                
                v1 = torch.rand(n, 1, device=self.device)  # n×1 vector
                v2 = torch.rand(1, n, device=self.device)
                V = v1 @ v2

                direction = U @ X + X @ V

                beta = self.compute_Hessian_vector_prod((parent_module, attr_name), X, direction)/(torch.norm(direction, p='fro')**2)
                values_dict[column_name].append(beta.item())
    
        return pd.DataFrame(values_dict)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    pretrained_model =  Model_Pretrained(model_name='roberta', dataset_name='sst2', fine_tuned=True, rank = 0, tuning_weights= 'one').get_model()
    pretrained_model.load_state_dict(torch.load('../logs/sst2/tuned=one_LoRA=0/lambda_0.1/lambda_0.1_epoch_100_lr.pth'))
    train_dataset = CustomDataset(task_name='sst2', split="train")
    checker= RSCM_checker(pretrained_model=pretrained_model, tuning_weights='one', train_dataset=train_dataset,rank=4, num_samples=50)
    print("hello, world!")
    print(checker.RSM())