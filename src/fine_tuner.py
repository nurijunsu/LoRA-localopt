import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class TransformerFineTuner:
    def __init__(
        self,
        model: nn.Module,
        tuning_weights: str = 'one',
        use_lora: bool = False,
        rank: int = 8,
        lambda_reg: float = 0.1,
        local_initialization: bool = True,
        device: str = 'cuda',
        seed: int = 42
    ):
        """
        Initialize the fine-tuning trainer.
        
        Args:
            model: Pre-trained model (RoBERTa or ViT)
            tuning_weights: Which attention matrices to tune ('one', 'last', 'all')
            use_lora: Whether to use LoRA or full fine-tuning
            rank: Rank for LoRA decomposition
            lambda_reg: Regularization strength
            local_initialization: If False, use exploding initialization
            device: Device to use for training
            seed: Random seed for initialization
        """
        self.model = model
        self.tuning_weights = tuning_weights
        self.use_lora = use_lora
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.local_initialization = local_initialization
        self.device = device
        self.seed = seed
        
        torch.manual_seed(seed)
        
        # Get attention layers based on model type
        if hasattr(model, 'roberta'):
            self.attention_layers = self._get_roberta_attention_layers()
        else:  # ViT
            self.attention_layers = self._get_vit_attention_layers()
            
        # Initialize delta weights or LoRA matrices
        self.delta_weights = {}
        self.lora_weights = {}
        self._initialize_tunable_weights()

    def _get_roberta_attention_layers(self) -> List[nn.Module]:
        """Get attention layers from RoBERTa model."""
        layers = []
        for layer in self.model.roberta.encoder.layer:
            layers.append(layer.attention.self)
        return layers
    
    def _get_vit_attention_layers(self) -> List[nn.Module]:
        """Get attention layers from ViT model."""
        layers = []
        for block in self.model.vit.encoder.layer:
            layers.append(block.attention)
        return layers

    def _initialize_tunable_weights(self):
        """Initialize delta weights or LoRA matrices based on configuration."""
        num_layers = len(self.attention_layers)
        
        if self.tuning_weights == 'one':
            layers_to_tune = [num_layers - 1]
        elif self.tuning_weights == 'last':
            layers_to_tune = [num_layers - 1]
        else:  # 'all'
            layers_to_tune = range(num_layers)
            
        for layer_idx in layers_to_tune:
            layer = self.attention_layers[layer_idx]
            hidden_size = layer.query.weight.shape[0]
            
            if self.use_lora:
                # Initialize LoRA matrices
                if not self.local_initialization:
                    scale = 100.0  # Large scale for exploding initialization
                else:
                    scale = 1.0 / math.sqrt(hidden_size)
                    
                self.lora_weights[f'layer_{layer_idx}_q'] = {
                    'A': nn.Parameter(torch.randn(hidden_size, self.rank) * scale).to(self.device),
                    'B': nn.Parameter(torch.randn(self.rank, hidden_size) * scale).to(self.device)
                }
                
                if self.tuning_weights in ['last', 'all']:
                    self.lora_weights[f'layer_{layer_idx}_k'] = {
                        'A': nn.Parameter(torch.randn(hidden_size, self.rank) * scale).to(self.device),
                        'B': nn.Parameter(torch.randn(self.rank, hidden_size) * scale).to(self.device)
                    }
            else:
                # Initialize delta weights for full fine-tuning
                if not self.local_initialization:
                    scale = 100.0
                else:
                    scale = 1.0 / math.sqrt(hidden_size)
                    
                self.delta_weights[f'layer_{layer_idx}_q'] = nn.Parameter(
                    torch.randn_like(layer.query.weight) * scale
                ).to(self.device)
                
                if self.tuning_weights in ['last', 'all']:
                    self.delta_weights[f'layer_{layer_idx}_k'] = nn.Parameter(
                        torch.randn_like(layer.key.weight) * scale
                    ).to(self.device)

    def compute_nuclear_norm_loss(self) -> torch.Tensor:
        """Compute nuclear norm regularization loss."""
        loss = 0.0
        for name, delta in self.delta_weights.items():
            # Compute SVD
            _, S, _ = torch.svd(delta)
            loss += self.lambda_reg * torch.sum(S)
        return loss

    def compute_rank(self, tolerance: float = 1e-8) -> Dict[str, int]:
        """Compute approximate rank of delta weights using SVD."""
        ranks = {}
        for name, weight in self.delta_weights.items():
            _, S, _ = torch.svd(weight)
            rank = torch.sum(S > tolerance).item()
            ranks[name] = rank
        return ranks

    def forward_hook(self, layer_idx: int, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        """Forward hook for modifying attention computations."""
        if self.use_lora:
            # Apply LoRA transformations
            q_lora = self.lora_weights[f'layer_{layer_idx}_q']
            delta_q = torch.mm(torch.mm(input[0], q_lora['A']), q_lora['B'])
            
            if self.tuning_weights in ['last', 'all']:
                k_lora = self.lora_weights[f'layer_{layer_idx}_k']
                delta_k = torch.mm(torch.mm(input[0], k_lora['A']), k_lora['B'])
            else:
                delta_k = 0
                
            return output + delta_q + delta_k
        else:
            # Apply delta weights
            delta_q = torch.mm(input[0], self.delta_weights[f'layer_{layer_idx}_q'])
            
            if self.tuning_weights in ['last', 'all']:
                delta_k = torch.mm(input[0], self.delta_weights[f'layer_{layer_idx}_k'])
            else:
                delta_k = 0
                
            return output + delta_q + delta_k

    def register_hooks(self):
        """Register forward hooks for all relevant attention layers."""
        self.hooks = []
        num_layers = len(self.attention_layers)
        
        if self.tuning_weights == 'one':
            layers_to_hook = [num_layers - 1]
        elif self.tuning_weights == 'last':
            layers_to_hook = [num_layers - 1]
        else:  # 'all'
            layers_to_hook = range(num_layers)
            
        for layer_idx in layers_to_hook:
            layer = self.attention_layers[layer_idx]
            hook = layer.register_forward_hook(
                lambda m, i, o, idx=layer_idx: self.forward_hook(idx, m, i, o)
            )
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()

    def get_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        if self.use_lora:
            params = []
            for lora_dict in self.lora_weights.values():
                params.extend([lora_dict['A'], lora_dict['B']])
            return params
        else:
            return list(self.delta_weights.values())

    def train_step(self, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Add regularization if not using LoRA
        if not self.use_lora:
            loss += self.compute_nuclear_norm_loss()
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}

    def print_ranks(self):
        """Print approximate ranks of delta weights."""
        if not self.use_lora:
            ranks = self.compute_rank()
            for name, rank in ranks.items():
                print(f"Rank of {name}: {rank}")

