import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, ViTForImageClassification
from fine_tuner import LoRALayer

class Model_Pretrained:
    def __init__(self, model_name: str, dataset_name: str, fine_tuned:bool = False, rank:int =0, tuning_weights:str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name.lower()
        self.valid_roberta_datasets = ['sst2', 'qnli', 'qqp']
        self.model = self._load_and_modify_model()
        self.rank = rank
        self.tuning_weights = tuning_weights
        if fine_tuned:
            print('fine_tuned')
            self._configure_layers()
        
        
    def _validate_combination(self):
        if self.model_name == 'roberta':
            if self.dataset_name not in self.valid_roberta_datasets:
                raise ValueError(f"Dataset {self.dataset_name} not valid for RoBERTa. Use one of {self.valid_roberta_datasets}")
        elif self.model_name == 'vit':
            if self.dataset_name != 'cifar100':
                raise ValueError(f"Dataset {self.dataset_name} not valid for ViT. Use 'cifar100'")
        else:
            raise ValueError(f"Model {self.model_name} not supported. Use 'roberta' or 'vit'")

    def _load_and_modify_model(self):
        self._validate_combination()
        
        # Load pretrained model
        if self.model_name == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        else:  # vit
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100, ignore_mismatched_sizes=True)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Load custom head weights
        head_path = f"../pretrained_models/model_{self.dataset_name}/classification_head.pt"
        head_state = torch.load(head_path)
        
        # Update classification head
        if self.model_name == 'roberta':
            model.classifier.dense.weight.data = head_state['dense.weight']
            model.classifier.dense.bias.data = head_state['dense.bias']
            model.classifier.out_proj.weight.data = head_state['out_proj.weight']
            model.classifier.out_proj.bias.data = head_state['out_proj.bias']
            print("Loaded two-layer classification head for RoBERTa.")
        else:  # vit
            model.classifier.weight.data = head_state['weight']
            model.classifier.bias.data = head_state['bias']
            print("Loaded linear classification head for ViT")
        return model
    
    def _configure_layers(self):
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
                print('replaced with lora')
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
            lora_alpha=self.rank,
            bias=(linear_layer.bias is not None)
        )
        # Copy the pretrained weight & bias (LoRA layer's base weight is kept frozen, but we copy it over)
        with torch.no_grad():
            lora_layer.weight.copy_(linear_layer.weight)
            if linear_layer.bias is not None:
                lora_layer.bias.copy_(linear_layer.bias)

        # Finally replace
        setattr(parent_module, attr_name, lora_layer)

    def get_model(self):
        return self.model
