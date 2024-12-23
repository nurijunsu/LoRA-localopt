import torch
from transformers import RobertaForSequenceClassification, ViTForImageClassification

class Model_Pretrained:
    def __init__(self, model_name: str, dataset_name: str):
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name.lower()
        self.valid_roberta_datasets = ['sst2', 'qnli', 'qqp']
        self.model = self._load_and_modify_model()
        
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
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Load custom head weights
        head_path = f"../pretrained_models/model_{self.dataset_name}/linear_head.pt"
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
        return model.to(self.device)

    def get_model(self):
        return self.model