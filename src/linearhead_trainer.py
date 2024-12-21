import os
import torch
from typing import Optional
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
import transformers
from transformers.utils import logging
import numpy as np

logger = logging.get_logger(__name__)

def get_token_prediction_layer(model):
    """Extract the final prediction layer based on model type."""
    if isinstance(model, transformers.RobertaForSequenceClassification):
        return model.classifier.out_proj
    elif isinstance(model, transformers.ViTForImageClassification):
        return model.classifier
    elif isinstance(model, transformers.Wav2Vec2ForSequenceClassification):
        return model.classifier.out_proj
    else:
        raise NotImplementedError(f"Model type {model.__class__} is not supported.")

def extract_features(model, *args, **kwargs):
    """Extract features from the layer before the final prediction layer."""
    features = {}
    def hook(model_, input_, output_):
        features["features"] = input_[0].detach()

    get_token_prediction_layer(model).register_forward_hook(hook)
    model.forward(*args, **kwargs)
    return features["features"]

class LinearHeadTrainer:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dataset_name: str = "default"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        self.dataset_name = dataset_name
        self.model = self.model.to(self.device)

    def collect_features(self, dataset):
        """Collect features and targets from the dataset."""
        features = []
        targets = []
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        self.model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                inputs = dataset[i]
                
                # Convert each field in `inputs` to a tensor on the correct device
                tensor_inputs = {}
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        # Case 1: Already a Tensor
                        tensor_inputs[k] = v.to(device)
                    elif isinstance(v, np.ndarray):
                        # Case 2: NumPy array
                        tensor_inputs[k] = torch.from_numpy(v).to(device)
                    else:
                        # Case 3: Python scalar, list, etc.
                        tensor_inputs[k] = torch.tensor(v).to(device)
                
                # Extract features using your custom function
                feature = extract_features(self.model, **tensor_inputs)
                
                # Collect features and labels on CPU to save GPU memory
                features.append(feature.cpu())
                targets.append(tensor_inputs["labels"].cpu())

        # Concatenate on CPU
        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Return NumPy arrays for downstream tasks (e.g., scikit-learn)
        return features.numpy(), targets.numpy()

    def train(self):
        """Train the linear head using regression."""
        logger.info("Starting to collect features for training dataset")
        features, targets = self.collect_features(self.train_dataset)
        
        if self.model.num_labels == 1:  # Regression
            targets = targets.squeeze().unsqueeze(-1).float()
            reg = LinearRegression().fit(features, targets)
        else:  # Classification
            reg = LogisticRegressionCV(
                max_iter=5000,
                multi_class="multinomial",
                random_state=0
            ).fit(features.numpy(), targets.numpy())
        
        logger.info("Fitting regression model")
        
        # Assign weights to model
        decoder = get_token_prediction_layer(self.model)
        coef_torch = torch.tensor(reg.coef_, device=self.device, dtype=decoder.weight.dtype)
        bias_torch = torch.tensor(reg.intercept_, device=self.device, dtype=decoder.bias.dtype)
        
        # Handle binary classification case
        if self.model.num_labels == 2 and coef_torch.size(0) == 1:
            coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
            bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)
        
        # Update model weights
        decoder.weight.data = coef_torch
        decoder.bias.data = bias_torch
        
        # Save the trained head
        self.save_head()
        
        return reg

    def save_head(self):
        """Save the trained linear head weights and biases."""
        save_dir = os.path.join("../pretrained_models", f"model_{self.dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        decoder = get_token_prediction_layer(self.model)
        head_state = {
            'weight': decoder.weight.data.cpu(),  # Save as CPU tensors
            'bias': decoder.bias.data.cpu()
        }
        
        save_path = os.path.join(save_dir, "linear_head.pt")
        torch.save(head_state, save_path)
        logger.info(f"Saved linear head to {save_path}")

    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate the model on the given dataset."""
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                inputs = dataset[i]
                # Convert inputs to tensors and move to GPU
                tensor_inputs = {
                    k: torch.tensor(v).to(self.device) if not torch.is_tensor(v) and not isinstance(v, np.ndarray)
                    else torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray)
                    else v.to(self.device)
                    for k, v in inputs.items()
                }
                
                outputs = self.model(**tensor_inputs)
                predictions = outputs.logits
                all_predictions.append(predictions)
                all_targets.append(tensor_inputs["labels"])

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if self.model.num_labels == 1:  # Regression
            targets = targets.squeeze().unsqueeze(-1).float()
            loss = torch.nn.functional.mse_loss(predictions, targets)
            metrics = {"mse": loss.item()}
        else:  # Classification
            loss = torch.nn.functional.cross_entropy(predictions, targets.squeeze())
            pred_classes = predictions.argmax(dim=-1)
            accuracy = (pred_classes == targets.squeeze()).float().mean()
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item()
            }
            
        return metrics
    

from dataset import CustomDataset
from transformers import ViTForImageClassification
import torch
from torch.utils.data import Dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    task_name ="cifar100"
    
    print(f"device:{device}")
    # Create custom datasets
    train_dataset = CustomDataset(task_name = task_name, split = "train")
    test_dataset = CustomDataset(task_name = task_name, split = "test")
    
    print("Datasets Created")

    model_vit = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=100,  # CIFAR-100 has 100 classes
        ignore_mismatched_sizes = True,
    )

    print("model loaded")
    
    # Initialize trainer
    linearheadtrainer = LinearHeadTrainer(
        model=model_vit,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        device=device,
        dataset_name="cifar100"
    )
    
    # Train the linear head
    print("Starting training...")
    linearheadtrainer.train()
    
    # Evaluate
    print("Evaluating...")
    metrics = linearheadtrainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()