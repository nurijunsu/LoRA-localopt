import os
import shutil
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
    RobertaForSequenceClassification,
    ViTForImageClassification
)
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import transformers
from transformers.utils import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from dataset import CustomDataset
from typing import Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set the logging level



def get_token_prediction_layer(model):
    """Extract the final prediction layer based on model type."""
    if isinstance(model, transformers.RobertaForSequenceClassification):
        # 'out_proj' is just the final linear layer. There's also a 'dense' layer in .classifier
        return model.classifier.out_proj
    elif isinstance(model, transformers.ViTForImageClassification):
        return model.classifier
    elif isinstance(model, transformers.Wav2Vec2ForSequenceClassification):
        return model.classifier.out_proj
    else:
        raise NotImplementedError(f"Model type {model.__class__} is not supported.")

def extract_features(model, *args, **kwargs):
    """
    Extract features from the layer before the final prediction layer.
    For Roberta, that's right before 'out_proj'—which means the hidden
    state after 'dense' + 'tanh' + dropout.
    """
    features = {}
    def hook(model_, input_, output_):
        # input_[0] is the representation after the 'dense' + 'tanh' + dropout layer for RoBERTa
        features["features"] = input_[0].detach()

    get_token_prediction_layer(model).register_forward_hook(hook)
    model.forward(*args, **kwargs)
    return features["features"]


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
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

    def collect_features(self, dataset, batch_size=512):
        features = []
        targets = []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move inputs to the correct device
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Extract features
                feature = extract_features(self.model, pixel_values=pixel_values)
                features.append(feature.cpu())
                targets.append(labels.cpu())

        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)

        return features.numpy(), targets.numpy()

    def train_roberta_class_head(
        self,
        epochs: int = 10,
        lr: float = 5e-4,
        batch_size: int = 256
    ):
        """
        Standard PyTorch training loop for RobertaForSequenceClassification,
        training the entire two-layer MLP head (dense + out_proj).
        By default, we freeze the encoder and only train the classifier.
        """
        # Freeze the entire backbone except the classification head
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Create a DataLoader
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

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Simple Adam optimizer
        optimizer = torch.optim.AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-2
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader):
                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"[Epoch {epoch+1}] loss: {avg_loss:.4f}")

        # (Optional) Save the classification head after training
        self.save_head()

    def train(self):
        """
        Main entry point:
        - If model is RobertaForSequenceClassification, do a standard
          training of its two-layer classification head (no scikit-learn).
        - Otherwise (e.g. ViT), continue with the feature-extraction + logistic/regression approach.
        """
        if isinstance(self.model, transformers.RobertaForSequenceClassification):
            logger.info("Detected RobertaForSequenceClassification. Training its two-layer MLP head directly.")
            self.train_roberta_class_head(epochs=20, lr=1e-3, batch_size=128)
            return None  # We don't return a scikit-learn model
        else:
            logger.info("Starting to collect features for training dataset (non-RoBERTa)")
            features, targets = self.collect_features(self.train_dataset)
            print(f"Features shape: {features.shape}")  # Expected: (N, D)
            print(f"Targets shape: {targets.shape}")    # Expected: (N,)
            
            # Decide if regression or classification
            if self.model.num_labels == 1:  # Regression
                targets = torch.tensor(targets).squeeze().unsqueeze(-1).float()
                reg = LinearRegression().fit(features, targets)
            else:  # Classification
                logger.info("Fitting regression model")
                reg = LogisticRegression(
                    C=1.0,
                    solver="saga",
                    max_iter=200,
                    multi_class="multinomial",
                    n_jobs=-1,
                    random_state=0,
                    verbose=1  # Add verbosity
                ).fit(features, targets)

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
        """Save the trained classification head weights and biases."""
        save_dir = os.path.join("../pretrained_models", f"model_{self.dataset_name}")
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(self.model, transformers.RobertaForSequenceClassification):
            # Save both layers of the classification head
            head_state = {
                'dense.weight': self.model.classifier.dense.weight.data.cpu(),
                'dense.bias': self.model.classifier.dense.bias.data.cpu(),
                'out_proj.weight': self.model.classifier.out_proj.weight.data.cpu(),
                'out_proj.bias': self.model.classifier.out_proj.bias.data.cpu()
            }
        else:
            # For other models (e.g. ViT), continue saving the final layer only
            decoder = get_token_prediction_layer(self.model)
            head_state = {
                'weight': decoder.weight.data.cpu(),
                'bias': decoder.bias.data.cpu()
            }

        save_path = os.path.join(save_dir, "classification_head.pt")
        torch.save(head_state, save_path)
        logger.info(f"Saved classification head to {save_path}")


    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
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

def check_labels(dataset, dataset_name="Dataset"):
    labels = dataset["labels"]
    unique_labels = set(labels)
    print(f"Unique labels in {dataset_name}: {unique_labels}")
    assert all(isinstance(label, int) for label in labels), "All labels must be integers."
    assert all(0 <= label < 2 for label in labels), "Labels must be 0 or 1 for binary classification."

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    from models import Model_Pretrained
    # Choose a task
    for task_name in ["cifar100"]:  # e.g., "sst2", "qnli", "qqp", "cifar100", "superb_ic"
        # Create custom datasets
        train_dataset = CustomDataset(task_name=task_name, split="train")
        test_dataset = CustomDataset(task_name=task_name, split="test")
        
        print("Datasets Created")

        # check_labels(train_dataset.dataset_split, "Train Dataset")
        # check_labels(test_dataset.dataset_split, "Test Dataset")

        if task_name == "cifar100":
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100, ignore_mismatched_sizes=True)
        else:
            # # # Example: Using RobertaForSequenceClassification with 2 labels
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2, ignore_mismatched_sizes=True)
        print("Model loaded")

        
        model_loader = Model_Pretrained("vit",task_name)  
        model = model_loader.get_model()
        #Initialize trainer
        linearheadtrainer = LinearHeadTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            device=device,
            dataset_name=task_name
        )
        
        # # Train
        # print("Starting training...")
        # linearheadtrainer.train() 
        
        # Evaluate
        print("Evaluating...")
        metrics = linearheadtrainer.evaluate()
        print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()