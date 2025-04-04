import os
import shutil
from datasets import load_dataset, DatasetDict, load_from_disk
import torch, torchvision
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor
)


class CustomDataset(torch.utils.data.Dataset):
    """
    A unified dataset class that:
    - Loads specified datasets (SST-2, QNLI, QQP for NLP; CIFAR-100 for vision; SUPERB IC for audio)
    - Adds a prompt for MLM-style tasks for NLP
    - Performs tokenization/feature-extraction so the resulting dataset can be directly fed into the model
    - Provides train/test splits
    - Saves processed datasets locally to ../data to avoid repeated downloads/preprocessing
    """
    def __init__(self, task_name, split="train", sample_ratio=1.0, random_seed=42):
        """
        Args:
            task_name (str): one of "sst2", "qnli", "qqp", "cifar100", "superb_ic"
            split (str): "train" or "test"
            sample_ratio (float): Fraction of dataset to use (default: 1.0 for full dataset)
            random_seed (int): Seed for random sampling
        """
        self.task_name = task_name
        self.split = split
        self.data_dir = "../data"
        self.sample_ratio = sample_ratio
        self.random_seed = random_seed

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Paths
        self.processed_path = os.path.join(self.data_dir, f"{self.task_name}_processed")

        # Model-specific resources
        # We'll assume:
        # - NLP: RoBERTa-base
        # - Vision: ViT-base-patch16-224
        # - Audio: wav2vec2-base
        self.roberta_name = "roberta-base"
        self.vit_name = "google/vit-base-patch16-224"
        self.wav2vec2_name = "facebook/wav2vec2-base"

        print("Preparing dataset")
        self.dataset = self._prepare_dataset()
        #self.dataset.save_to_disk(self.processed_path)
        print("Dataset prepared")
        # Now select the split
        if self.split not in self.dataset:
            raise ValueError(f"Split {self.split} not found in dataset {self.task_name}. Available splits: {list(self.dataset.keys())}")
        self.dataset_split = self.dataset[self.split]
        self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _prepare_dataset(self):
        # Load raw datasets from HuggingFace Hub
        # For NLP tasks: SST-2, QNLI, QQP
        # For Vision: CIFAR-100
        # For Audio: SUPERB IC
        if self.task_name == "sst2":
            print("Loading dataset")
            raw = load_dataset("glue", "sst2")
            print("Dataset loaded")
            
            # Take first N% if sample_ratio < 1.0
            if self.sample_ratio < 1.0:
                print(f"Using first {int(self.sample_ratio * 100)}% of data")
                # Calculate how many examples to keep
                train_size = int(len(raw["train"]) * self.sample_ratio)
                val_size = int(len(raw["validation"]) * self.sample_ratio)
                
                # Select the first N examples
                raw = DatasetDict({
                    "train": raw["train"].select(range(train_size)),
                    "validation": raw["validation"].select(range(val_size))
                })
                print(f"Train size: {len(raw['train'])}, Validation size: {len(raw['validation'])}")
            
            # We'll create a prompt:
            # "The sentiment of this sentence is <mask>. Sentence: {sentence}"
            # Labels: 0 = negative, 1 = positive
            tokenizer = AutoTokenizer.from_pretrained(self.roberta_name, use_fast=True)
            def preprocess_sst2(example):
                sentence = example["sentence"]
                prompt_text = f"The sentiment of this sentence is <mask>. Sentence: {sentence}"
                tokenized = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=128)
                tokenized["labels"] = example["label"]
                return tokenized
            processed = raw.map(preprocess_sst2, batched=False, remove_columns=raw["train"].column_names)
           
        elif self.task_name == "qnli":
            raw = load_dataset("glue", "qnli")
            
            # Take first N% if sample_ratio < 1.0
            if self.sample_ratio < 1.0:
                print(f"Using first {int(self.sample_ratio * 100)}% of data")
                # Calculate how many examples to keep
                train_size = int(len(raw["train"]) * self.sample_ratio)
                val_size = int(len(raw["validation"]) * self.sample_ratio)
                
                # Select the first N examples
                raw = DatasetDict({
                    "train": raw["train"].select(range(train_size)),
                    "validation": raw["validation"].select(range(val_size))
                })
                
            # QNLI: Given question and sentence, is the sentence an answer to the question?
            # Labels: 0 = entailed, 1 = not_entailment
            # Prompt: "Given the question: {question}, does the following sentence answer it <mask>? {sentence}"
            tokenizer = AutoTokenizer.from_pretrained(self.roberta_name, use_fast=True)
            def preprocess_qnli(example):
                question = example["question"]
                sentence = example["sentence"]
                prompt_text = f"Given the question: {question}, does the following sentence answer it <mask>? {sentence}"
                tokenized = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=128)
                tokenized["labels"] = example["label"]
                return tokenized
            processed = raw.map(preprocess_qnli, batched=False, remove_columns=raw["train"].column_names)

        elif self.task_name == "qqp":
            raw = load_dataset("glue", "qqp")
            
            # Take first N% if sample_ratio < 1.0
            if self.sample_ratio < 1.0:
                print(f"Using first {int(self.sample_ratio * 100)}% of data")
                # Calculate how many examples to keep
                train_size = int(len(raw["train"]) * self.sample_ratio)
                val_size = int(len(raw["validation"]) * self.sample_ratio)
                
                # Select the first N examples
                raw = DatasetDict({
                    "train": raw["train"].select(range(train_size)),
                    "validation": raw["validation"].select(range(val_size))
                })
            
            tokenizer = AutoTokenizer.from_pretrained(self.roberta_name, use_fast=True)
            
            def preprocess_qqp(example):
                q1 = example["question1"]
                q2 = example["question2"]
                prompt_text = f"Are these two questions asking the same thing <mask>? Q1: {q1}, Q2: {q2}"
                tokenized = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=128)
                tokenized["labels"] = example["label"]
                return tokenized
            processed = raw.map(preprocess_qqp, batched=False, remove_columns=raw["train"].column_names)

        elif self.task_name == "cifar100":
            processed = load_dataset("cifar100")
        
        elif self.task_name =="beans":
            processed = load_dataset("beans")

        elif self.task_name =="food101":
            train = load_dataset("food101", split="train[:10%]")
            validation = load_dataset("food101", split="validation[:10%]")
            processed = DatasetDict({
                "train": train,
                "validation": validation
            })            
            
        elif self.task_name == "superb_ic":
            raw = load_dataset("superb", "ic")
            feature_extractor = AutoProcessor.from_pretrained(self.wav2vec2_name)
            
            def preprocess_superb_ic(example):
                audio = example["audio"]["array"]
                inputs = feature_extractor(
                    audio,
                    sampling_rate=example["audio"]["sampling_rate"],
                    return_tensors="np"  # Use NumPy arrays directly
                )
                return {
                    "input_values": inputs["input_values"][0],  # Already a NumPy array
                    "labels": example["label"]
                }
            processed = raw.map(preprocess_superb_ic, batched=False, remove_columns=raw["train"].column_names)

        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # We only need train/test splits. Some datasets have validation sets; 
        # We'll consider 'validation' as 'test' if no test split available.
        # Check splits
        
        split_names = processed.keys()
        if "validation" in split_names:
            processed = DatasetDict({
                "train": processed["train"],
                "test": processed["validation"]
            })
        else:
            processed = DatasetDict({
                "train": processed["train"],
                "test": processed["test"]
            })



        return processed

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        item = self.dataset_split[idx] 

        if self.task_name == "cifar100":
            pixel_values = self.image_transform(item["img"])
            labels = item["fine_label"]
            
            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
        elif self.task_name =="beans":
            pixel_values = self.image_transform(item["image"])
            labels = item["labels"]
            
            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
        elif self.task_name in ["food101"]:
            pixel_values = self.image_transform(item["image"])
            labels = item["label"]
            
            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
        else:
            # For NLP or other tasks:
            return item

    
if __name__ == "__main__":
    from datasets import load_dataset

    raw = load_dataset("cifar100")
    print(type(raw["train"][0]["img"]))  # Should ideally be <class 'PIL.Image.Image'>
