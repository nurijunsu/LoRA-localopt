import os
import shutil
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor
)
import torch
from torchvision import transforms
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    """
    A unified dataset class that:
    - Loads specified datasets (SST-2, QNLI, QQP for NLP; CIFAR-100 for vision; SUPERB IC for audio)
    - Adds a prompt for MLM-style tasks for NLP
    - Performs tokenization/feature-extraction so the resulting dataset can be directly fed into the model
    - Provides train/test splits
    - Saves processed datasets locally to ../data to avoid repeated downloads/preprocessing
    """
    def __init__(self, task_name, split="train"):
        """
        Args:
            task_name (str): one of "sst2", "qnli", "qqp", "cifar100", "superb_ic"
            split (str): "train" or "test"
        """
        self.task_name = task_name
        self.split = split
        self.data_dir = "../data"

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

        self.dataset = self._prepare_dataset()
        #self.dataset.save_to_disk(self.processed_path)

        # Now select the split
        if self.split not in self.dataset:
            raise ValueError(f"Split {self.split} not found in dataset {self.task_name}. Available splits: {list(self.dataset.keys())}")
        self.dataset_split = self.dataset[self.split]
        self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _prepare_dataset(self):
        # Load raw datasets from HuggingFace Hub
        # For NLP tasks: SST-2, QNLI, QQP
        # For Vision: CIFAR-100
        # For Audio: SUPERB IC
        if self.task_name == "sst2":
            raw = load_dataset("glue", "sst2")
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
        else:
            # For NLP or other tasks:
            return item

    
if __name__ == "__main__":
    from datasets import load_dataset

    raw = load_dataset("cifar100")
    print(type(raw["train"][0]["img"]))  # Should ideally be <class 'PIL.Image.Image'>
