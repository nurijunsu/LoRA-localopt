from dataset import CustomDataset
from models import Model_Pretrained
from fine_tuner import FineTuningTrainer
import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, ViTForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Choose a task
task_name = 'sst2' # e.g., "sst2", "qnli", "qqp", "cifar100", "superb_ic"


# Create custom datasets
train_dataset = CustomDataset(task_name=task_name, split="train")

test_dataset = CustomDataset(task_name=task_name, split="test")

print("Datasets Created")

# check_labels(train_dataset.dataset_split, "Train Dataset")
# check_labels(test_dataset.dataset_split, "Test Dataset")

if task_name == "cifar100":
    model_loader = Model_Pretrained("vit",task_name)  
else:
    model_loader = Model_Pretrained("roberta",task_name)  

model = model_loader.get_model()


print("Model Loaded")

project_name=f'global_minimizer_rank_{task_name}'

for tuning_weights in ['one', 'last','all']:
    for lmbda in [2e-1,1e-2,1e-3]:
        #Initialize trainer
        trainer= FineTuningTrainer(                                                                                                                                                          
                model = model,
                train_dataset = train_dataset,
                test_dataset= test_dataset,
                tuning_weights= tuning_weights,    # one, last, or all
                rank = 0,
                lmbda = lmbda,            # Weight decay OR nuclear-norm coefficient
                local_initialization= True,
                num_epochs = 100,
                learning_rate= 1e-2,
                batch_size=256,
                device = device,
                project_name=None
            )

        trainer.train()

        # Evaluate
        print("Evaluating...")
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
