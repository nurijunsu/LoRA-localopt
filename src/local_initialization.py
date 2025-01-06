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
for task_name in ['sst2', 'qnli', 'qqp','cifar100']:

    # Create custom datasets
    train_dataset = CustomDataset(task_name=task_name, split="train")

    test_dataset = CustomDataset(task_name=task_name, split="test")

    print("Datasets Created")

    project_name=f'nonlocal_initialization_{task_name}'
    tuning_weights = 'all'# one, last, or all
    rank = 16
    lmbda = 1e-3
    local_initialization = True 

    if task_name == "cifar100":
        model_loader = Model_Pretrained("vit",task_name,  fine_tuned=True, rank=rank, tuning_weights=tuning_weights, local_init=local_initialization)  
    else:
        model_loader = Model_Pretrained("roberta",task_name, fine_tuned=True, rank=rank, tuning_weights=tuning_weights, local_init = local_initialization)  

    model = model_loader.get_model()

    trainer= FineTuningTrainer(                                                                                                                                                          
            model = model,
            train_dataset = train_dataset,
            test_dataset= test_dataset,
            tuning_weights= tuning_weights,    
            rank = rank,
            lmbda = lmbda,            # Weight decay OR nuclear-norm coefficient
            local_initialization= local_initialization,
            num_epochs = 200,
            learning_rate= 5e-3,
            batch_size=384,
            device = device,
            project_name=project_name, 
            log_dir = f'../logs/local_init/{task_name}/tuned={tuning_weights}_LoRA={rank}_lmbda={lmbda}_local={local_initialization}'
        )

    trainer.train()