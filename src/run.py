from dataset import CustomDataset
from models import Model_Pretrained
from fine_tuner import FineTuningTrainer
import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, ViTForImageClassification
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, help='device')
    parse.add_argument('--task_name', default='sst2', type=str, help='fine tuning task')
    parse.add_argument('--lmbda', default=0.01, type=float, help='weight decay parameter')
    parse.add_argument('--tuning_weights', default='all', type=str, help='finetuning type')
    parse.add_argument('--learning_rate', default=0.001, type=float, help ='learning rate')
    parse.add_argument('--rank', default=8, type=int, help ='rank')
    parse.add_argument('--local_init', default='LargeRandom', type=str, help ='local initialization')
    parse.add_argument('--Scheduler', default='None', type=str, help ='Learning Rate Scheduler')
    parse.add_argument('--sample_ratio', default=1.0, type=float, help ='sample ratio')
    args = parse.parse_args()
    return args

#python run.py --device cuda:0 --task_name sst2 --lmbda 0.01 --tuning_weights all --learning_rate 0.005 --rank 8 --local_init LargeRandom --Scheduler CosineDecay
torch.manual_seed(42)

args = parse_args()
task_name = args.task_name
# ["sst2", "qnli", "qqp", "cifar100", "superb_ic"]
lmbda = args.lmbda
# [0.1, 0.01 0.001, 0.0001]
tuning_weights = args.tuning_weights
# [one, last, all]
learning_rate = args.learning_rate
rank = args.rank
local_init = args.local_init
Scheduler = args.Scheduler

device = args.device
print(f"Using device: {device}")

# Create custom datasets
train_dataset = CustomDataset(task_name=task_name, split="train", sample_ratio=args.sample_ratio)

test_dataset = CustomDataset(task_name=task_name, split="test", sample_ratio=args.sample_ratio)

print("Datasets Created")

# check_labels(train_dataset.dataset_split, "Train Dataset")
# check_labels(test_dataset.dataset_split, "Test Dataset")


if task_name in ["cifar100", "beans", "food101"]:
    model_loader = Model_Pretrained("vit",task_name,  fine_tuned=True, rank=rank, tuning_weights=tuning_weights, local_init=local_init)  
else:
    model_loader = Model_Pretrained("roberta",task_name, fine_tuned=True, rank=rank, tuning_weights=tuning_weights, local_init = local_init)  

model = model_loader.get_model()


print("Model Loaded")

trainer= FineTuningTrainer(                                                                                                                                                          
        model = model,
        train_dataset = train_dataset,
        test_dataset= test_dataset,
        tuning_weights= tuning_weights,    
        rank = rank,
        lmbda = lmbda,            # Weight decay OR nuclear-norm coefficient
        L2_reg = False,
        local_initialization= local_init, #True, LargeRandom, Ortho
        num_epochs = 500,
        save_epoch = 100,
        learning_rate= learning_rate,
        batch_size=len(train_dataset),
        device = device,
        optimizer = "SGD",
        proximal_gradient= True,
        project_name='Rebuttal_Sameglobalmin',
        lr_scheduler= Scheduler, #ReduceLROnPlateu, CosineAnnealing, CosineDecay, LinearWarmup
        run_name = "tryout3_fasttrain"
    )

trainer.train()

# Evaluate
print("Evaluating...")
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")