import torch
from torch import nn
import yaml
from easydict import EasyDict
from datetime import datetime
from os import makedirs
from shutil import copy
from data.dataloader import load_data
from model.network import create_model, setup_training
from model.utils import train_model
import wandb
import gc
import torch.backends.cudnn as cudnn
import os
import time
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()
gc.collect()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not hasattr(config, 'gradient_accumulation_steps'):
    config.gradient_accumulation_steps = 16
if not hasattr(config, 'use_mixed_precision'):
    config.use_mixed_precision = True
if not hasattr(config, 'sequence_length'):
    config.sequence_length = 512

scaler = GradScaler() if torch.cuda.is_available() else None

model = create_model(config).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)
train_loader, valid_loader, test_loader = load_data(config)
criterion, mae_loss, optimizer, scheduler = setup_training(config, model)

if config.optim.scheduler == "onecycle":

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5, 
        total_steps=config.epochs * len(train_loader) // config.gradient_accumulation_steps,
        pct_start=0.1,  
        anneal_strategy='cos'
    )

if not config.debug:
    run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
    save_dir = f'./checkpoints/{run_name}'
    config.run_name = run_name
    config.save_dir = save_dir
    makedirs(save_dir, exist_ok=True)
    copy('./config.yaml', f'{save_dir}/config.yaml')
    copy('./model/network.py', f'{save_dir}/network.py')
    wandb.init(project='llama_0423', name=run_name, config=config)

best_r2, best_epoch, time_to_best = train_model(
    config, model, train_loader, valid_loader, test_loader, criterion, mae_loss, optimizer, scheduler, scaler
)

print(f"Best R² Score: {best_r2:.4f}")
print(f"Achieved at epoch: {best_epoch + 1}")
print(f"Time to best model: {timedelta(seconds=int(time_to_best))}")

with open(f'{config.save_dir}/training_efficiency.txt', 'w') as f:
    f.write(f"Best R² Score: {best_r2:.4f}\n")
    f.write(f"Achieved at epoch: {best_epoch + 1}\n")
    f.write(f"Time to best model: {timedelta(seconds=int(time_to_best))}\n")

