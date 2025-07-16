import torch
from torch import nn
import yaml
from easydict import EasyDict
from datetime import datetime
from os import makedirs
from shutil import copy
from model.network import create_t5_model, setup_training
from data.dataloader import load_data_t5
from model.utils import train_model
import wandb
import gc
import torch.backends.cudnn as cudnn
import os
import time
from datetime import timedelta
from sklearn.metrics import r2_score
from model.network import create_t5_model, setup_training
from data.dataloader import load_data_t5

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
gc.collect()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
print(f'Device: {device}\n')

config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
config.device = device 
original_batch_size = config.batch_size
print(f"Using batch size: {config.batch_size}")

model_type = "t5" 
model_variant = "original"  
model_size = "t5-small"  
model = create_t5_model(config, model_size=model_size)
max_length = 256 
train_loader, valid_loader, test_loader = load_data_t5(config, model_size=model_size, max_length=max_length)
model = model.to(device)
criterion, mae_loss, optimizer, scheduler = setup_training(config, model)

if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
else:
    
    scaler = None

if not config.debug:
    run_name = f't5_{model_size.split("-")[-1]}_{datetime.now().strftime("%m%d_%H%M")}'
    save_dir = f'./checkpoints/{run_name}'
    config.run_name = run_name
    config.save_dir = save_dir
    makedirs(save_dir, exist_ok=True)
    copy('./config.yaml', f'{save_dir}/config.yaml')
    copy('./model/network.py', f'{save_dir}/network.py')

    wandb.init(project='T5_0407', name=run_name, config=config)

best_r2, best_epoch, time_to_best = train_model(
    config, model, train_loader, valid_loader, test_loader,
    criterion, mae_loss, optimizer, scheduler
)

print(f"Best R² Score: {best_r2:.4f}")
print(f"Achieved at epoch: {best_epoch + 1}")
print(f"Time to best model: {timedelta(seconds=int(time_to_best))}")

with open(f'{config.save_dir}/training_efficiency.txt', 'w') as f:
    f.write(f"Best R² Score: {best_r2:.4f}\n")
    f.write(f"Achieved at epoch: {best_epoch + 1}\n")
    f.write(f"Time to best model: {timedelta(seconds=int(time_to_best))}\n")

