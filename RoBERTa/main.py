import torch
from torch import nn
import yaml
from easydict import EasyDict
from datetime import datetime
from os import makedirs
from shutil import copy
from data.dataloader import load_data
from model.network_attention import create_model, setup_training
from model.utils import train_model
import wandb
import gc
import torch.backends.cudnn as cudnn
import os
import time
from datetime import timedelta

torch.cuda.empty_cache()


gc.collect()


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
model = create_model(config)
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)

train_loader, valid_loader, test_loader = load_data(config)
criterion, mae_loss, optimizer, scheduler = setup_training(config, model)

if not config.debug:
    run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
    save_dir = f'./checkpoints/{run_name}'
    config.run_name = run_name
    config.save_dir = save_dir

    makedirs(save_dir, exist_ok=True)
    copy('./config.yaml', f'{save_dir}/config.yaml')
    copy('./model/network.py', f'{save_dir}/network.py')
    wandb.init(project='Aflow_0910', name=run_name, config=config)


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


