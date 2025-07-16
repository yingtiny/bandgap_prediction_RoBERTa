##0701 1604
import torch
import wandb
from os.path import exists
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import seaborn as sns
import time
from datetime import timedelta
from torch.cuda.amp import autocast


wandb.init(project='llama_0423')

def train(model, dataloader, criterion, mae_loss, optimizer, device, scaler, accumulation_steps=8):
    model.train()
    bar = tqdm(dataloader, desc='Train', leave=False, dynamic_ncols=True)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    optimizer.zero_grad() 

    for i, batch in enumerate(bar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device, dtype=torch.float32)

        with autocast():
            outputs = model(ids, mask)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1))
            loss = loss / accumulation_steps 

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps 
        loss_mae = mae_loss(outputs.squeeze(1), targets.squeeze(1))
        total_mae += loss_mae.item()
        rmse = torch.sqrt(loss * accumulation_steps)  
        total_rmse += rmse.item()

        bar.set_postfix(
            loss=f'{total_loss / (i + 1):.3f}',
            mae=f'{total_mae / (i + 1):.3f}',
            rmse=f'{total_rmse / (i + 1):.3f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
        )

        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return total_loss / len(dataloader), total_mae / len(dataloader), total_rmse / len(dataloader)

def test(model, dataloader, criterion, mae_loss, device):

    model.eval()
    bar = tqdm(dataloader, desc='Test', leave=False, dynamic_ncols=True)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    true_labels = []
    predicted_labels = []
    with torch.inference_mode():
        for i, batch in enumerate(bar):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device, dtype=torch.float32)

            outputs = model(ids, mask)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1))
            total_loss += loss.item()
            loss_mae = mae_loss(outputs.squeeze(1), targets.squeeze(1))
            total_mae += loss_mae.item()
            rmse = torch.sqrt(loss)
            total_rmse += rmse.item()
            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(outputs.squeeze(1).cpu().numpy())

            bar.set_postfix(
                loss=f'{total_loss / (i + 1):.3f}',
                mae=f'{total_mae / (i + 1):.3f}',
                rmse=f'{total_rmse / (i + 1):.3f}',
            )

            bar.update()
        bar.close()
   

    return total_loss / len(dataloader), total_mae / len(dataloader), total_rmse / len(dataloader), true_labels, predicted_labels

def train_model(config, model, train_loader, valid_loader, test_loader, criterion, mae_loss, optimizer, scheduler, scaler):
    
    if not config.debug:
        wandb.init(project='llama_0423', name=config.run_name)
    
    loss_over_epochs_train = []
    loss_over_epochs_valid = []
    loss_over_epochs_test = []
    mae_over_epochs_train = []
    mae_over_epochs_valid = []
    mae_over_epochs_test = []
    rmse_over_epochs_train = []
    rmse_over_epochs_valid = []
    rmse_over_epochs_test = []
    r2_over_epochs = []
    best_loss = float('inf')
    best_r2 = float('-inf')
    best_epoch = -1
    
    start_time = time.time() 
    best_model_time = None

    for epoch in range(config.epochs):
        train_loss, train_mae, train_rmse = train(model, train_loader, criterion, mae_loss, optimizer, config.device, scaler)
        loss_over_epochs_train.append(train_loss)
        mae_over_epochs_train.append(train_mae)
        rmse_over_epochs_train.append(train_rmse)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.3f}\tLR: {curr_lr:.3e}')
        
        valid_loss, valid_mae, valid_rmse, valid_true, valid_pred = test(model, valid_loader, criterion, mae_loss, config.device)
        loss_over_epochs_valid.append(valid_loss)
        mae_over_epochs_valid.append(valid_mae)
        rmse_over_epochs_valid.append(valid_rmse)
        print(f'Epoch {epoch+1}/{config.epochs} - Validation Loss: {valid_loss:.3f}')

        test_loss, test_mae, test_rmse, test_true, test_pred = test(model, test_loader, criterion, mae_loss, config.device)
        loss_over_epochs_test.append(test_loss)
        mae_over_epochs_test.append(test_mae)
        rmse_over_epochs_test.append(test_rmse)
        print(f'Epoch {epoch+1}/{config.epochs} - Test Loss: {test_loss:.3f}')

        r2 = calculate_r2_score(model, test_loader)
        r2_over_epochs.append(r2)
        print(f'Epoch {epoch+1}/{config.epochs} - R² Score: {r2:.3f}')

        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch
            best_model_time = time.time() - start_time
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'train_mae': train_mae,
                'valid_mae': valid_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'valid_rmse': valid_rmse,
                'test_rmse': test_rmse,
                'r2_score': r2,
                'time_to_best': best_model_time
            }, f'{config.save_dir}/best_model.pt')
            print(f'New best model saved. R² Score: {r2:.3f}, Time: {timedelta(seconds=best_model_time)}')

        if not config.debug:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'train_mae': train_mae,
                'valid_mae': valid_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'valid_rmse': valid_rmse,
                'test_rmse': test_rmse,
                'r2_score': r2,
                'lr': curr_lr,
            })

        scheduler.step(valid_loss)
    
    total_training_time = time.time() - start_time
    
    final_results = {
        'Train Loss': train_loss,
        'Validation Loss': valid_loss,
        'Test Loss': test_loss,
        'Train MAE': train_mae,
        'Validation MAE': valid_mae,
        'Test MAE': test_mae,
        'Train RMSE': train_rmse,
        'Validation RMSE': valid_rmse,
        'Test RMSE': test_rmse,
        'Best R2 Score': best_r2,
        'Time to Best Model': str(timedelta(seconds=best_model_time)),
        'Total Training Time': str(timedelta(seconds=total_training_time))
    }
    wandb.log(final_results)

    print(f'Best R² Score: {best_r2:.3f}')
    print(f'Time to Best Model: {timedelta(seconds=best_model_time)}')
    print(f'Total Training Time: {timedelta(seconds=total_training_time)}')

    uncertainty_metrics, final_test_loss = calculate_uncertainty(model, test_loader, criterion, mae_loss, config.device)
    for metric, value in uncertainty_metrics.items():
        if isinstance(value, tuple):
            print(f"{metric}: {value[0]:.4f} ± {value[1]:.4f}")
            wandb.log({f"{metric}": value[0], f"{metric}_uncertainty": value[1]})
        else:
            print(f"{metric}: {value:.4f}")
            wandb.log({f"{metric}": value})
    wandb.log({'Final Test Loss': final_test_loss})

    plt.figure(figsize=(10,5))
    plt.title("Training Loss vs. Epoch")
    plt.plot(loss_over_epochs_train, label="Train Loss")
    plt.plot(loss_over_epochs_valid, label="Valid Loss")
    plt.plot(loss_over_epochs_test, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('plot_loss.png')
    wandb.log({"loss_plot": wandb.Image('plot_loss.png')})
    plt.close()

    if not config.debug:
        wandb.finish()
        save_results(config.run_name, best_r2, loss_over_epochs_train[-1], loss_over_epochs_valid[-1], loss_over_epochs_test[-1], config.optim.lr, config.batch_size, config.epochs, best_model_time, total_training_time)

    return best_r2, best_epoch, best_model_time

def calculate_r2_score(model, dataloader):
    model.eval()
    true_labels = []
    predicted_labels = []
    device = next(model.parameters()).device 
    with torch.no_grad():
        for data in dataloader:
            inputs, mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
            outputs = model(inputs, mask)  
            true_labels.extend(labels.tolist())
            predicted_labels.extend(outputs.squeeze().tolist())
    r2 = r2_score(true_labels, predicted_labels)
    return r2

def calculate_uncertainty(model, test_loader, criterion, mae_loss, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    test_loss, test_mae, test_rmse, true_labels, predicted_labels = test(model, test_loader, criterion, mae_loss, device)
    

    r2 = r2_score(true_labels, predicted_labels)
    

    n = len(true_labels)
    rmse_uncertainty = test_rmse / np.sqrt(2 * (n - 1))
    mae_uncertainty = test_mae * np.sqrt(np.pi / (2 * n))
    r2_uncertainty = np.sqrt((4 * r2 * (1 - r2)**2) / (n - 2)) 
    

    residuals = np.array(predicted_labels) - np.array(true_labels)
    prediction_std = np.std(residuals)
    
    uncertainty_metrics = {
        'RMSE': (test_rmse, rmse_uncertainty),
        'MAE': (test_mae, mae_uncertainty),
        'R2': (r2, r2_uncertainty),
        'Prediction Std': prediction_std
    }
    
    return uncertainty_metrics, test_loss



def save_results(run_name, r2, final_train_loss, final_valid_loss, final_test_loss, lr, batch_size, epochs, best_model_time, total_training_time):
    if not exists('./results.csv'):
        with open('results.csv', 'w') as f:
            f.write('Run Name,Epochs,LR,Batch Size,Train Loss,Valid Loss,Test Loss,R²,Time to Best Model,Total Training Time\n')
    with open('./results.csv', 'a') as f:
        f.write(f'{run_name},{epochs},{lr},{batch_size},{final_train_loss},{final_valid_loss},{final_test_loss},{r2},{timedelta(seconds=best_model_time)},{timedelta(seconds=total_training_time)}\n')

