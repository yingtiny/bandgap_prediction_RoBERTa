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



wandb.init(project='Aflow_0910')

def train(model, dataloader, criterion, mae_loss, optimizer, device):
    model.train()
    bar = tqdm(dataloader, desc='Train', leave=False, dynamic_ncols=True)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    # all_embeddings = [] 
    for i, batch in enumerate(bar):
        
        

        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(ids, mask)#.module
        loss = criterion(outputs.squeeze(1), targets.squeeze(1))
        total_loss += loss.item()
        loss_mae = mae_loss(outputs.squeeze(1), targets.squeeze(1))
        total_mae += loss_mae.item()

        rmse = torch.sqrt(loss)
        total_rmse += rmse.item()


        bar.set_postfix(
            loss=f'{total_loss / (i + 1):.3f}',
            mae=f'{total_mae / (i + 1):.3f}',
            rmse=f'{total_rmse / (i + 1):.3f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
        )
        loss.backward()
        optimizer.step()

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
def train_model(config, model, train_loader, valid_loader, test_loader, criterion, mae_loss, optimizer, scheduler):
    if not config.debug:
        wandb.init(project='Aflow_0910', name=config.run_name)
    
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
    
    start_time = time.time()  # Record start time
    best_model_time = None

    for epoch in range(config.epochs):
        train_loss, train_mae, train_rmse = train(model, train_loader, criterion, mae_loss, optimizer, config.device)
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
    
    # Final log after training completes
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

    plot_priority_plot(model, test_loader)
    plot_hexbin_parity(model, test_loader)

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
        save_results(config.run_name, best_r2, loss_over_epochs_train[-1], loss_over_epochs_valid[-1], loss_over_epochs_test[-1], config.lr, config.batch_size, config.epochs, best_model_time, total_training_time)

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
    r2_uncertainty = np.sqrt((4 * r2 * (1 - r2)**2) / (n - 2))  # 近似值
    

    residuals = np.array(predicted_labels) - np.array(true_labels)
    prediction_std = np.std(residuals)
    
    uncertainty_metrics = {
        'RMSE': (test_rmse, rmse_uncertainty),
        'MAE': (test_mae, mae_uncertainty),
        'R2': (r2, r2_uncertainty),
        'Prediction Std': prediction_std
    }
    
    return uncertainty_metrics, test_loss

def plot_hexbin_parity(model, dataloader):
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
    
    np.savez('./parity_data.npz', true_labels = np.array(true_labels),
              predicted_labels = np.array(predicted_labels))
    fig, ax = plt.subplots(figsize=(10, 8))    

    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    hb = ax.hexbin(true_labels, predicted_labels, gridsize=50, cmap='viridis', mincnt=1)
    

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count', fontsize=14)

    ax.plot([0, 5], [0, 5], 'k--', linewidth=1.5, alpha=0.75, zorder=0)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.xlabel('Aflow Bandgap (eV)', fontsize=16)
    plt.ylabel('Predicted Bandgap (eV)', fontsize=16)
    plt.title('Roberta model based on description', fontsize=16)


    
    # Add R-squared value
    r2 = r2_score(true_labels, predicted_labels)
    mae = np.mean(np.abs(np.array(true_labels) - np.array(predicted_labels)))
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predicted_labels))**2))
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add border to all sides of the plot and set ticks to the outside
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    ax.tick_params(top=False, right=False)
    

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    

    
    plt.tight_layout()

    plt.savefig('hexbin_parity_plot.png', dpi=300, bbox_inches='tight')
    wandb.log({"hexbin_parity_plot": wandb.Image('hexbin_parity_plot.png')})
    plt.close()

def calculate_metrics_with_uncertainty(true_labels, predicted_labels):
    mse = np.mean((np.array(true_labels) - np.array(predicted_labels))**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(true_labels) - np.array(predicted_labels)))
    r2 = r2_score(true_labels, predicted_labels)
    

    n = len(true_labels)
    rmse_uncertainty = rmse / np.sqrt(2 * (n - 1))
    mae_uncertainty = mae * np.sqrt(np.pi / (2 * n))
    r2_uncertainty = np.sqrt((4 * r2 * (1 - r2)**2) / (n - 2))  # Approximate

    return {
        'RMSE': (rmse, rmse_uncertainty),
        'MAE': (mae, mae_uncertainty),
        'R2': (r2, r2_uncertainty)
    }

def plot_priority_plot(model, dataloader):
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
    plt.scatter(true_labels, predicted_labels, alpha=0.3)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], '--k', linewidth=0.5)
    plt.xlabel('Aflow Bandgap')
    plt.ylabel('Predicted Bandgap')
    plt.title('Roberta model based on description')
    plt.savefig('plot.png')
    wandb.log({"priority_plot": wandb.Image('plot.png')})
    plt.close()

def save_results(run_name, r2, final_train_loss, final_valid_loss, final_test_loss, lr, batch_size, epochs, best_model_time, total_training_time):
    if not exists('./results.csv'):
        with open('results.csv', 'w') as f:
            f.write('Run Name,Epochs,LR,Batch Size,Train Loss,Valid Loss,Test Loss,R²,Time to Best Model,Total Training Time\n')
    with open('./results.csv', 'a') as f:
        f.write(f'{run_name},{epochs},{lr},{batch_size},{final_train_loss},{final_valid_loss},{final_test_loss},{r2},{timedelta(seconds=best_model_time)},{timedelta(seconds=total_training_time)}\n')

def save_attention_scores(attentions, epoch, phase='train'):
    # Save attention scores to a file
    attention_file = f'attention_scores_{phase}_epoch_{epoch}.pkl'
    with open(attention_file, 'wb') as f:
        pickle.dump(attentions, f)
    
    # Upload to WandB
    wandb.save(attention_file)


def analyze_embeddings(embeddings):
    # Ensure embeddings shape is correct
    embeddings = embeddings.view(-1, embeddings.size(-1))  # Flatten for PCA

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    plt.title('PCA of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('plot_pca.png')
    wandb.log({"embedding_pca_plot": wandb.Image('plot_pca.png')})
    plt.close()