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

parity_data = np.load('./parity_data.npz')
true_labels = parity_data['true_labels']
predicted_labels = parity_data['predicted_labels']


def plot_hexbin_parity(true_labels, predicted_labels):




    fig, ax = plt.subplots(figsize=(10, 8))    
    # sns.set_style("ticks")
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    # Hexbin plot
    # scatter = ax.scatter(true_labels, predicted_labels, alpha=0.5, s=10)

    # # Plot the ideal line
    # ax.plot([0, 5], [0, 5], 'k--', linewidth=1.5, alpha=0.75, zorder=0)
    # hb = ax.hexbin(true_labels, predicted_labels, gridsize=50, cmap='viridis', 
    #                vmin=1, vmax=100, bins='log', extent=[0, 5, 0, 5])
    hb = ax.hexbin(true_labels, predicted_labels, gridsize=50, cmap='viridis', 
                  bins='log')

    # Add colorbar
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.tick_params(labelsize=28)
    cb.set_label('Count', fontsize=28)
    

    # Plot the ideal line
    ax.plot([0, 5], [0, 5], 'k--', linewidth=1.5, alpha=0.75, zorder=0)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.xlabel('Aflow Bandgap (eV)', fontsize=40)
    plt.ylabel('RoBERTa$_{\mathrm{string}}$\nBandgap (eV)', fontsize=40)
    # plt.title('Roberta model based on description', fontsize=40)
    

    ax.tick_params(axis='both', which='major', labelsize=40)
    
    # Add R-squared value
    r2 = r2_score(true_labels, predicted_labels)
    mae = np.mean(np.abs(np.array(true_labels) - np.array(predicted_labels)))
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predicted_labels))**2))
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV'
    # plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
    #          verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add border to all sides of the plot and set ticks to the outside
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    ax.tick_params(top=False, right=False)
    
    # Adjust tick marks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    

    plt.tight_layout()
    plt.savefig('hexbin_parity_plot_0918_1.png', dpi=300, bbox_inches='tight')
    plt.close()





def plot_priority_plot(model, dataloader):

    plt.scatter(true_labels, predicted_labels)#, alpha=0.3)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], '--k', linewidth=0.5)
    plt.xlabel('Aflow Bandgap')
    plt.ylabel('Predicted Bandgap')
    plt.title('Roberta model based on description')
    plt.savefig('parity_plot_0918_1.png')

    plt.close()

#hexbin_parity_plot_0918_1.png'
parity_data = np.load('./parity_data.npz')
true_labels = parity_data['true_labels']
predicted_labels = parity_data['predicted_labels']
# print(true_labels,predicted_labels)
true_labels = np.ravel(true_labels)
print(true_labels)
print(predicted_labels)
plot_hexbin_parity(true_labels, predicted_labels)
# plot_priority_plot(true_labels, predicted_labels)

