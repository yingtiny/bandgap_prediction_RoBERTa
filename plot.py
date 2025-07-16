import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def plot_hexbin_parity(true_labels, predicted_labels, output_name='hexbin_parity_plot.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    
    hb = ax.hexbin(true_labels, predicted_labels, gridsize=50, cmap='viridis', bins='log')
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.tick_params(labelsize=28)
    cb.set_label('Count', fontsize=28)
    
    ax.plot([0, 5], [0, 5], 'k--', linewidth=1.5, alpha=0.75, zorder=0)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    
    plt.xlabel('Aflow Bandgap (eV)', fontsize=40)
    plt.ylabel('Predicted Bandgap (eV)', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
    
    r2 = r2_score(true_labels, predicted_labels)
    mae = np.mean(np.abs(np.array(true_labels) - np.array(predicted_labels)))
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predicted_labels))**2))
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    ax.tick_params(top=False, right=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parity_data = np.load('./parity_data_0518_r_4_freezefirst.npz')
    true_labels = np.ravel(parity_data['true_labels'])
    predicted_labels = parity_data['predicted_labels']
    
    plot_hexbin_parity(true_labels, predicted_labels, 'hexbin_parity_plot_roberta_freeze_first.png')
