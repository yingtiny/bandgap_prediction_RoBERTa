import numpy as np
import pandas as pd
import os, pickle
from transformers import RobertaTokenizer, RobertaModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.network_attention import create_model, setup_training
from easydict import EasyDict
import yaml
from data.dataloader import load_data
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.features = None
        
    def hook_fn(self, module, input, output):
        self.features = output

def get_embeddings(model, data_loader, device, save_dir='embedding_analysis_pre'):
    os.makedirs(save_dir, exist_ok=True)
    
    embeddings_cls = []
    band_gaps = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            

            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :] 
            

            embeddings_cls.append(cls_embedding.cpu().numpy())
            band_gaps.append(targets.cpu().numpy())
    

    embeddings_cls = np.vstack(embeddings_cls)
    band_gaps = np.concatenate(band_gaps)

    np.save(f'{save_dir}/pretrained_cls_embeddings.npy', embeddings_cls)
    np.save(f'{save_dir}/pretrained_band_gaps.npy', band_gaps)
    
    print(f"CLS embeddings shape: {embeddings_cls.shape}")
    print(f"Band gaps shape: {band_gaps.shape}")
    

    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_cls)
    

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=band_gaps.ravel(), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Band Gap')
    plt.title('t-SNE visualization of pre-trained embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'{save_dir}/pretrained_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    print(f"\nPre-trained Embeddings Statistics:")
    print(f"Mean: {np.mean(embeddings_cls):.4f}")
    print(f"Std: {np.std(embeddings_cls):.4f}")
    print(f"Mean L2 norm: {np.mean(np.linalg.norm(embeddings_cls, axis=1)):.4f}")
    
    return embeddings_cls, band_gaps

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    

    print("\nModel parameters summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter means for first few layers:")
    for name, param in list(model.named_parameters())[:5]:
        print(f"{name}: {param.mean().item():.4f}")
    

    config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
    _, _, test_loader = load_data(config)
    

    cls_emb, band_gaps = get_embeddings(model, test_loader, device)
    

    try:
        finetuned_emb = np.load('embedding_analysis/cls_embeddings.npy')
        if finetuned_emb.shape == cls_emb.shape:
            print("\nComparing with fine-tuned embeddings:")
            diff = np.mean(np.abs(cls_emb - finetuned_emb))
            correlation = np.corrcoef(cls_emb.reshape(-1), finetuned_emb.reshape(-1))[0,1]
            print(f"Average absolute difference: {diff:.4f}")
            print(f"Correlation: {correlation:.4f}")
    except FileNotFoundError:
        print("\nNo fine-tuned embeddings found for comparison")