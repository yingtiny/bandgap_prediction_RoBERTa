import numpy as np
import pandas as pd
import os, pickle
from transformers import AutoTokenizer, LlamaModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from easydict import EasyDict
import yaml
from data.dataloader import load_data
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.features = None
        
    def hook_fn(self, module, input, output):
        self.features = output

def get_embeddings(model, data_loader, device, save_dir='0522_llama_pretrained_embedding_analysis'):
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_extractor = FeatureExtractor()
    encoder_hook = model.register_forward_hook(
        lambda m, i, o: encoder_extractor.__setattr__('features', o.last_hidden_state.mean(dim=1))
    )
    
    embeddings_encoder = []
    band_gaps = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            
            if encoder_extractor.features is not None:
                embeddings_encoder.append(encoder_extractor.features.cpu().numpy())
                band_gaps.append(targets.cpu().numpy())
    
    encoder_hook.remove()
    
    embeddings_encoder = np.vstack(embeddings_encoder)
    band_gaps = np.concatenate(band_gaps)
    
    np.save(f'{save_dir}/encoder_embeddings.npy', embeddings_encoder)
    np.save(f'{save_dir}/band_gaps.npy', band_gaps)
    
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_encoder)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=band_gaps.ravel(), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Band Gap')
    plt.title('t-SNE visualization of Llama encoder embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'{save_dir}/encoder_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return embeddings_encoder, band_gaps

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    model = LlamaModel.from_pretrained('meta-llama/Llama-3.2-1B').to(device)
    _, _, test_loader = load_data(config)
    
    encoder_emb, band_gaps = get_embeddings(model, test_loader, device)