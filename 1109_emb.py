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

def get_embeddings(model, data_loader, device, save_dir='embedding_analysis'):
    os.makedirs(save_dir, exist_ok=True)
    

    cls_extractor = FeatureExtractor()
    linear_extractor = FeatureExtractor()

    cls_hook = model.base.register_forward_hook(
        lambda m, i, o: cls_extractor.__setattr__('features', o[0][:, 0, :])
    )

    linear_hook = model.head[0].register_forward_hook(linear_extractor.hook_fn)
    
    embeddings_cls = []
    embeddings_linear = []
    band_gaps = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            _ = model(input_ids=inputs, attention_mask=attention_mask)
            

            embeddings_cls.append(cls_extractor.features.cpu().numpy())
            embeddings_linear.append(linear_extractor.features.cpu().numpy())
            band_gaps.append(targets.cpu().numpy())

    cls_hook.remove()
    linear_hook.remove()
    

    embeddings_cls = np.vstack(embeddings_cls)
    embeddings_linear = np.vstack(embeddings_linear)
    band_gaps = np.concatenate(band_gaps)

    np.save(f'{save_dir}/cls_embeddings.npy', embeddings_cls)
    np.save(f'{save_dir}/linear_embeddings.npy', embeddings_linear)
    np.save(f'{save_dir}/band_gaps.npy', band_gaps)
    
    print(f"CLS embeddings shape: {embeddings_cls.shape}")
    print(f"Linear layer embeddings shape: {embeddings_linear.shape}")
    print(f"Band gaps shape: {band_gaps.shape}")

    for name, embeddings in [('cls', embeddings_cls), ('linear', embeddings_linear)]:

        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=band_gaps.ravel(), cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Band Gap')
        plt.title(f't-SNE visualization of {name} embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(f'{save_dir}/{name}_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        

        print(f"\n{name.upper()} Embeddings Statistics:")
        print(f"Mean: {np.mean(embeddings):.4f}")
        print(f"Std: {np.std(embeddings):.4f}")
        print(f"Mean L2 norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    
    return embeddings_cls, embeddings_linear, band_gaps


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    model = create_model(config).to(device)
    _, _, test_loader = load_data(config)
    

    cls_emb, linear_emb, band_gaps = get_embeddings(model, test_loader, device)