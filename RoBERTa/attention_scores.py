import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import yaml
import random
from easydict import EasyDict
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from model.network_attention import create_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

def load_finetuned_model(config, device):
    model = create_model(config).to(device)
    checkpoint = torch.load('./checkpoints/c0911_0203/best_model.pt')
    

    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():

        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
 
    print("\nProcessed model parameters to load:", new_state_dict.keys())

    print("\nCurrent model parameters:", model.state_dict().keys())
    

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

def load_pretrained_model(device):
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()
    return model

class ImprovedAttentionVisualizer:
    def __init__(self, model, tokenizer, device='cuda', max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_token_length = max_length
        self.categories = {
            'composition': ['compound', 'species', 'composition'],
            'electronic_structure': ['valence_cell_iupac', 'species_pp'],
            'crystal_structure': ['crystal_class', 'crystal_family', 'crystal_system'],
            'geometry': ['positions_fractional', 'geometry'],
            'lattice_deformation': ['lattice_system_relax', 'lattice_variation_relax', 'spacegroup_relax'],
            'symmetry': ['sg', 'sg2'],
            'point_group': ['point_group_orbifold', 'point_group_order', 'point_group_structure', 'point_group_type'],
            'magnetic_properties': ['spinD', 'spin_atom', 'spin_cell'],
            'physical_properties': ['density']
        }

    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_token_length)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )

            print("\nModel output type:", type(outputs))
            print("Model output attributes:", dir(outputs))
            
            if isinstance(outputs, tuple):
                attention_scores = outputs[1]
            else:
                attention_scores = outputs.attentions
                
            if attention_scores is not None:

                print("Attention scores shapes:", [attn.shape for attn in attention_scores])
            
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return attention_scores, tokens

    def get_s_token_attention(self, attention_scores):
        return [layer_attn[:, :, 0, :] for layer_attn in attention_scores]

    def categorize_tokens(self, tokens, text):
        categorized_indices = {category: [] for category in self.categories}
        for category, features in self.categories.items():
            for feature in features:
                pattern = rf"{feature}:\s*(.+?)(?=,\s*\w+:|$)"
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    start_idx = match.start()
                    end_idx = match.end()
                    start_token = len(self.tokenizer.encode(text[:start_idx])) - 1
                    end_token = len(self.tokenizer.encode(text[:end_idx])) - 1
                    start_token = min(start_token, self.max_token_length - 1)
                    end_token = min(end_token, self.max_token_length - 1)
                    categorized_indices[category].extend(range(start_token, end_token + 1))
        return categorized_indices

    def calculate_category_attention(self, s_token_attention, categorized_indices):
        category_attention = {category: [] for category in self.categories}
        for layer_attn in s_token_attention:
            layer_attn = layer_attn.squeeze(0).cpu().numpy()
            layer_category_attention = {}
            for category, indices in categorized_indices.items():
                if indices:
                    category_scores = layer_attn[:, indices]
                    layer_category_attention[category] = np.sum(category_scores, axis=1)
                else:
                    layer_category_attention[category] = np.zeros(layer_attn.shape[0])
            for category in self.categories:
                category_attention[category].append(layer_category_attention[category])
        return category_attention

    def create_head_attention_heatmap(self, category_attention, output_dir, text_index, head_index):
        plt.rcParams.update({'font.size': 14})  
        plt.figure(figsize=(15, 8))
        
        categories = list(category_attention.keys())
        initial_data = [category_attention[cat][0][head_index] for cat in categories]
        last_data = [category_attention[cat][-1][head_index] for cat in categories]
        
        data = np.array([initial_data, last_data])
        
        ax = sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd', 
                    xticklabels=categories, 
                    yticklabels=['Initial Layer', 'Last Layer'],
                    annot_kws={'size': 14})  
 
        plt.title(f'Head {head_index} Attention Scores (Initial vs Last Layer)', 
                fontsize=18, pad=20)
        plt.xlabel('Categories', fontsize=20, labelpad=10)
        plt.ylabel('Layers', fontsize=20, labelpad=10)
        

        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)
        

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/head_{head_index}_attention_heatmap_{text_index}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_average_total_attention_heatmap(self, category_attention, output_dir, text_index, num_layers):
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(15, 10))
        
        categories = list(category_attention.keys())
        num_heads = len(category_attention[categories[0]][0])
        
        avg_attention = np.zeros((num_heads, len(categories)))
        for i, category in enumerate(categories):
            category_data = np.array(category_attention[category])
            avg_attention[:, i] = np.mean(category_data, axis=0)
        
  
        ax = sns.heatmap(avg_attention, 
                        annot=True, 
                        fmt='.4f', 
                        cmap='YlOrRd',
                        xticklabels=categories,
                        yticklabels=[f'Head {i}' for i in range(num_heads)],
                        annot_kws={'size': 15})  

        plt.xlabel('Categories', fontsize=20, labelpad=10)
        plt.ylabel('Attention Heads', fontsize=20, labelpad=10)
        

        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/average_total_attention_heatmap_{text_index}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def validate_attention_scores(finetuned_attention, pretrained_attention):

    differences = {}
    
    for category in finetuned_attention:
        category_diff = []
        for layer_idx in range(len(finetuned_attention[category])):
            fine_scores = finetuned_attention[category][layer_idx]
            pre_scores = pretrained_attention[category][layer_idx]
            

            diff = np.abs(fine_scores - pre_scores)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            category_diff.append({
                'layer': layer_idx,
                'mean_diff': mean_diff,
                'max_diff': max_diff
            })
            
        differences[category] = category_diff
    

    for category, diffs in differences.items():
        print(f"\nCategory: {category}")
        for diff in diffs:
            print(f"Layer {diff['layer']}: Mean diff = {diff['mean_diff']:.6f}, "
                  f"Max diff = {diff['max_diff']:.6f}")
            
    return differences

def process_model(model, tokenizer, device, texts, output_dir):
    visualizer = ImprovedAttentionVisualizer(model, tokenizer, device=device)
    total_category_attention = None
    valid_samples = 0

    for i, text in enumerate(tqdm(texts, desc="Processing texts")):
        try:
            attention_scores, tokens = visualizer.process_text(text)
            
            if attention_scores is None:
                continue
            
            s_token_attention = visualizer.get_s_token_attention(attention_scores)
            categorized_indices = visualizer.categorize_tokens(tokens, text)
            category_attention = visualizer.calculate_category_attention(
                s_token_attention, 
                categorized_indices
            )
            
            if total_category_attention is None:
                total_category_attention = category_attention
            else:
                for category in total_category_attention:
                    for layer_idx in range(len(total_category_attention[category])):
                        total_category_attention[category][layer_idx] += (
                            category_attention[category][layer_idx]
                        )
            
            valid_samples += 1
            
            if valid_samples % 100 == 0:
                print(f"\nProcessed {valid_samples} valid samples")
            
        except Exception as e:
            print(f"\nError processing text {i+1}: {str(e)}")
            continue

    if total_category_attention is not None and valid_samples > 0:
        print(f"\nCalculating average over {valid_samples} valid samples")
        for category in total_category_attention:
            for layer_idx in range(len(total_category_attention[category])):
                total_category_attention[category][layer_idx] /= valid_samples

        visualizer.create_average_total_attention_heatmap(
            total_category_attention,
            output_dir,
            "full_dataset_average",
            len(attention_scores)
        )
        
        num_heads = len(total_category_attention[list(total_category_attention.keys())[0]][0])
        for head in range(num_heads):
            visualizer.create_head_attention_heatmap(
                total_category_attention,
                output_dir,
                "full_dataset_average",
                head
            )
        
        return {'total_category_attention': total_category_attention}
    else:
        print("\nNo valid attention scores were processed")
        return None

if __name__ == "__main__":

    config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    texts = pd.read_pickle('./data/df_aflow_4.pkl')
    y = texts[['Egap']]
    texts = texts['new_column'].tolist()
    
    texts, X_test, y_train_plus_remaining, y_test = train_test_split(
        texts, y, test_size=0.10, random_state=42
    )


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    print("\nProcessing fine-tuned model...")
    finetuned_model = load_finetuned_model(config, device)
    finetuned_output_dir = 'finetuned_attention_1209'
    os.makedirs(finetuned_output_dir, exist_ok=True)
    finetuned_results = process_model(finetuned_model, tokenizer, device, texts, finetuned_output_dir)


    print("\nProcessing pre-trained model...")
    pretrained_model = load_pretrained_model(device)
    pretrained_output_dir = 'pretrained_attention_1209'
    os.makedirs(pretrained_output_dir, exist_ok=True)
    pretrained_results = process_model(pretrained_model, tokenizer, device, texts, pretrained_output_dir)


    if finetuned_results is not None and pretrained_results is not None:
        print("\nComparing attention scores between models:")
        differences = validate_attention_scores(
            finetuned_results['total_category_attention'],
            pretrained_results['total_category_attention']
        )
    else:
        print("\nCannot compare results due to processing errors")