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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def process_same_samples_for_both_models(finetuned_model, pretrained_model, tokenizer, texts, 
                                       finetuned_dir, pretrained_dir, device='cuda', num_samples=10):

    random_indices = random.sample(range(len(texts)), num_samples)
    selected_texts = [texts[i] for i in random_indices]
    

    with open('selected_samples.txt', 'w', encoding='utf-8') as f:
        f.write("Selected Sample Indices and Contents:\n\n")
        for idx, text in zip(random_indices, selected_texts):
            f.write(f"Index {idx}:\n{text}\n{'-'*80}\n")
    
    print("\nSelected sample indices:", random_indices)
    
    print("\nProcessing fine-tuned model samples...")
    finetuned_visualizer = ImprovedAttentionVisualizer(finetuned_model, tokenizer, device=device)
    os.makedirs(finetuned_dir, exist_ok=True)
    

    print("\nProcessing pre-trained model samples...")
    pretrained_visualizer = ImprovedAttentionVisualizer(pretrained_model, tokenizer, device=device)
    os.makedirs(pretrained_dir, exist_ok=True)
    
    finetuned_results = []
    pretrained_results = []
    

    for i, text in enumerate(tqdm(selected_texts, desc="Processing samples"), 1):
        try:

            fine_result = process_single_sample(finetuned_model, tokenizer, 
                                             finetuned_visualizer, text, 
                                             finetuned_dir, i)
            if fine_result is not None:
                finetuned_results.append(fine_result)
            

            pre_result = process_single_sample(pretrained_model, tokenizer, 
                                            pretrained_visualizer, text, 
                                            pretrained_dir, i)
            if pre_result is not None:
                pretrained_results.append(pre_result)
                
        except Exception as e:
            print(f"\nError processing sample {i}: {str(e)}")
            continue
    
    return finetuned_results, pretrained_results

def process_single_sample(model, tokenizer, visualizer, text, output_dir, sample_index):

    attention_scores, tokens = visualizer.process_text(text)
    if attention_scores is None:
        return None
        
    s_token_attention = visualizer.get_s_token_attention(attention_scores)
    categorized_indices = visualizer.categorize_tokens(tokens, text)
    category_attention = visualizer.calculate_category_attention(
        s_token_attention, 
        categorized_indices
    )

    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(16, 11))
    
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
                    annot_kws={'size': 14})

    plt.xlabel('Categories', fontsize=16, labelpad=10)
    plt.ylabel('Attention Heads', fontsize=16, labelpad=10)
    

    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_{sample_index}_attention_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_attention

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
        plt.figure(figsize=(15, 8))
        
        categories = list(category_attention.keys())
        initial_data = [category_attention[cat][0][head_index] for cat in categories]
        last_data = [category_attention[cat][-1][head_index] for cat in categories]
        
        data = np.array([initial_data, last_data])
        
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd', 
                    xticklabels=categories, 
                    yticklabels=['Initial Layer', 'Last Layer'])
        
        plt.title(f'Head {head_index} Attention Scores (Initial vs Last Layer)')
        plt.xlabel('Categories')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/head_{head_index}_attention_heatmap_{text_index}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_average_total_attention_heatmap(self, category_attention, output_dir, text_index, num_layers):
        plt.figure(figsize=(15, 10))
        
        categories = list(category_attention.keys())
        num_heads = len(category_attention[categories[0]][0])
        
        avg_attention = np.zeros((num_heads, len(categories)))
        for i, category in enumerate(categories):
            category_data = np.array(category_attention[category])
            avg_attention[:, i] = np.mean(category_data, axis=0)
        
        sns.heatmap(avg_attention, annot=True, fmt='.4f', cmap='YlOrRd', 
                    xticklabels=categories, 
                    yticklabels=[f'Head {i}' for i in range(num_heads)])
        

        plt.xlabel('Categories')
        plt.ylabel('Attention Heads')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/average_total_attention_heatmap_{text_index}.png', dpi=300, bbox_inches='tight')
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
    finetuned_model = load_finetuned_model(config, device)
    pretrained_model = load_pretrained_model(device)

    print("\nProcessing samples for both models...")
    finetuned_output_dir = 'finetuned_attention_samples_1209'
    pretrained_output_dir = 'pretrained_attention_samples_1209'
    
    finetuned_results, pretrained_results = process_same_samples_for_both_models(
        finetuned_model,
        pretrained_model, 
        tokenizer, 
        texts, 
        finetuned_output_dir,
        pretrained_output_dir,
        device=device
    )