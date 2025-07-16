import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import yaml
import random
from easydict import EasyDict
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoTokenizer, LlamaModel
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_finetuned_model(config, device):
    try:
        from model.network_attention import create_model
        model = create_model(config).to(device)
    except Exception:
        try:
            model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
        except Exception:
            return None
    
    try:
        checkpoint_path = './checkpoints/c0428_2336/best_model.pt'
        if not os.path.exists(checkpoint_path):
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        return None
    
    try:
        if 'model_state_dict' not in checkpoint:
            return None
            
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
            
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except Exception:
            model.load_state_dict(new_state_dict, strict=False)
    except Exception:
        return None
    
    model.eval()
    return model

def load_pretrained_model(device):
    try:
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
        model.eval()
        return model
    except Exception:
        return None

class LlamaAttentionVisualizer:
    def __init__(self, model, tokenizer, device='cuda', max_length=512, debug_mode=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_token_length = max_length
        self.debug_mode = debug_mode
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_token_length)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True
                )

                attention_scores = None
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attention_scores = outputs[1]
                elif hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attention_scores = outputs.attentions
                else:
                    return None, None

                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                return attention_scores, tokens
            except Exception:
                return None, None

    def categorize_tokens(self, tokens, text):
        categorized_indices = {category: [] for category in self.categories}

        category_keywords = {
            'composition': ['compound', 'species', 'composition'],
            'electronic_structure': ['valence', 'iupac', 'species_pp'],
            'crystal_structure': ['crystal', 'class', 'family', 'system'],
            'geometry': ['positions', 'fractional', 'geometry'],
            'lattice_deformation': ['lattice', 'relax', 'spacegroup'],
            'symmetry': ['sg:', 'sg2:'],
            'point_group': ['point_group', 'orbifold', 'order', 'structure', 'type'],
            'magnetic_properties': ['spin', 'magnetic'],
            'physical_properties': ['density', 'physical']
        }

        for i, token in enumerate(tokens):
            if token in ['<|begin_of_text|>', '<|end_of_text|>', '<pad>']:
                continue

            token_clean = token.lower().replace('ġ', '').replace('_', '').strip()

            if len(token_clean) < 2 and token_clean != 'o' and token_clean != 'h':
                continue

            found_match = False
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if (len(keyword) >= 2 and len(token_clean) >= 2 and
                        (keyword.lower() in token_clean or token_clean in keyword.lower())):
                        categorized_indices[category].append(i)
                        found_match = True
                        break
                if found_match:
                    break

        for category in categorized_indices:
            categorized_indices[category] = sorted(list(set(categorized_indices[category])))

        return categorized_indices

    def calculate_category_attention(self, full_attention_scores, categorized_indices):
        category_attention = {category: [] for category in self.categories}

        for layer_idx, layer_attn_tensor in enumerate(full_attention_scores):
            if layer_attn_tensor is None:
                continue

            layer_attn_np = layer_attn_tensor.squeeze(0).cpu().numpy()

            layer_category_attention = {}
            for category, indices in self.categories.items():
                if category not in categorized_indices or not categorized_indices[category]:
                    layer_category_attention[category] = np.zeros(layer_attn_np.shape[0])
                    continue

                relevant_key_indices = categorized_indices[category]
                valid_key_indices = [idx for idx in relevant_key_indices if idx < layer_attn_np.shape[2]]

                if valid_key_indices:
                    category_scores_for_category = layer_attn_np[:, :, valid_key_indices]
                    mean_attention_per_head = np.mean(category_scores_for_category, axis=(1, 2))
                    layer_category_attention[category] = mean_attention_per_head
                else:
                    layer_category_attention[category] = np.zeros(layer_attn_np.shape[0])

            for category in self.categories:
                category_attention[category].append(layer_category_attention.get(category, np.zeros(layer_attn_np.shape[0])))

        return category_attention

    def create_head_attention_heatmap(self, category_attention, output_dir, text_index, head_index):
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(15, 8))

        categories = list(category_attention.keys())
        initial_layer_data = category_attention[categories[0]][0]
        last_layer_data = category_attention[categories[0]][-1]
        
        if head_index >= len(initial_layer_data) or head_index >= len(last_layer_data):
            plt.close()
            return

        initial_head_data = [category_attention[cat][0][head_index] for cat in categories]
        last_head_data = [category_attention[cat][-1][head_index] for cat in categories]

        data = np.array([initial_head_data, last_head_data])

        ax = sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=categories,
                    yticklabels=['Initial Layer', 'Last Layer'],
                    annot_kws={'size': 14})

        plt.title(f'Head {head_index} Attention Scores (Initial vs Last Layer) - Averaged over Dataset',
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
        ax = sns.heatmap(avg_attention, annot=True, fmt='.4f', cmap='YlOrRd',
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
        plt.savefig(f'{output_dir}/average_total_attention_heatmap_{text_index}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def process_model(self, texts, output_dir, sample_size):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'base') and hasattr(self.model.base, 'model') and hasattr(self.model.base.model, 'layers'):
            num_layers = len(self.model.base.model.layers)
        elif hasattr(self.model, 'base') and hasattr(self.model.base, 'layers'):
            num_layers = len(self.model.base.layers)
        else:
            layer_pattern = re.compile(r'base\.layers\.(\d+)')
            max_layer = -1
            for key in self.model.state_dict().keys():
                match = layer_pattern.search(key)
                if match:
                    layer_num = int(match.group(1))
                    max_layer = max(max_layer, layer_num)

            if max_layer >= 0:
                num_layers = max_layer + 1
            else:
                num_layers = 16

        total_category_attention = None
        valid_samples = 0

        os.makedirs(output_dir, exist_ok=True)

        for i, text in enumerate(tqdm(texts[:sample_size], desc="Processing texts")):
            if i == 0:
                self.debug_mode = True
            else:
                self.debug_mode = False

            attention_scores, tokens = self.process_text(text)
            if attention_scores is None or tokens is None:
                continue

            categorized_indices = self.categorize_tokens(tokens, text)
            category_attention = self.calculate_category_attention(attention_scores, categorized_indices)

            if total_category_attention is None:
                total_category_attention = category_attention
            else:
                for category in total_category_attention:
                    if category in category_attention and category_attention[category] and len(category_attention[category]) == len(total_category_attention[category]):
                        for layer_idx in range(len(total_category_attention[category])):
                            if isinstance(total_category_attention[category][layer_idx], np.ndarray) and \
                               isinstance(category_attention[category][layer_idx], np.ndarray) and \
                               total_category_attention[category][layer_idx].shape == category_attention[category][layer_idx].shape:
                                total_category_attention[category][layer_idx] += category_attention[category][layer_idx]

            valid_samples += 1

        if total_category_attention is not None and valid_samples > 0:
            for category in total_category_attention:
                for layer_idx in range(len(total_category_attention[category])):
                    total_category_attention[category][layer_idx] /= valid_samples

            self.create_average_total_attention_heatmap(
                total_category_attention,
                output_dir,
                "full_dataset_average",
                num_layers
            )

            if hasattr(self, 'create_head_attention_heatmap') and total_category_attention and list(total_category_attention.keys()):
                first_category_data = total_category_attention[list(total_category_attention.keys())[0]]
                if first_category_data and first_category_data[0] is not None:
                    num_heads = len(first_category_data[0])
                    for head in range(num_heads):
                        self.create_head_attention_heatmap(
                            total_category_attention,
                            output_dir,
                            "full_dataset_average",
                            head
                        )

            return {"num_layers": num_layers, "total_category_attention": total_category_attention}
        else:
            return None

def process_model_with_visualizer(model, tokenizer, device, texts, output_dir, sample_size):
    visualizer = LlamaAttentionVisualizer(model, tokenizer, device=device)
    return visualizer.process_model(texts, output_dir, sample_size)

if __name__ == "__main__":
    config = EasyDict(yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    texts = pd.read_pickle('./data/df_aflow_4.pkl')
    texts = texts['new_column'].tolist()
    texts, _, _, _ = train_test_split(texts, texts, test_size=0.10, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    SAMPLE_SIZE = 27600

    try:
        finetuned_model = load_finetuned_model(config, device)
        if finetuned_model is not None:
            finetuned_output_dir = 'finetuned_attention_4'
            os.makedirs(finetuned_output_dir, exist_ok=True)
            finetuned_results = process_model_with_visualizer(
                finetuned_model, tokenizer, device, texts, finetuned_output_dir, SAMPLE_SIZE
            )
    except Exception:
        pass

    try:
        pretrained_model = load_pretrained_model(device)
        if pretrained_model is not None:
            pretrained_output_dir = 'pretrained_attention_4'
            os.makedirs(pretrained_output_dir, exist_ok=True)
            pretrained_results = process_model_with_visualizer(
                pretrained_model, tokenizer, device, texts, pretrained_output_dir, SAMPLE_SIZE
            )
    except Exception:
        pass