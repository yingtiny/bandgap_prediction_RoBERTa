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

def compare_model_structure(model1, model2, name1="Finetuned", name2="Pretrained"):
    """Compare the structure of two models and print differences."""
    model1_state = model1.state_dict()
    model2_state = model2.state_dict()

    model1_keys = set(model1_state.keys())
    model2_keys = set(model2_state.keys())

    print(f"\n{name1} model has {len(model1_keys)} parameters")
    print(f"{name2} model has {len(model2_keys)} parameters")

    print(f"\nParameters in {name1} but not in {name2}: {len(model1_keys - model2_keys)}")
    print(f"Parameters in {name2} but not in {name1}: {len(model2_keys - model1_keys)}")

    # Check common parameters
    common_keys = model1_keys.intersection(model2_keys)
    print(f"\nCommon parameters: {len(common_keys)}")

    # Check for shape differences in common parameters
    shape_diffs = 0
    for key in common_keys:
        if model1_state[key].shape != model2_state[key].shape:
            shape_diffs += 1
            print(f"Shape difference in {key}: {model1_state[key].shape} vs {model2_state[key].shape}")

    print(f"\nParameters with shape differences: {shape_diffs}")

def load_finetuned_model(config, device):
    """
    Load a finetuned Llama model.

    Args:
        config: Configuration object with model parameters
        device: The device to load the model to (CPU or CUDA)

    Returns:
        A finetuned Llama model loaded to the specified device
    """
    try:
        from model.network_attention import create_model
        model = create_model(config).to(device)
        print("Successfully created fine-tuned model")
    except Exception as e:
        print(f"Error creating fine-tuned model: {str(e)}")
        try:
            model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
            print("Loaded base model as fallback")
        except Exception as e2:
            print(f"Error loading fallback model: {str(e2)}")
            return None

    try:
        checkpoint_path = './checkpoints/c0428_2336/best_model.pt'
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint keys: {checkpoint.keys()}")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

    try:
        if 'model_state_dict' not in checkpoint:
            print(f"model_state_dict not found in checkpoint. Available keys: {checkpoint.keys()}")
            return None

        state_keys = list(checkpoint['model_state_dict'].keys())
        print(f"First 5 state dict keys: {state_keys[:5]}")

        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        model_keys = list(model.state_dict().keys())
        print(f"First 5 model keys: {model_keys[:5]}")

        checkpoint_keys = set(new_state_dict.keys())
        model_expected_keys = set(model.state_dict().keys())
        missing_keys = model_expected_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_expected_keys

        if missing_keys:
            print(f"Missing keys in checkpoint: {list(missing_keys)[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:5]}")

        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded state dict with strict=False")
        except Exception as e:
            print(f"Error loading with strict=False: {str(e)}")

        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Loaded state dict with strict=True")
        except Exception as e:
            print(f"Error loading with strict=True: {str(e)}")
            print("Continuing with strict=False loading")
    except Exception as e:
        print(f"Error processing state dict: {str(e)}")
        return None

    model.eval()
    return model

def load_pretrained_model(device):
    """
    Load a pretrained Llama model from HuggingFace.

    Args:
        device: The device to load the model to (CPU or CUDA)

    Returns:
        A LlamaForCausalLM instance loaded to the specified device
    """
    try:
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
        print("Successfully loaded pre-trained Llama-3.2-1B model")

        if hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                print(f"Pretrained model has {len(model.model.layers)} layers")
            else:
                print("WARNING: Pretrained model structure is different than expected!")
        else:
            print("WARNING: Pretrained model does not have 'model' attribute!")

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {str(e)}")
        import traceback
        traceback.print_exc()
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

                print(f"--- Raw Model Outputs Debug for text: '{text[:50]}...' ---")
                print(f"Type of outputs: {type(outputs)}")
                if isinstance(outputs, tuple):
                    print(f"Outputs is a tuple of length: {len(outputs)}")
                    for i, item in enumerate(outputs):
                        print(f"  Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'No shape')}")
                elif hasattr(outputs, 'attentions'):
                    print(f"Outputs has .attentions attribute.")

                attention_scores = None
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attention_scores = outputs[1]  # Assuming your custom model returns attention as the second element
                    print(f"Extracted attention_scores from tuple element 1.")
                elif hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attention_scores = outputs.attentions
                    print(f"Extracted attention_scores from .attentions attribute.")
                else:
                    print("No attention scores found in expected places within model outputs.")
                    return None, None

                if attention_scores is not None:
                    print(f"Type of attention_scores: {type(attention_scores)}")
                    if isinstance(attention_scores, tuple) or isinstance(attention_scores, list):
                        print(f"Number of attention layers found: {len(attention_scores)}")
                        if len(attention_scores) > 0:
                            first_layer_attn = attention_scores[0]
                            print(f"First layer attention type: {type(first_layer_attn)}")
                            if hasattr(first_layer_attn, 'shape'):
                                print(f"First layer attention shape: {first_layer_attn.shape}")
                                print(f"First layer attention min: {first_layer_attn.min():.6f}, max: {first_layer_attn.max():.6f}, mean: {first_layer_attn.mean():.6f}")
                            else:
                                print(f"First layer attention has no shape attribute.")
                        else:
                            print("Attention scores list/tuple is empty.")
                    elif hasattr(attention_scores, 'shape'):
                        print(f"Attention scores shape: {attention_scores.shape}")
                        print(f"Attention scores min: {attention_scores.min():.6f}, max: {attention_scores.max():.6f}, mean: {attention_scores.mean():.6f}")
                    else:
                        print("Attention scores is not a list/tuple and has no shape.")
                else:
                    print("Attention scores variable is None.")

                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                return attention_scores, tokens
            except Exception as e:
                print(f"Error in process_text: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None

    def debug_model_outputs(self, text):
        """Debug function to inspect model outputs structure"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_token_length)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            try:
                # Only use output_attentions to avoid errors with custom models
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True
                )

                print("Type of outputs:", type(outputs))
                if hasattr(outputs, '_fields'):
                    print("Output fields:", outputs._fields)

                if isinstance(outputs, torch.nn.modules.module.Module): # If it's just the model itself returned, not an output object
                    print("Output is likely the model itself, not a specific output object. This is unusual.")
                    # Try to call it again and inspect its return type
                    outputs_test = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_attentions=True)
                    print(f"Outputs_test type: {type(outputs_test)}")
                    # Proceed with checking outputs_test
                    outputs = outputs_test


                if isinstance(outputs, tuple):
                    print("Outputs is a tuple of length:", len(outputs))
                    for i, item in enumerate(outputs):
                        print(f"Item {i} type:", type(item))
                        if item is not None:
                            if isinstance(item, tuple):
                                print(f"Item {i} is a tuple of length:", len(item))
                                if len(item) > 0:
                                    print(f"First element type: {type(item[0])}")
                                    if hasattr(item[0], 'shape'):
                                        print(f"First element shape: {item[0].shape}")
                            elif hasattr(item, 'shape'):
                                print(f"Item {i} shape:", item.shape)


                else:
                    # For Hugging Face output objects
                        for attr_name in dir(outputs):
                            if not attr_name.startswith('_'):
                                attr = getattr(outputs, attr_name)
                                if attr is not None and not callable(attr):
                                    if isinstance(attr, (list, tuple)):
                                        print(f"{attr_name}: {type(attr)} of length {len(attr)}")
                                        if len(attr) > 0:
                                            print(f"  First element type: {type(attr[0])}")
                                            if hasattr(attr[0], 'shape'):
                                                print(f"  First element shape: {attr[0].shape}")
                                    elif hasattr(attr, 'shape'):
                                        print(f"{attr_name}: shape {attr.shape}")
                                    else:
                                        print(f"{attr_name}: {type(attr)}")

                return outputs
            except Exception as e:
                print(f"Error debugging outputs: {str(e)}")
                import traceback
                traceback.print_exc()
                return None



    def debug_model_output_format(self, text):
        """Debug function to inspect model output format"""
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

                print("Type of outputs:", type(outputs))

                if isinstance(outputs, tuple):
                    print("Output is a tuple with length:", len(outputs))
                    for i, item in enumerate(outputs):
                        print(f"Item {i} type:", type(item))
                        if isinstance(item, tuple):
                            print(f"  Tuple length: {len(item)}")
                            for j, subitem in enumerate(item):
                                print(f"    Subitem {j} type/shape:", type(subitem),
                                    getattr(subitem, 'shape', 'No shape attribute'))
                        elif hasattr(item, 'shape'):
                            print(f"  Shape: {item.shape}")
                else:
                    print("Output attributes:", [attr for attr in dir(outputs) if not attr.startswith('_') and not callable(getattr(outputs, attr))])

                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        print("Attentions type:", type(outputs.attentions))
                        print("Number of attention layers:", len(outputs.attentions))
                        print("First attention layer shape:", outputs.attentions[0].shape)

                return outputs
            except Exception as e:
                print(f"Error in debug: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
    def get_first_token_attention(self, attention_scores):
        # 移除大量不必要的print
        # print("--- Debug: Inside get_first_token_attention ---")
        first_token_attentions = []
        for i, layer_attn in enumerate(attention_scores):
            if layer_attn is None:
                # print(f"Warning: Layer {i} attention is None. Skipping.")
                first_token_attentions.append(None)
                continue

            if isinstance(layer_attn, torch.Tensor):
                # 確保 layer_attn 的維度足夠，防止索引錯誤
                # layer_attn 預期形狀: [batch_size, num_heads, query_seq_len, key_seq_len]
                if layer_attn.dim() < 4 or layer_attn.shape[2] == 0:
                    print(f"Warning: Layer {i} attention has unexpected shape {layer_attn.shape} or empty sequence. Skipping.")
                    first_token_attentions.append(None)
                    continue

                # 提取第一個詞元對所有詞元的注意力
                # 結果形狀: [batch_size, num_heads, key_seq_len]
                attn_slice = layer_attn[:, :, 0, :]
                first_token_attentions.append(attn_slice)

                # 僅針對第一個文本的第一層進行詳細列印，以減少輸出
                if i == 0 and self.debug_mode: # 新增一個debug_mode旗標來控制輸出
                    print(f"Debug: First layer, first token attention slice shape: {attn_slice.shape}")
                    print(f"Debug: First layer, first token attention min: {attn_slice.min().item():.6f}, max: {attn_slice.max().item():.6f}, mean: {attn_slice.mean().item():.6f}")
            else:
                print(f"Warning: Layer {i} attention is not a Tensor. Skipping.")
                first_token_attentions.append(None)

        return first_token_attentions

    def categorize_tokens(self, tokens, text):
        categorized_indices = {category: [] for category in self.categories}

        category_keywords = {
            'composition': ['compound', 'species', 'composition'],
            'electronic_structure': ['valence', 'iupac', 'species_pp'],
            'crystal_structure': ['crystal', 'class', 'family', 'system'],
            'geometry': ['positions', 'fractional', 'geometry'],
            'lattice_deformation': ['lattice', 'relax', 'spacegroup'], # 'spacegroup' 應該是完整的詞
            'symmetry': ['sg:', 'sg2:'], # 'sg:' 這樣的關鍵詞更有針對性
            'point_group': ['point_group', 'orbifold', 'order', 'structure', 'type'], # '_group' 應該與 'point_group' 結合
            'magnetic_properties': ['spin', 'magnetic'],
            'physical_properties': ['density', 'physical']
        }

        # 移除這個函式內部的大量print，只保留最終總結
        # print("Token categorization debug:") # 保持這行

        for i, token in enumerate(tokens):
            if token in ['<|begin_of_text|>', '<|end_of_text|>', '<pad>']:
                continue

            token_clean = token.lower().replace('ġ', '').replace('_', '').strip()

            if len(token_clean) < 2 and token_clean != 'o' and token_clean != 'h': # 允許單字母元素，但排除其他過短或無意義的
                continue

            found_match = False
            # 優先匹配更長的、更具體的關鍵詞，避免短詞元被錯誤分類
            # 例如，先檢查 'point_group' 再檢查 '_group'

            # 可以調整匹配順序，或者使用更嚴格的詞邊界匹配 (例如 regex \bword\b)
            # 為了簡化，我們先維持目前的邏輯，但記住這可能是一個優化點

            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if (len(keyword) >= 2 and len(token_clean) >= 2 and
                        (keyword.lower() in token_clean or token_clean in keyword.lower())):
                        categorized_indices[category].append(i)
                        # print(f"  Token {i} '{token}' -> {category} (matched '{keyword}')") # 暫時註釋掉，除非需要非常詳細的單詞元級別調試
                        found_match = True
                        break
                if found_match:
                    break

        for category in categorized_indices:
            categorized_indices[category] = sorted(list(set(categorized_indices[category])))
            # 保持這個列印，因為它總結了每個類別的詞元數量，很有用
            # if categorized_indices[category]:
            #     print(f"Final {category}: {len(categorized_indices[category])} tokens")

        return categorized_indices


    # 修改為處理 full_attention_scores
    def calculate_category_attention(self, full_attention_scores, categorized_indices): # 參數名稱改為 full_attention_scores
        category_attention = {category: [] for category in self.categories}
        debug_this_calculation = True if self.debug_mode else False

        # full_attention_scores 是一個列表，每個元素是形狀為 [batch, heads, query_seq_len, key_seq_len] 的 tensor
        for layer_idx, layer_attn_tensor in enumerate(full_attention_scores):
            if layer_attn_tensor is None:
                if debug_this_calculation and layer_idx == 0:
                    print(f"Debug: Layer {layer_idx} attention is None, skipping category calculation.")
                continue

            # 假設 batch_size 為 1，將其擠壓掉
            # layer_attn_np 的形狀會是 [num_heads, query_seq_len, key_seq_len]
            layer_attn_np = layer_attn_tensor.squeeze(0).cpu().numpy()

            if debug_this_calculation and layer_idx == 0:
                print(f"Debug: Full attention for Layer {layer_idx} shape: {layer_attn_np.shape}")
                print(f"Debug: Full attention for Layer {layer_idx} min: {layer_attn_np.min():.6f}, max: {layer_attn_np.max():.6f}, mean: {layer_attn_np.mean():.6f}")

            layer_category_attention = {}
            for category, indices in self.categories.items(): # 迭代所有預期的類別
                if category not in categorized_indices or not categorized_indices[category]:
                    layer_category_attention[category] = np.zeros(layer_attn_np.shape[0]) # num_heads
                    if debug_this_calculation and layer_idx == 0:
                        print(f"Debug: Category '{category}' has no tokens. Attention set to zeros.")
                    continue

                # 獲取該類別在當前文本中的詞元索引
                # 這些索引是對應到 key_seq_len 維度的
                relevant_key_indices = categorized_indices[category]

                # 確保索引在 key_seq_len 範圍內
                valid_key_indices = [idx for idx in relevant_key_indices if idx < layer_attn_np.shape[2]] # 注意這裡的 shape[2] 對應 key_seq_len

                if valid_key_indices:
                    # 選擇所有 query token 對這些 valid_key_indices 的注意力
                    # category_scores_for_category shape: [num_heads, query_seq_len, num_valid_key_indices]
                    category_scores_for_category = layer_attn_np[:, :, valid_key_indices]

                    # 在這裡添加調試列印，檢查切片後的 attention values
                    if debug_this_calculation and layer_idx == 0:
                        print(f"Debug: Category '{category}' - category_scores_for_category shape: {category_scores_for_category.shape}")
                        print(f"Debug: Category '{category}' - category_scores_for_category sample (head 0, first 3 queries, first 3 keys): \n{category_scores_for_category[0, :3, :min(3, len(valid_key_indices))]}")
                        print(f"Debug: Category '{category}' - category_scores_for_category min: {category_scores_for_category.min():.6f}, max: {category_scores_for_category.max():.6f}, mean: {category_scores_for_category.mean():.6f}")

                    # 對 query_seq_len (axis=1) 和 num_valid_key_indices (axis=2) 兩個維度求平均
                    mean_attention_per_head = np.mean(category_scores_for_category, axis=(1, 2)) # 改回平均
                    layer_category_attention[category] = mean_attention_per_head

                    if debug_this_calculation and layer_idx == 0:
                        print(f"Debug: Category '{category}' - {len(valid_key_indices)} valid tokens (keys). Mean attention (first 3 heads): {mean_attention_per_head[:3]}")
                        print(f"Debug: Category '{category}' - final mean: {mean_attention_per_head.mean():.6f}")
                else:
                    layer_category_attention[category] = np.zeros(layer_attn_np.shape[0]) # num_heads
                    if debug_this_calculation and layer_idx == 0:
                        print(f"Debug: Category '{category}' has no valid key indices in this text. Attention set to zeros.")

            for category in self.categories:
                category_attention[category].append(layer_category_attention.get(category, np.zeros(layer_attn_np.shape[0])))

        return category_attention

    def create_head_attention_heatmap(self, category_attention, output_dir, text_index, head_index):
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(15, 8))

        categories = list(category_attention.keys())
        # 這裡的 category_attention[cat] 是一個列表，包含所有層的 [num_heads] 平均注意力
        # 我們需要選擇特定 head_index 的數據
        initial_layer_data = category_attention[categories[0]][0] # 第0層所有head的注意力
        last_layer_data = category_attention[categories[0]][-1] # 最後一層所有head的注意力
        
        # 檢查確保 head_index 在有效範圍內
        if head_index >= len(initial_layer_data) or head_index >= len(last_layer_data):
            print(f"Warning: Head index {head_index} is out of bounds for attention data. Skipping heatmap for this head.")
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
            # avg_attention[:, i] = np.mean(category_data, axis=0) # 這是所有層的平均，對於整體熱圖
            avg_attention[:, i] = np.mean(category_data, axis=0) # 這是所有層的平均
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
        """
        Process the model to extract attention scores and generate visualizations.
        """
        # Determine number of layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            print(f"Found {num_layers} layers in HuggingFace model.model.layers")
        elif hasattr(self.model, 'base') and hasattr(self.model.base, 'model') and hasattr(self.model.base.model, 'layers'):
            num_layers = len(self.model.base.model.layers)
            print(f"Found {num_layers} layers in custom model.base.model.layers")
        elif hasattr(self.model, 'base') and hasattr(self.model.base, 'layers'):
            num_layers = len(self.model.base.layers)
            print(f"Found {num_layers} layers in custom model.base.layers")
        else:
            print("Standard layer detection failed. Attempting deeper inspection...")
            layer_pattern = re.compile(r'base\.layers\.(\d+)')
            max_layer = -1
            for key in self.model.state_dict().keys():
                match = layer_pattern.search(key)
                if match:
                    layer_num = int(match.group(1))
                    max_layer = max(max_layer, layer_num)

            if max_layer >= 0:
                num_layers = max_layer + 1
                print(f"Detected {num_layers} layers from state_dict keys")
            else:
                print("Using default of 16 layers for Llama-3.2-1B")
                num_layers = 16

        print(f"Processing model with {num_layers} layers")

        total_category_attention = None
        valid_samples = 0

        os.makedirs(output_dir, exist_ok=True)

        for i, text in enumerate(tqdm(texts[:sample_size], desc="Processing texts")):
            if i == 0:
                self.debug_mode = True
                print("\n--- DEBUG FOR FIRST TEXT ONLY ---")
            else:
                self.debug_mode = False

            attention_scores, tokens = self.process_text(text)
            if attention_scores is None or tokens is None:
                print(f"Skipping text {i + 1} due to processing error.")
                continue

            # 修改：直接傳遞完整的 attention_scores
            # first_token_attention = self.get_first_token_attention(attention_scores) # 移除這行
            # if not first_token_attention or any(item is None for item in first_token_attention): # 移除這行
            #     print(f"Skipping text {i + 1} as first token attention extraction failed or returned None.") # 移除這行
            #     continue # 移除這行

            print("Token categorization debug:")
            categorized_indices = self.categorize_tokens(tokens, text)

            # 修改：傳遞完整的 attention_scores
            category_attention = self.calculate_category_attention(attention_scores, categorized_indices) # 傳遞 attention_scores

            if total_category_attention is None:
                total_category_attention = category_attention
            else:
                for category in total_category_attention:
                    # 確保 category_attention[category] 不為空，或者進行適當初始化
                    if category in category_attention and category_attention[category] and len(category_attention[category]) == len(total_category_attention[category]):
                        for layer_idx in range(len(total_category_attention[category])):
                            # 檢查數據是否是有效的 NumPy 數組，且形狀匹配
                            if isinstance(total_category_attention[category][layer_idx], np.ndarray) and \
                               isinstance(category_attention[category][layer_idx], np.ndarray) and \
                               total_category_attention[category][layer_idx].shape == category_attention[category][layer_idx].shape:
                                total_category_attention[category][layer_idx] += category_attention[category][layer_idx]
                            else:
                                print(f"Warning: Shape or type mismatch for category '{category}' layer {layer_idx}. Skipping addition.")
                    else:
                        # 如果某個 category 在當前 sample 中沒有有效 token，其 category_attention 會是 zeros
                        # 我們需要確保 total_category_attention 也累加 zeros
                        for layer_idx in range(len(total_category_attention[category])):
                            if isinstance(total_category_attention[category][layer_idx], np.ndarray):
                                total_category_attention[category][layer_idx] += np.zeros_like(total_category_attention[category][layer_idx])


            valid_samples += 1

        if total_category_attention is not None and valid_samples > 0:
            print(f"\nCalculating average over {valid_samples} valid samples")
            for category in total_category_attention:
                for layer_idx in range(len(total_category_attention[category])):
                    total_category_attention[category][layer_idx] /= valid_samples

            self.create_average_total_attention_heatmap(
                total_category_attention,
                output_dir,
                "full_dataset_average",
                num_layers
            )

            # 添加生成每個 Head 的平均熱圖的邏輯
            if hasattr(self, 'create_head_attention_heatmap') and total_category_attention and list(total_category_attention.keys()):
                # 確保 category_attention[categories[0]] 有至少一個 layer_data，並且這個 data 是非空的
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
                else:
                    print("Warning: Cannot determine num_heads for create_head_attention_heatmap. Skipping.")

            return {"num_layers": num_layers, "total_category_attention": total_category_attention}
        else:
            print("\nNo valid attention scores were processed")
            return None

    def debug_attention_extraction(self, text):
        """調試attention提取過程"""
        print("=== DEBUG ATTENTION EXTRACTION ===")

        # 1. 檢查tokenization
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_token_length)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        print(f"Input text: {text[:100]}...")
        print(f"Number of tokens: {len(tokens)}")
        print(f"First 10 tokens: {tokens[:10]}")

        # 2. 檢查attention scores結構
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )

            # 處理不同的輸出格式
            if isinstance(outputs, tuple):
                attention_scores = outputs[1] if len(outputs) > 1 else None
            elif hasattr(outputs, 'attentions'):
                attention_scores = outputs.attentions
            else:
                attention_scores = None

            if attention_scores is not None:
                print(f"Number of attention layers: {len(attention_scores)}")
                print(f"First layer attention shape: {attention_scores[0].shape}")
                print(f"Attention score range: {attention_scores[0].min():.6f} to {attention_scores[0].max():.6f}")

                # 檢查第一個token的attention
                first_token_attn = attention_scores[0][:, :, 0, :]  # [batch, heads, seq_len]
                print(f"First token attention shape: {first_token_attn.shape}")
                print(f"First token attention sum per head: {first_token_attn.sum(dim=-1)}")

            else:
                print("ERROR: No attention scores found!")

        # 3. 檢查token分類
        categorized_indices = self.categorize_tokens(tokens, text)
        print(f"\nToken categorization:")
        for category, indices in categorized_indices.items():
            if indices:
                print(f"  {category}: {len(indices)} tokens at indices {indices[:5]}...")
                # 顯示這些indices對應的實際tokens
                category_tokens = [tokens[i] for i in indices if i < len(tokens)]
                print(f"    Tokens: {category_tokens[:5]}")
            else:
                print(f"  {category}: 0 tokens")

        # 4. 檢查正則表達式匹配
        print(f"\nRegex matching test:")
        for category, features in self.categories.items():
            for feature in features:
                pattern = rf"{feature}:\s*(.+?)(?=,\s*\w+:|$)"
                matches = list(re.finditer(pattern, text))
                if matches:
                    print(f"  {feature}: {len(matches)} matches")
                    for match in matches:
                        print(f"    Match: '{match.group()}'")

        return attention_scores, tokens, categorized_indices

    def debug_category_attention_calculation(self, attention_scores, categorized_indices):
        print("\n=== DEBUG CATEGORY ATTENTION CALCULATION ===")

        # 這裡直接使用傳入的 attention_scores (完整的注意力列表)
        for layer_idx, layer_attn_tensor in enumerate(attention_scores[:2]): # 只檢查前2層
            if layer_attn_tensor is None:
                print(f"Layer {layer_idx}: attention is None")
                continue

            layer_attn_np = layer_attn_tensor.squeeze(0).cpu().numpy() # shape: [heads, query_seq_len, key_seq_len]
            print(f"\nLayer {layer_idx} (Full attention):")
            print(f"  Attention shape: {layer_attn_np.shape}")
            print(f"  Attention range: {layer_attn_np.min():.6f} to {layer_attn_np.max():.6f}")

            for category, relevant_indices in categorized_indices.items():
                valid_key_indices = [idx for idx in relevant_indices if idx < layer_attn_np.shape[2]] # 注意這裡的 shape[2]
                if valid_key_indices:
                    category_scores = layer_attn_np[:, :, valid_key_indices] # [heads, query_seq_len, num_valid_key_indices]
                    # 改回平均
                    category_mean_per_head = np.mean(category_scores, axis=(1, 2)) # 對 query_seq_len 和 key_seq_len 維度求平均
                    print(f"  {category}: indices={len(valid_key_indices)}, mean_attention (first 3 heads): {category_mean_per_head[:3]}...")
                    print(f"  {category}: total mean: {category_mean_per_head.mean():.6f}")
                else:
                    print(f"  {category}: no valid indices")

    def process_text_with_mask(self, text):
        """修改版本，返回attention_mask"""
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

                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attention_scores = outputs[1]
                elif hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attention_scores = outputs.attentions
                else:
                    return None, None, None

                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                return attention_scores, tokens, inputs['attention_mask']
            except Exception as e:
                print(f"Error in process_text: {str(e)}")
                return None, None, None

    def test_attention_values(self, text):
        """快速測試不同token的attention值"""
        attention_scores, tokens, attention_mask = self.process_text_with_mask(text)
        if attention_scores is None:
            return

        print("=== ATTENTION VALUES TEST ===")
        first_layer = attention_scores[0] # 只看第一層
        print(f"First layer attention shape: {first_layer.shape}")

        # 測試前幾個tokens的attention
        for i in range(min(5, first_layer.shape[2])):
            token_attn = first_layer[0, :, i, :] # [heads, seq_len]
            mean_val = token_attn.mean().item()
            max_val = token_attn.max().item()
            print(f"Token {i} '{tokens[i]}': mean={mean_val:.6f}, max={max_val:.6f}")

        # 測試最後幾個tokens
        print("\nLast few tokens:")
        for i in range(max(0, first_layer.shape[2] - 3), first_layer.shape[2]):
            token_attn = first_layer[0, :, i, :] # [heads, seq_len]
            mean_val = token_attn.mean().item()
            max_val = token_attn.max().item()
            print(f"Token {i} '{tokens[i]}': mean={mean_val:.6f}, max={max_val:.6f}")

# Function to process models with the visualizer
def process_model_with_visualizer(model, tokenizer, device, texts, output_dir, sample_size):
    """Wrapper function to create visualizer and process the model"""
    visualizer = LlamaAttentionVisualizer(model, tokenizer, device=device)
    return visualizer.process_model(texts, output_dir, sample_size)


def test_attention_extraction(model, tokenizer, device):
    print("\n=== TESTING ATTENTION EXTRACTION ===")
    visualizer = LlamaAttentionVisualizer(model, tokenizer, device=device)

    test_text = "compound: Cu2O, species: ['Cu', 'O'], composition: [2, 1]"

    with torch.no_grad():
        outputs = visualizer.debug_model_output_format(test_text)

        attention_scores, tokens = visualizer.process_text(test_text)

        if attention_scores is not None and tokens is not None:
            print("Successfully extracted attention scores!")
            print(f"Number of layers in attention scores: {len(attention_scores)}")
            print(f"Shape of first layer attention: {attention_scores[0].shape if hasattr(attention_scores[0], 'shape') else 'No shape'}")
            print(f"First few tokens: {tokens[:10]}")

            # 這裡不再需要 first_token_attention 了
            # first_token_attention = visualizer.get_first_token_attention(attention_scores)

            categorized_indices = visualizer.categorize_tokens(tokens, test_text)
            print(f"Categorized indices: {len(categorized_indices)} categories")
            for category, indices in categorized_indices.items():
                if indices:
                    print(f"  {category}: {len(indices)} indices")

            # 將 full_attention_scores 傳遞給 calculate_category_attention
            if attention_scores[0] is not None:
                category_attention = visualizer.calculate_category_attention(attention_scores, categorized_indices)
                print(f"Category attention: {len(category_attention)} categories")
                for category, attentions in category_attention.items():
                    if attentions:
                        print(f"  {category}: {len(attentions)} layers")
        else:
            print("Failed to extract attention scores!")

    print("=== END TESTING ===\n")


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
        print("\nProcessing fine-tuned model...")
        finetuned_model = load_finetuned_model(config, device)
        if finetuned_model is not None:
            visualizer = LlamaAttentionVisualizer(finetuned_model, tokenizer, device=device)

            # 調試第一個文本
            sample_text = texts[0] if texts else "compound: Cu2O, species: ['Cu', 'O'], composition: [2, 1]"

            # 注意：這裡的 debug_attention_extraction 仍會執行 get_first_token_attention 的邏輯
            attention_scores, tokens, categorized_indices = visualizer.debug_attention_extraction(sample_text)

            if attention_scores is not None:
                # 這裡的 debug_category_attention_calculation 現在會使用 full_attention_scores
                visualizer.debug_category_attention_calculation(attention_scores, categorized_indices)

        if finetuned_model is not None:
            # Test the attention extraction first
            test_attention_extraction(finetuned_model, tokenizer, device)

            # Create a visualizer to debug the output format
            visualizer = LlamaAttentionVisualizer(finetuned_model, tokenizer, device=device)
            print("Debugging model output format...")
            sample_text = texts[0] if texts else "compound: Cu2O, species: ['Cu', 'O'], composition: [2, 1]"
            debug_output = visualizer.debug_model_output_format(sample_text)
            # 添加這行來運行測試！
            visualizer.test_attention_values(sample_text)

            # 注意：這裡的 debug_attention_extraction 仍會執行 get_first_token_attention 的邏輯
            attention_scores, tokens, categorized_indices = visualizer.debug_attention_extraction(sample_text)

            # Process the model
            finetuned_output_dir = 'finetuned_attention_4'
            os.makedirs(finetuned_output_dir, exist_ok=True)
            finetuned_results = process_model_with_visualizer(
                finetuned_model, tokenizer, device, texts, finetuned_output_dir, SAMPLE_SIZE
            )
            if finetuned_results:
                print(f"Successfully processed finetuned model with {finetuned_results['num_layers']} layers")
            else:
                print("Failed to process finetuned model")
    except Exception as e:
        print(f"Error in finetuned model processing: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        print("\nProcessing pre-trained model...")
        pretrained_model = load_pretrained_model(device)
        if pretrained_model is not None:
            # Test the attention extraction first
            test_attention_extraction(pretrained_model, tokenizer, device)

            # Process the model
            pretrained_output_dir = 'pretrained_attention_4'
            os.makedirs(pretrained_output_dir, exist_ok=True)
            pretrained_results = process_model_with_visualizer(
                pretrained_model, tokenizer, device, texts, pretrained_output_dir, SAMPLE_SIZE
            )
            if pretrained_results:
                print(f"Successfully processed pretrained model with {pretrained_results['num_layers']} layers")
            else:
                print("Failed to process pretrained model")
    except Exception as e:
        print(f"Error in pretrained model processing: {str(e)}")
        import traceback
        traceback.print_exc()