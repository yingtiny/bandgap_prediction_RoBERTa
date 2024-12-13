# RoBERTa-based Semiconductor Band Gap Prediction

This repository implements an approach using RoBERTa to predict semiconductor band gaps from material descriptions. The project demonstrates how pre-trained language models can be effectively applied to materials properties prediction.


1. **Core Model Implementation**
   - Band gap prediction using RoBERTa
   - Custom regression head for prediction
   - Training and evaluation pipeline

2. **Analysis Tools**
   - Visualization of embedding analysis
   - Self-attention analysis

## Repository Structure

```
├── config.yaml              # Model configuration
├── main.py                  # Main training script
├── model/
│   ├── network.py          # Model architecture definition
│   └── utils.py            # Training utilities
├── data/
│   ├── dataloader.py       # Data loading and preprocessing
│   └── dataset.py          # PyTorch dataset implementation
└── analysis/
    ├── attention_scores.py        # Self-attention analysis
    ├── attention_scores_samples.py # Sample-specific self-attention analysis
    ├── 1109_emb.py               # Embedding analysis for band gap
    └── 1109_emb_c.py             # Embedding analysis for crystal system
```

## Features

### 1. Data Processing
- Loads AFLOW dataset with material descriptions
- Implements efficient data loading and preprocessing
- Handles train/validation/test splitting

### 2. Model Architecture
- Base: RoBERTa pre-trained model
- Custom regression head for band gap prediction
- Attention mechanism for interpretability

### 3. Analysis Capabilities

#### Embedding Analysis (`1109_emb.py`, `1109_emb_c.py`)
- Visualizes learned embeddings using t-SNE
- Analyzes clustering by crystal systems

#### Attention Analysis (`attention_scores.py`, `attention_scores_samples.py`)
- Visualizes attention patterns across different material properties categories
- Feature importance through attention weights
- Compares pre-trained vs fine-tuned model attention

#### `1109_emb_c.py`
- Analyzes embeddings specifically for crystal systems
- Creates t-SNE visualizations colored by crystal structure
- Generates distribution analysis of crystal systems

#### `attention_scores_samples.py`
- Provides detailed attention analysis for specific samples
- Compares attention patterns between pre-trained and fine-tuned models
- Generates attention heatmaps for different material properties

## Setup and Usage

1. Install dependencies:
```bash
pip install torch transformers wandb easydict tqdm sklearn matplotlib seaborn
```


2. Train the model:
```bash
python main.py
```

3. Run analysis:
```bash
# Embedding analysis
python 1109_emb.py

# Crystal system analysis
python 1109_emb_c.py

# Attention pattern analysis
python attention_scores.py
```
