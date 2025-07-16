import torch
from torch import nn, optim
from transformers import LlamaModel, LlamaConfig

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.base = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")
        hidden_size = self.base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return self.head(mean_embeddings)

def create_model(config):
    model = Network(config).to(config.device)
    
    if hasattr(config, 'freeze_strategy'):
        freeze_layers(model, config.freeze_strategy)
    
    return model

def freeze_layers(model, strategy):

    base_model = model.base if not isinstance(model, nn.DataParallel) else model.module.base
    num_layers = base_model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    for param in model.parameters():
        param.requires_grad = True
    
    if strategy == 'freeze_all':
        # Freeze the entire base model
        for param in base_model.parameters():
            param.requires_grad = False
        print("Froze all layers")
        
    elif strategy == 'freeze_all_but_final':
        # Freeze all layers except the last one
        for i, layer in enumerate(base_model.layers):
            if i < num_layers - 1:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Froze all layers except layer {num_layers-1}")
        
    elif strategy == 'freeze_all_but_final_3':
        # Freeze all layers except the last 3
        for i, layer in enumerate(base_model.layers):
            if i < num_layers - 3:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Froze all layers except the last 3 layers ({num_layers-3}-{num_layers-1})")
        
    elif strategy == 'freeze_first':
        # Freeze only the first layer
        for param in base_model.layers[0].parameters():
            param.requires_grad = False
        print("Froze only the first layer")
        
    elif strategy == 'no_freeze':
        print("No layers frozen")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

def setup_training(config, model, train_loader=None):
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )
    
    return criterion, mae_loss, optimizer, scheduler