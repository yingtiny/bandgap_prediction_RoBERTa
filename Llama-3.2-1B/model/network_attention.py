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
    
    def forward(self, input_ids, attention_mask, output_attentions=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  
        mean_embeddings = sum_embeddings / sum_mask
        prediction = self.head(mean_embeddings)
        if output_attentions:
            return prediction, outputs.attentions
        
        return prediction

def create_model(config):
    model = Network(config).to(config.device)
    return model

def setup_training(config, model):
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
        patience=5,
        verbose=True
    )
    
    return criterion, mae_loss, optimizer, scheduler