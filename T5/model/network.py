import torch
from torch import nn
from transformers import T5Model, T5Config

class T5NetworkOriginal(nn.Module):

    def __init__(self, model_size="t5-small"):

        super(T5NetworkOriginal, self).__init__()
        self.base = T5Model.from_pretrained(model_size)
        self.hidden_size = self.base.config.d_model
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask):

        encoder_outputs = self.base.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]

        return self.head(pooled_output)
        
def create_t5_model(config, model_size="t5-small"):

    model = T5NetworkOriginal(model_size)
    
    return model

def setup_training(config, model):
    
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=0.01  
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  
        factor=0.1,  
        patience=5,  
        verbose=True  
    )

    return criterion, mae_loss, optimizer, scheduler
