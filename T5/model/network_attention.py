# For T5 Model with attention output
import torch
from torch import nn
from transformers import T5Model, T5Config

class T5NetworkOriginal(nn.Module):

    def __init__(self, model_size="t5-small"):

        super(T5NetworkOriginal, self).__init__()       
        config = T5Config.from_pretrained(model_size, output_attentions=True)
        self.base = T5Model.from_pretrained(model_size, config=config)
        self.hidden_size = self.base.config.d_model
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask, output_attentions=None):

        encoder_outputs = self.base.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        prediction = self.head(pooled_output)

        if output_attentions:
            return prediction, encoder_outputs.attentions
        
        return prediction

def create_t5_model(config, model_size="t5-small"):

    t5_config = T5Config.from_pretrained(model_size, output_attentions=True)
    model = T5NetworkOriginal(model_size).to(config.device)

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