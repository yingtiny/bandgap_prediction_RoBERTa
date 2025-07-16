import torch
from torch import nn, optim
from transformers import RobertaModel, RobertaConfig


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.base = RobertaModel.from_pretrained("roberta-base", config=config, ignore_mismatched_sizes=True)
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(768, 1)
        )

    def forward(self,  input_ids, attention_mask):
        return self.head(self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output)

def create_model(config):
    bert_config = RobertaConfig.from_pretrained('roberta-base')
    bert_config.max_position_embeddings = 514
    bert_config.num_hidden_layers = 12
    bert_config.num_attention_heads = 12
    bert_config.hidden_size = 768
    model = Network(bert_config).to(config.device)

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
