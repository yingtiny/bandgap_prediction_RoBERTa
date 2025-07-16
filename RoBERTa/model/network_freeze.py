import torch
from torch import nn, optim
from transformers import RobertaModel, RobertaConfig
import torch.utils.checkpoint as checkpoint

class Network(nn.Module):
    def __init__(self, config, freeze_layers=None):
        super(Network, self).__init__()

        self.base = RobertaModel.from_pretrained("roberta-base", config=config, ignore_mismatched_sizes=True)
        self.dropout1 = nn.Dropout(0.3) 
        self.dropout2 = nn.Dropout(0.3) 
        self.dropout3 = nn.Dropout(0.3) 
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            self.dropout1,
            nn.Linear(768, 768),
            nn.ReLU(),
            self.dropout2,
            nn.Linear(768, 768),
            nn.ReLU(),
            self.dropout3,
            nn.Linear(768, 1)
        )

    def forward(self,  input_ids, attention_mask):
        pooled_output = self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        return self.head(pooled_output)

def create_model(config, freeze_layers=None):
    """
    Create the model with the specified configuration and freeze layers if provided.
    Args:
        config: Configuration object containing model parameters.
        freeze_layers: List of layer indices to freeze. If None, no layers are frozen.
    Returns:
        model: An instance of the Network class with the specified configuration.
    """
    bert_config = RobertaConfig.from_pretrained('roberta-base')
    bert_config.max_position_embeddings = 514
    bert_config.num_hidden_layers = 12
    bert_config.num_attention_heads = 12
    bert_config.hidden_size = 768
    model = Network(bert_config, freeze_layers).to(config.device)

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




    


