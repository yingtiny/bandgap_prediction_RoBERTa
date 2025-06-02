import torch
from torch.utils.data import DataLoader
from data.dataset import BandDataset
# from dataset import BandDataset
import pandas as pd
from sklearn.model_selection import train_test_split
# from transformers import RobertaTokenizerFast

from transformers import AutoTokenizer #, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")




# First, fix the tokenizer in load_data
def load_data(config):
    df = pd.read_pickle('./data/df_aflow_4.pkl')
    X = df['new_column']
    y = df[['Egap']]
    
    # Split data
    X_train_plus_remaining, X_test, y_train_plus_remaining, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_plus_remaining, y_train_plus_remaining, test_size=0.20, random_state=42
    )
    
    # Use the Llama tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # You may need to set padding token if it's not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize data - adjust max_length as needed for your data
    X_train = tokenizer(X_train.tolist(), padding='max_length', truncation=True, 
                       max_length=512, return_tensors='pt')
    X_valid = tokenizer(X_valid.tolist(), padding='max_length', truncation=True, 
                       max_length=512, return_tensors='pt')  
    X_test = tokenizer(X_test.tolist(), padding='max_length', truncation=True,
                      max_length=512, return_tensors='pt')
    
    # Continue with dataset creation as before
    train_encodings = {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']}
    valid_encodings = {'input_ids': X_valid['input_ids'], 'attention_mask': X_valid['attention_mask']}
    test_encodings = {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']}
    
    # Rest of your function remains the same
    # ...
    train_labels = torch.tensor(y_train[['Egap']].values)
    valid_labels = torch.tensor(y_valid[['Egap']].values)
    test_labels = torch.tensor(y_test[['Egap']].values)
    
    train_set = BandDataset(train_encodings, train_labels)
    valid_set = BandDataset(valid_encodings, valid_labels)
    test_set = BandDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    print('Batch size: ', config.batch_size)
    print('Train dataset samples: ', len(train_set))
    print('Valid dataset samples: ', len(valid_set))
    print('Test dataset samples: ', len(test_set))
    print('Train dataset batches: ', len(train_loader))
    print('Valid dataset batches: ', len(valid_loader))
    print('Test dataset batches: ', len(test_loader))
    print()
    
    return train_loader, valid_loader, test_loader
