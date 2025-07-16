import torch
from torch.utils.data import DataLoader
from data.dataset import BandDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5TokenizerFast, AutoTokenizer

class BandDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def load_data_t5(config, model_size="t5-small", max_length=512):

    df = pd.read_pickle('./data/df_aflow_4.pkl') 

    X = df['new_column']
    y = df[['Egap']]
    
    X_train_plus_remaining, X_test, y_train_plus_remaining, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_plus_remaining, y_train_plus_remaining, test_size=0.20, random_state=42)

    tkz = AutoTokenizer.from_pretrained(model_size)
    
    X_train = ["predict band gap: " + text for text in X_train.tolist()]
    X_valid = ["predict band gap: " + text for text in X_valid.tolist()]
    X_test = ["predict band gap: " + text for text in X_test.tolist()]
    
    print(f"Sample input: {X_train[0][:100]}...")
    
    X_train = tkz(X_train, 
                 padding='max_length', 
                 truncation=True, 
                 max_length=max_length, 
                 return_tensors='pt')
                 
    X_valid = tkz(X_valid, 
                 padding='max_length', 
                 truncation=True, 
                 max_length=max_length, 
                 return_tensors='pt')
                 
    X_test = tkz(X_test, 
                padding='max_length', 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt')
    
    train_encodings = {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']}
    valid_encodings = {'input_ids': X_valid['input_ids'], 'attention_mask': X_valid['attention_mask']}
    test_encodings = {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']}

    train_labels = torch.tensor(y_train[['Egap']].values, dtype=torch.float32).squeeze(-1)
    valid_labels = torch.tensor(y_valid[['Egap']].values, dtype=torch.float32).squeeze(-1)
    test_labels = torch.tensor(y_test[['Egap']].values, dtype=torch.float32).squeeze(-1)

    train_set = BandDataset(train_encodings, train_labels)
    valid_set = BandDataset(valid_encodings, valid_labels)
    test_set = BandDataset(test_encodings, test_labels)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4  # Speed up data transfer to GPU
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )

    print('Batch size: ', config.batch_size)
    print('Max sequence length: ', max_length)
    print('Train dataset samples: ', len(train_set))
    print('Valid dataset samples: ', len(valid_set))
    print('Test dataset samples: ', len(test_set))
    print('Train dataset batches: ', len(train_loader))
    print('Valid dataset batches: ', len(valid_loader))
    print('Test dataset batches: ', len(test_loader))
    print()

    return train_loader, valid_loader, test_loader