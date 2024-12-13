import torch
from torch.utils.data import DataLoader
from data.dataset import BandDataset
# from dataset import BandDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast




def load_data(config):
    df = pd.read_pickle('./data/df_aflow_4.pkl') 

    X = df['new_column']
    y = df[['Egap']]#, 'Band_gap_GGA', 'Band_gap_HSE_optical', 'Band_gap_GGA_optical']]
    

    # X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.3)
    

    # X_valid, X_test, y_valid, y_test = train_test_split(X_remain, y_remain, test_size=0.5)



    X_train_plus_remaining, X_test, y_train_plus_remaining, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


    X_train, X_valid, y_train, y_valid = train_test_split(X_train_plus_remaining, y_train_plus_remaining, test_size=0.20, random_state=42)

    tkz = RobertaTokenizerFast.from_pretrained('roberta-base')
    X_train = tkz(X_train.tolist(), padding='longest', truncation=True, return_tensors='pt')
    X_valid = tkz(X_valid.tolist(), padding='longest', truncation=True, return_tensors='pt')
    X_test = tkz(X_test.tolist(), padding='longest', truncation=True, return_tensors='pt')

    train_encodings = {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']}
    valid_encodings = {'input_ids': X_valid['input_ids'], 'attention_mask': X_valid['attention_mask']}
    test_encodings = {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']}
    
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
