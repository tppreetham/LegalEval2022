import json
from urllib.request import urlopen
import torch
from transformers import BertTokenizer
import pandas as pd
from config import config
from sklearn import preprocessing

TRAIN_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json"
VAL_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"

def json_to_df(t_json):
    docs = []
    for rec in t_json:
        doc = []
        for line in rec['annotations'][0]['result']:
            doc.append({'text': line['value']['text'], 'label': line['value']['labels'][0] 
            if len(line['value']['labels'])>0 else 'NONE'})
        docs.append(doc)
    docs_df = pd.json_normalize([item for sublist in docs for item in sublist])
    return docs_df

class LegalEvalDataset(torch.utils.data.Dataset):
    '''
        Holds the dataset and also does the tokenization part
    '''
    def __init__(self, df, max_len=512):
        self.df = df
        self.max_len = max_len
        le = preprocessing.LabelEncoder()
        self.labels = le.fit(self.df.label).classes_
        self.df.label = le.transform(self.df.label)
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'], do_lower_case = True, use_fast = False)

    def __len__(self):
        return len(self.df)

    def get_token_mask(self,text):
        
        tokens = []
        mask = []
        text = self.tokenizer.encode(text)
        size = len(text)
        pads = self.tokenizer.encode(['PAD']*(max(0,self.max_len-size)))
        tokens[:max(self.max_len,size)] = text[:max(self.max_len,size)]
        tokens = tokens + pads[1:-1]
        mask = [1]*size+[0]*len(pads[1:-1])
        tokens_len = len(tokens)
        
        return tokens,mask,tokens_len

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]

        text = data_row.text
        label = data_row.label

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=label,
            tokens_len = encoding["input_ids"].flatten().shape[0]
        )

def get_train_val_loaders(batch_size=8):
    train_json = json.loads(urlopen(TRAIN_DATA).read())
    val_json = json.loads(urlopen(VAL_DATA).read())
    train_df = json_to_df(train_json)
    val_df = json_to_df(val_json)

    train_loader = torch.utils.data.DataLoader(
        LegalEvalDataset(train_df), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        LegalEvalDataset(val_df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        LegalEvalDataset(df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)    
    return loader