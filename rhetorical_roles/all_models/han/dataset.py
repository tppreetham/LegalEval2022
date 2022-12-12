import json
from urllib.request import urlopen
import torch
import pandas as pd
from config import config
import re

TRAIN_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json"
VAL_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"


def preprocess_json(t_json):
    docs = []
    for rec in t_json:
        doc = []
        for line in rec['annotations'][0]['result']:
            doc.append({'text': line['value']['text'], 'label': line['value']['labels'][0] 
            if len(line['value']['labels'])>0 else 'NONE'})
        docs.append(doc)
    MAX_SENT_LEN = config['max_length']
    docs_l, doc_sent_labels = [], []
    unique_tokens = set()
    unique_labels = set()
    
    for doc in docs:
        sents_l = []
        sent_labels_l = []
        for sent in doc:
            sent_l = sent['label']
            sent = sent['text']
            if len(sent) != 0:
                sent = [x.lower() for x in re.findall(r"\w+", sent)]
                if len(sent) >= MAX_SENT_LEN:
                    sent = sent[:MAX_SENT_LEN]
                else:
                    for _ in range(MAX_SENT_LEN - len(sent)):
                        sent.append("<pad>")
                sents_l.append(sent)
                sent_labels_l.append(sent_l)
                unique_tokens.update(sent)
                unique_labels.add(sent_l)
        docs_l.append(sents_l)
        doc_sent_labels.append(sent_labels_l)
    unique_tokens_map = {tok: i for i,tok in enumerate(unique_tokens)}
    unique_labels_map = {tok: i for i,tok in enumerate(unique_labels)}
    PAD_TOKEN_IND = unique_tokens_map['<pad>']
    # print(unique_labels, unique_labels_map)

    max_doc_size = max([len(d) for d in docs_l])
    padded_max_sent = [PAD_TOKEN_IND] * config['max_length']

    for i in range(len(docs_l)):
        for j in range(len(docs_l[i])):
            docs_l[i][j] = [unique_tokens_map[x] for x in docs_l[i][j]]
            if len(docs_l[i])<max_doc_size:
                docs_l[i]+=[padded_max_sent]*(max_doc_size - len(docs_l[i]))

    for i in range(len(doc_sent_labels)):
        doc_sent_labels[i] = [unique_labels_map[x] for x in doc_sent_labels[i]]
        if len(doc_sent_labels[i])<max_doc_size:
            doc_sent_labels[i]+=[padded_max_sent]*(max_doc_size - len(doc_sent_labels[i]))

    return docs_l, doc_sent_labels

class LegalEvalDataset(torch.utils.data.Dataset):
    '''
        Holds the dataset and also does the tokenization part
    '''
    def __init__(self, docs, labels):
        self.docs = docs
        self.labels = labels
        
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index: int):
        doc = self.docs[index]
        labels = self.labels[index]
        print(len(doc), len(labels))
        return [torch.tensor(doc, dtype = torch.long), torch.tensor(labels)]

def get_train_val_loaders(batch_size=8):
    train_json = json.loads(urlopen(TRAIN_DATA).read())
    val_json = json.loads(urlopen(VAL_DATA).read())
    train_docs, train_labels = preprocess_json(train_json)
    val_docs, val_labels = preprocess_json(val_json)

    train_loader = torch.utils.data.DataLoader(
        LegalEvalDataset(train_docs, train_labels), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        LegalEvalDataset(val_docs, val_labels), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"train": train_loader, "val": val_loader, "vocab_size": get_vocab_size(train_docs+val_docs)}

    return dataloaders_dict

def get_vocab_size(docs):
    print(type(docs), type(docs[0]), type(docs[0][0]), type(docs[0][0][0]))
    return max([max(sent) for doc in docs[:2] for sent in doc])

def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        LegalEvalDataset(df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)    
    return loader

df = get_train_val_loaders(8)
d=0
for i,t in df['train']:
    d+=1
    print(i,t)
    if d>10:
        break