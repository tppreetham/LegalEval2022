import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from config import config
import sklearn
import torch
import numpy as np

class DBModel(nn.Module):
    def __init__(self):
        super(DBModel, self).__init__() 
        mconfig = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=13
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            config=mconfig,
            ignore_mismatched_sizes = True
        )
        
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels = None):
        if labels!=None:
            loss, x = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict = False)
            #print(x)
            return self.relu(x), loss
        x = self.model(input_ids, attention_mask, return_dict = False)
        x = self.relu(x)
        return x