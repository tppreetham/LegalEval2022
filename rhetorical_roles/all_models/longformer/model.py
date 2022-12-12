import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers import LongformerModel, LongformerConfig
from config import config
import sklearn
import torch
import numpy as np

class LFModel(nn.Module):
    def __init__(self):
        super(LFModel, self).__init__() 
        #configuration = LongformerConfig(attention_window = 512)
        self.model = AutoModel.from_pretrained("jpwahle/longformer-base-plagiarism-detection")
        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.o = nn.Linear(64, 13)

        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask)
        x = self.dropout(x.pooler_output)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.o(x)
        x = self.relu(x)
        return x