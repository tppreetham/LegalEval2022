import torch.nn as nn
from transformers import AutoConfig
from transformers import BertModel
from config import config
import sklearn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class BertLSTMEvalModel(nn.Module):
    def __init__(self):
        super(BertLSTMEvalModel, self).__init__() 
        mconfig = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=13
        )
        self.model = BertModel.from_pretrained(
            config['model_name'],
            return_dict=False
            #config=mconfig
        )
        self.dropout = nn.Dropout(0.3)
        self.LSTM = nn.LSTM(768,768,bidirectional=True)
        self.clf = nn.Linear(768*2,13)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, tokens_len):
        encoded_layers, pooled_output= self.model(input_ids=input_ids,attention_mask=attention_mask)
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, tokens_len))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        output = self.clf(output_hidden)
    
        return self.relu(output)