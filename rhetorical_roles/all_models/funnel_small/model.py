import torch.nn as nn
from transformers import FunnelModel
from config import config
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import sklearn
import torch
import numpy as np

class LegalEvalModel(nn.Module):
    def __init__(self):
        super(LegalEvalModel, self).__init__() 
        self.bert = FunnelModel.from_pretrained(config['model_name'], return_dict=False)
        self.hidden_size = self.bert.config.hidden_size
        self.LSTM = nn.LSTM(self.hidden_size,self.hidden_size,bidirectional=True)
        self.clf = nn.Linear(self.hidden_size*2,config['output_size'])

    def forward(self, input_ids, attention_mask, len_tokens):
        encoded_layers = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        encoded_layers = encoded_layers[0]
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, len_tokens))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = F.dropout(output_hidden,0.2)
        output = self.clf(output_hidden)
        return torch.sigmoid(output)
