import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from config import config

class LSGBARTEvalModel(nn.Module):
    def __init__(self):
        super(LSGBARTEvalModel, self).__init__() 
        mconfig = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=13
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            # trust_remote_code=True, 
            # num_global_tokens=1,
            # block_size=32,
            # sparse_block_size=32,
            # attention_probs_dropout_prob=0.0,
            # sparsity_factor=8,
            # sparsity_type="none",
            # mask_first_token=True,
            # trust_remote_code=True, 
            # pool_with_global=True,
            config=mconfig
        )
        
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels = None):
        if labels!=None:
            loss, x = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict = False)
            return self.relu(x), loss
        x = self.model(input_ids, attention_mask, return_dict = False)
        x = self.relu(x)
        return x