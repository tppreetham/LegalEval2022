import torch.optim as optim
import torch
import numpy as np
import os
import random

config = {
    "num_epochs": 5,
    "batch_size": 8,
    "learning_rate": 4e-5,
    "optimizer": optim.AdamW,
    # 'model_name': "ccdv/lsg-bart-large-4096",
    'model_name': "ccdv/lsg-xlm-roberta-base-4096",
    'output_size': 13,
    # 'scheduler': torch.optim.lr_scheduler.OneCycleLR,
    'weight_decay': 0.001,
    "max_length": 512
}

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True