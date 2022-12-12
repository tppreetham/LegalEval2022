from tqdm import tqdm
from config import config
import pandas as pd
import torch
from model import DBModel
from utils import loss_fn, score, multi_acc, dump_dict
from dataset import get_train_val_loaders
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import logging

def train():
    '''
        train the model using config as hyperparameters
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    model = DBModel()

    train_loss_epoch, val_loss_epoch = [], []
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
    criterion = loss_fn()   
    dataloaders_dict = get_train_val_loaders(config['batch_size'])
    train_dataloader, val_dataloader = dataloaders_dict['train'], dataloaders_dict['val']
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,total_steps=config['num_epochs'] * len(train_dataloader.dataset)//config['batch_size'])
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    gc.collect()
    torch.cuda.empty_cache()

    f1_met = {
        "train": [],
        "val": []
    }
    loss_met = {
        "train": [],
        "val": []
    }

    for epoch_num in range(config['num_epochs']):
        total_loss_train = 0
        accum_iter = 4

        print(f"**********EPOCH {epoch_num+1}***********")
        preds, actual = [], []
        
        for b_id, td in enumerate(tqdm(train_dataloader)):
            train_label = td['label'].to(device)
            mask = td['attention_mask'].to(device)
            input_id = td['input_ids'].squeeze(1).to(device)

            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():
                    output, batch_loss = model(input_id, mask, labels=train_label)
                    # print(output, train_label.squeeze().long())
                    # assert(len(output) == len(train_label))
                    # batch_loss = criterion(output, train_label.long())
                # print(batch_loss)
                total_loss_train += batch_loss.item()
                train_loss_epoch.append(batch_loss.item())

                #print(torch.max(output, dim = 1)[1])
                preds.append(torch.max(output, dim = 1)[1].cpu().detach().numpy())
                actual.append(train_label.long().cpu().detach().numpy())

                model.zero_grad()
                batch_loss /= accum_iter
                scaler.scale(batch_loss).backward()
                #scheduler.step()

                if ((b_id + 1) % accum_iter == 0) or (b_id + 1 == len(train_dataloader)):
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    
                gc.collect()
                torch.cuda.empty_cache()

        #print(preds,actual)
        train_metrics = score(np.concatenate(preds),np.concatenate(actual))
        train_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        total_loss_val = 0
        preds, actual = [], []

        with torch.no_grad():
            for val_input in tqdm(val_dataloader):

                val_label = val_input['label'].to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, loss = model(input_id, mask, val_label)

                preds.append(torch.max(output, dim = 1)[1].cpu().detach().numpy())
                actual.append(val_label.long().cpu().detach().numpy())


                batch_loss = loss
                total_loss_val += batch_loss.item()
                val_loss_epoch.append(batch_loss.item())
                
        val_metrics = score(np.concatenate(preds),np.concatenate(actual))
        val_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        

        
        f1_met['train'].append(train_metrics[2])
        loss_met['train'].append(total_loss_train / len(train_dataloader.dataset))
        f1_met['val'].append(val_metrics[2])
        loss_met['val'].append(total_loss_val / len(val_dataloader.dataset))


        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
            | Train Accuracy: {train_acc} \
            | Train Metrics (Precision, Recall, F1-Score): {train_metrics} \
            | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
            | Val Accuracy: {val_acc} \
            | Val Metrics (Precision, Recall, F1-Score): {val_metrics}')
            
        torch.save(model.state_dict(), f"./models/{config['model']}_epoch{epoch_num+1}.pth")
                
    dump_dict(f1_met, loss_met, f"new_{config['model']}_epoch{epoch_num + 1}")        
    return

if __name__ == '__main__':
    logging.set_verbosity_warning()
    logging.set_verbosity_error()

    train()