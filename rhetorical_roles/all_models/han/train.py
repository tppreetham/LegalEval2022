from tqdm import tqdm
from config import config
import torch.nn as nn
import torch
from model import HANModel, wordEncoder, sentEncoder
from utils import loss_fn, score, multi_acc, dump_dict
from dataset import get_train_val_loaders
import gc
import numpy as np
from transformers import logging


class EarlyStopping():
    def __init__(self, tolerance=2, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


def train():
    '''
        train the model using config as hyperparameters
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    dataloaders_dict = get_train_val_loaders(config['batch_size'])
    VOCAB_SIZE = dataloaders_dict['vocab_size']
    HIDDEN_SIZE = config['hidden_size']
    EMBEDDING_DIM = config['embedding_dim']
    LEARNING_RATE = config['learning_rate']
    NUM_CLASSES = config['output_size']

    word_encoder = wordEncoder(VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM).to(device)
    sent_encoder = sentEncoder(HIDDEN_SIZE * 2).to(device)
    model = HANModel(word_encoder, sent_encoder, NUM_CLASSES, device).to(device)

    # loss = []
    # weights = []

    # for i in tqdm(range(config['num_epochs'])):
    # current_loss = 0
    # for j in range(len(tweets)):
    #     tweet, score = torch.tensor(tweets[j], dtype = torch.long).to(DEVICE), torch.tensor(sent_scores[j]).to(DEVICE)
    #     word_weights, sent_weights, output = model(tweet)

    #     optimizer.zero_grad()
    #     current_loss += criterion(output.unsqueeze(0), score.unsqueeze(0))
    #     current_loss.backward(retain_graph=True)
    #     optimizer.step()

    # loss.append(current_loss.item()/(j+1))

    train_loss_epoch, val_loss_epoch = [], []
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
    criterion = loss_fn()   
    train_dataloader, val_dataloader = dataloaders_dict['train'], dataloaders_dict['val']
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

    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    for epoch_num in range(config['num_epochs']):
        total_loss_train = 0
        accum_iter = 4

        print(f"**********EPOCH {epoch_num+1}***********")
        preds, actual = [], []
        
        for b_id, td in enumerate(tqdm(train_dataloader)):
            texts, labels = td
            train_texts = texts.to(device)
            train_label = labels.to(device)

            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():
                    _, _, output = model(train_texts)
                    batch_loss = criterion(output.unsqueeze(0), train_label.unsqueeze(0))
                total_loss_train += batch_loss.item()
                train_loss_epoch.append(batch_loss.item())

                #log_softmax = torch.log_softmax(output, dim=1).cpu().detach()

                tor_max = torch.max(output.cpu().detach(), dim=1)[1]
                preds.append(tor_max.numpy())
                actual.append(train_label.long().cpu().detach().numpy())

                model.zero_grad()
                batch_loss /= accum_iter
                scaler.scale(batch_loss).backward()

                if ((b_id + 1) % accum_iter == 0) or (b_id + 1 == len(train_dataloader)):
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    
                gc.collect()
                torch.cuda.empty_cache()

        train_metrics = score(np.concatenate(preds),np.concatenate(actual))
        #print(len(np.concatenate(preds)), len(np.concatenate(actual)))
        train_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        total_loss_val = 0
        preds, actual = [], []

        with torch.no_grad():
            for val_input in tqdm(val_dataloader):

                texts, labels = td
                val_texts = texts.to(device)
                val_label = labels.to(device)


                _, _, output = model(val_texts)

                #torch.log_softmax(output, dim=1).cpu().detach()
                preds.append(torch.max(output.cpu().detach(),dim=1)[1].numpy())
                actual.append(val_label.long().cpu().detach().numpy())


                batch_loss = criterion(output.unsqueeze(0), val_label.unsqueeze(0))
                total_loss_val += batch_loss.item()
                val_loss_epoch.append(batch_loss.item())
                
        val_metrics = score(np.concatenate(preds),np.concatenate(actual))
        val_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        
        train_loss, val_loss = total_loss_train / len(train_dataloader.dataset), total_loss_val / len(val_dataloader.dataset)
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("#########################")
            print(f"We are at epoch: {epoch_num+1} and are stopping")
            break

        
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
            
        torch.save(model.state_dict(), f"./models/longformer_epoch{epoch_num+1}.pth")
                
    dump_dict(f1_met, loss_met, f"longformer_epoch{epoch_num + 1}")        
    return

if __name__ == '__main__':
    logging.set_verbosity_warning()
    logging.set_verbosity_error()

    train()