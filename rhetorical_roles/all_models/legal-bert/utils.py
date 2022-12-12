
import sklearn
import numpy as np
import torch.nn as nn
import pickle

def score(preds, actual):
    eval_report = sklearn.metrics.classification_report(actual, preds,labels=[0,1,2], zero_division = 1, output_dict=True)
    return (eval_report["weighted avg"]["precision"], eval_report["weighted avg"]["recall"], eval_report["weighted avg"]["f1-score"])

def multi_acc(y_pred, y_test):    
    
    correct_pred = (y_pred == y_test)
    acc = correct_pred.sum() * 1.0 / len(correct_pred)
    acc = np.round_(acc * 100, decimals = 3)
    return acc

def loss_fn():
    '''
        calculates the loss use CE loss function
    '''
    return nn.CrossEntropyLoss()



def dump_dict(f1_met, loss_met, acc_met, langu):
    with open(f'./results/f1_met_{langu}.pkl', 'wb') as f:
        pickle.dump(f1_met, f)
    
    with open(f'./results/acc_met_{langu}.pkl', 'wb') as f:
        pickle.dump(acc_met, f)
        
    with open(f'./results/loss_met_{langu}.pkl', 'wb') as f:
        pickle.dump(loss_met, f)
    