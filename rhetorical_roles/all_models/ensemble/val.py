import pickle
from dataset import get_train_val_loaders
from utils import multi_acc, score
import numpy as np

files = [
'bert_base_uncased_val_preds.pkl', 
'deberta_val_preds.pkl', 
# 'scibert_lstm_val_preds.pkl',
'bert_lstm_val_preds.pkl',
'distilbert_val_preds.pkl',
# 'tfidf_val_preds.pkl'
]

_, val_labels = get_train_val_loaders()

model_preds = []
for file in files:
    with open('./model_preds/'+file, 'rb') as f:
        data = pickle.load(f)
    val_preds = [da for dat in data for da in dat] if file!='tfidf_val_preds.pkl' else data
    model_preds.append(val_preds)

model_preds_t = torch.tensor(model_preds, dtype=torch.float64)
mean_model_preds = torch.mean(model_preds_t, 0)

preds = np.argmax(torch.log_softmax(mean_model_preds, dim=1).numpy(), 1)

print(len(val_labels), len(preds))
print(score(np.array(preds), np.array(val_labels)))
print(multi_acc(np.array(preds), np.array(val_labels))) 
print(files)