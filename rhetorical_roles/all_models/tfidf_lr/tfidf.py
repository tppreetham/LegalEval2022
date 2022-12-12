import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json"
train_json = json.loads(urlopen(TRAIN_DATA).read())
VAL_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"
val_json = json.loads(urlopen(VAL_DATA).read())

def preprocess_json(t_json):
    docs = []
    for rec in t_json:
        doc = []
        for line in rec['annotations'][0]['result']:
            doc.append({'text': line['value']['text'], 'label': line['value']['labels'][0] 
            if len(line['value']['labels'])>0 else 'NONE'})
        docs.append(doc)
    docs_df = pd.json_normalize([item for sublist in docs for item in sublist])
    le = preprocessing.LabelEncoder()
    le.fit(docs_df.label)
    sd = le.transform(docs_df.label)
    docs_df.label = sd
    return docs_df, docs


def multi_acc(y_pred, y_test): 
    print(len(y_pred), len(y_test))   
    correct_pred = (y_pred == y_test)
    print(correct_pred)
    acc = correct_pred.sum() * 1.0 / len(correct_pred)
    acc = np.round_(acc * 100, decimals = 3)
    return acc


t_docs_df, t_docs = preprocess_json(train_json)
v_docs_df, v_docs = preprocess_json(val_json)

# ndocs = t_docs
# label_text_map = {}
# for i in range(len(ndocs)):
#     for j in range(len(ndocs[i])):
#         l, t = ndocs[i][j]['label'], ndocs[i][j]['text']
#         if l in label_text_map:
#             label_text_map[l].append(t)
#         else:
#             label_text_map[l] = [t]


# text_l, label_l = [], []
# for k,v in label_text_map.items():
#     text_l.append(' '.join(v))
#     label_l.append(k)
# assert(len(text_l) == len(label_l))

xtrain, xval = list(t_docs_df['text']), list(v_docs_df['text'])
ytrain, yval = np.array(t_docs_df['label']), np.array(v_docs_df['label'])

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(xtrain+xval)
xtrain_tfv = tfv.transform(xtrain) 
xval_tfv = tfv.transform(xval)

clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xval_tfv)
y_pred = [np.argmax(p) for p in predictions]


print('accuracy: ', multi_acc(np.array(y_pred), yval))