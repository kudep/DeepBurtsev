#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

from models.CNN.preprocessing import NLTKTokenizer
from models.CNN.dataset import Dataset
from models.CNN.multiclass import KerasMulticlassModel
from models.CNN.metrics import fmeasure_

config_file = sys.argv[1]
data_file = './data/vkusvill/vkusvill_all_categories.csv'


def read_dataset(filepath, duplicates=False, clean=True):
    file = open(filepath, 'r', encoding='ISO-8859-1')
    data = pd.read_csv(file)

    old_names = data.keys()
    names = [n.encode('ISO-8859-1').decode('cp1251').encode('utf8') for n in old_names]
    names = [n.decode('utf-8') for n in names]

    new_data = dict()
    for old, new in zip(old_names, names):
        new_data[new] = list()
        for c in data[old]:
            try:
                s = c.encode('ISO-8859-1').decode('cp1251').encode('utf8')
                s = s.decode('utf-8')
                new_data[new].append(s)
            except AttributeError:
                new_data[new].append(c)

    new_data = pd.DataFrame(new_data, columns=['Описание', 'Категория жалобы'])
    new_data.rename(columns={'Описание': 'req', 'Категория жалобы': 'cat'}, inplace=True)
    new_data = new_data.dropna()  # dell nan
    if not duplicates:
        new_data = new_data.drop_duplicates()  # dell duplicates

    # как отдельную ветвь можно использовать
    if clean:
        delete_bad_symbols = lambda x: " ".join(re.sub('[^а-яa-zё0-9]', ' ', x.lower()).split())
        new_data['req'] = new_data['req'].apply(delete_bad_symbols)

    new_data = new_data.reset_index()

    return new_data


# Reading full data
comment_name = "req"
cat_name = 'cat'
train_data = read_dataset(data_file)
print(train_data.head())

# Tokenization that splits words and punctuation by space
preprocessor = NLTKTokenizer()
train_data.loc[:, comment_name] = preprocessor.infer(train_data.loc[:, comment_name].values)

# Reading parameters of intent_model from json
with open(config_file, "r") as f:
    opt = json.load(f)

# Initializing classes from dataset
# columns = [str(x) for x in list(data['cat'].unique())]
columns = list(train_data[cat_name].unique())
# columns.remove(comment_name)
classes = np.array(columns)
opt["classes"] = " ".join([str(x) for x in classes])
# opt["classes"] = classes
print(classes)
print(np.array(opt['classes'].split(" ")))


# Constructing data
data_dict = dict()
train_pairs = []

for i in range(train_data.shape[0]):
    train_pairs.append((train_data[comment_name][i], train_data['cat'][i]))

data_dict["train"] = train_pairs

# Building dataset splitting full dataset on train and valid in proportion 9:1
dataset = Dataset(data=data_dict, seed=42, classes=classes,
                  field_to_split="train", splitted_fields="train valid", splitting_proportions="0.9 0.1")

valid_data = dataset.data['valid']
# print(valid_data)

# data for scoring
scoring_dict = dict()

for x, y in valid_data:
    if y not in scoring_dict.keys():
        scoring_dict[y] = list()
    else:
        scoring_dict[y].append(x)

# path to fasttext model
opt['fasttext_model'] = './embeddings/russian/ft_0.8.3_nltk_yalen_sg_300.bin'
if Path(opt['fasttext_model']).is_file():
    print('All ok')

# Initilizing intent_model with given parameters
print("Initializing intent_model")
model = KerasMulticlassModel(opt)

print('Start scoring ...')
predictions = dict()
for x in scoring_dict.keys():
    predictions[str(x)] = dict()
    preds = model.infer(scoring_dict[x])
    print('Predictions {0}: {1}\n'.format(x, preds.shape))
    inds = np.argmax(preds, axis=1)
    z = np.zeros_like(preds)
    for i in range(len(z)):
        z[i, inds[i]] = 1
    preds = z
    print('Predictions {0}: {1}\n'.format(x, preds))

    y_true = np.zeros_like(z)
    for i in range(len(y_true)):
        y_true[i, x-1] = 1

    print('y_true {0}: {1}\n'.format(x, y_true))
    predictions[str(x)]['F1'], predictions[str(x)]['prec'], predictions[str(x)]['rec'] = fmeasure_(y_true, preds)

with open('./scoring_data.txt', 'w') as f:
    f.write(json.dumps(predictions))

print('Scoring data loads in {}'.format('./scoring_data.txt'))
print('End of scoring.')
