#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

from models.CNN.preprocessing import NLTKTokenizer
from models.CNN.dataset import Dataset
from models.CNN.multiclass import KerasMulticlassModel
from models.CNN.metrics import fmeasure_
from utils import HyperPar


def read_dataset_fromV(filepath, duplicates=False, clean=True):
    file = open(filepath, 'r')
    data = pd.read_csv(file)

    new_data = pd.DataFrame(data, columns=['Описание', 'Категория жалобы'])
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


def preproces(data, comment_name="req"):
    # Constructing data
    data_dict = dict()
    classes = None

    for j, x in enumerate(data):
        train_data = read_dataset_fromV(x)
        print(train_data.head())

        # Tokenization that splits words and punctuation by space
        preprocessor = NLTKTokenizer()
        train_data.loc[:, comment_name] = preprocessor.infer(train_data.loc[:, comment_name].values)

        # Initializing classes from dataset
        # columns = [str(x) for x in list(data['cat'].unique())]
        columns = list(train_data['cat'].unique())
        # columns.remove(comment_name)
        classes = np.array(columns)
        classes = " ".join(list(str(classes)))
        # opt["classes"] = classes
        print(classes)
        train_pairs = []

        for i in range(train_data.shape[0]):
            train_pairs.append((train_data[comment_name][i], train_data['cat'][i]))

        if j == 0:
            data_dict["train"] = train_pairs
        elif j == 1:
            data_dict["test"] = train_pairs
        else:
            raise ValueError

    return [data_dict, classes]


def train(data, config):
    # Building dataset splitting full dataset on train and valid in proportion 9:1
    dataset = Dataset(data=data[0], seed=42, classes=data[1],
                      field_to_split="train", splitted_fields="train valid", splitting_proportions="0.9 0.1")
    valid_data = dataset.data['valid']
    test_data = data[0]['test']

    scoring_dict = dict()
    for x, y in test_data:
        if y not in scoring_dict.keys():
            scoring_dict[y] = list()
        else:
            scoring_dict[y].append(x)

    # path to fasttext model
    config['classes'] = data[1]
    config['fasttext_model'] = './embeddings/russian/ft_0.8.3_nltk_yalen_sg_300.bin'
    if Path(config['fasttext_model']).is_file():
        print('All ok')

    # Initilizing intent_model with given parameters
    print("Initializing intent_model")
    model = KerasMulticlassModel(config)

    # Training intent_model on the given dataset
    print("Training intent_model")
    model.train(dataset=dataset)

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

    return None


def parsearch(xtrain, xtest, n):

    dataset = preproces([xtrain, xtest])
    net_params = HyperPar(filters_cnn={'range': [100, 300], 'discrete': True},
                          dense_size={'range': [50, 200], 'discrete': True},
                          batch_size={'range': [50, 200], 'discrete': True},
                          confident_threshold={'range': [0.1, 1.0], 'discrete': True},
                          lear_rate={'range': [0.001, 0.1], 'discrete': True},
                          lear_rate_decay={'range': [0.01, 0.1], 'discrete': True},
                          text_size={'range': [10, 50], 'discrete': True},
                          coef_reg_cnn={'range': [1e-2, 1e-6], 'discrete': True},
                          coef_reg_den={'range': [1e-2, 1e-6], 'discrete': True},
                          dropout_rate={'range': [0.1, 0.8], 'discrete': True},
                          epochs={'range': [80, 100], 'discrete': True},
                          model_path="./models/CNN/checkpoints/test/",
                          kernel_sizes_cnn="1 2 3",
                          embedding_size=300,
                          lear_metrics="binary_accuracy fmeasure",
                          model_from_saved=False,
                          optimizer='Adam',
                          loss="categorical_crossentropy",
                          fasttext_model="",
                          model_name="cnn_model",
                          val_every_n_epochs=5,
                          verbose=True,
                          show_examples=False,
                          val_patience=5)

    for i in range(n):
        configuration = net_params.sample_params()
        train(dataset, configuration)

    return None


if __name__ == '__main__':
    Xtr = './data/vkusvill/X_train.csv'
    xts = './data/vkusvill/X_test.csv'
    n = 10
    parsearch(Xtr, xts, n)


