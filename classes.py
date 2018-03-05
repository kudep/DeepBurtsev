import numpy as np
import pandas as pd
import pymorphy2
import fasttext
import json
import os

from dataset import Dataset
from utils import transform, tokenize
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from models.CNN.multiclass import KerasMulticlassModel
morph = pymorphy2.MorphAnalyzer()


# class IntentRecognition():
#     def __init__(self, Data, standartization_function=None, linear_only=False, neural_only=False):


class Pipeline(object):
    def __init__(self, dataset, config=None):
        if config is None:
            self.config = {'lemma': True,
                           'lower': True
                           'n-gram': False,
                           'vectorization': {'count': True,
                                             'tf-idf': False,
                                             'fasttext': False},
                           'model': {'name': 'LR', 'model_config': None},
                           'fasttext_model': '../embeddings/ft_0.8.3_nltk_yalen_sg_300.bin'}
        else:
            self.config = config
        self.status = ''

        # models
        if self.config['model']['name'] == 'LR':
            self.model = LogisticRegression(n_jobs=-1, solver='lbfgs')
        elif self.config['model']['name'] == 'GBM':
            self.model = LGBMClassifier(n_estimators=200, n_jobs=-1, learning_rate=0.1)
        elif self.config['model']['name'] == 'SVM':
            self.model = LinearSVC(C=0.8873076204728344, class_weight='balanced')
        elif self.config['model']['name'] == 'RF':
            self.model = RandomForestClassifier(max_depth=5, random_state=0)

        # TODO fix CNN model
        elif self.config['model']['name'] == 'CNN':
            # Reading parameters of intent_model from json
            if os.path.isfile(self.config['model']['model_config']):
                with open(self.config['model']['model_config'], "r") as f:
                    self.opt = json.load(f)
                self.model = KerasMulticlassModel(self.opt)
            else:
                raise FileExistsError('File {} is not exist.'.format(self.config['model']['model_config']))
        else:
            raise NotImplementedError('{} is not implemented'.format(self.config['model']['name']))

        # vectorizers
        if self.config['vectorization']['count']:
            self.vectorizer = CountVectorizer(min_df=5)  # tokenizer=self.tokenizer,
            self.config['tokenization'] = False
        elif self.config['vectorization']['tf-idf']:
            self.vectorizer = TfidfVectorizer()  # tokenizer=self.tokenizer
            self.config['tokenization'] = False

        # TODO fix fasttext
        elif self.config['vectorization']['fasttext']:
            self.vectorizer = fasttext.load_model(self.config['fasttext_model'])
            self.config['tokenization'] = True
        else:
            raise NotImplementedError('Not implemented vectorizer.')

    def preprocessing(self, data):
        return transform(data, lower=self.config['lower'], lemma=self.config['lemma'], ngramm=self.config['ngramm'])

    def vectorization(self, data, train=False):
        # vectorization
        if not self.config['tokenization']:
            if train:
                vec = self.vectorizer.fit_transform(data)
            else:
                vec = self.vectorizer.transform(data)
        else:
            data = tokenize(data)
            if self.config['model']['name'] == 'CNN':
                vec = np.zeros((len(data), self.opt['text_size'], self.opt['embedding_size']))
                for j, x in enumerate(data):
                    for i, y in enumerate(x):
                        if i < self.opt['text_size']:
                            vec[j, i] = self.vectorizer[y]
                        else:
                            break
            else:
                raise NotImplementedError()

        self.status += 'Vectorization: done\n'
        return vec

    def train(self, data, y):
        vec = self.vectorization(data, train=True)
        self.model.fit(vec, y)
        self.status += 'Train: done\n'
        return None

    def status(self):
        return self.status

    def config(self):
        return self.config
