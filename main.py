import pandas as pd
import numpy as np
import re
import os
import pymorphy2
import fasttext

from os.path import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from utils import tokenization


morph = pymorphy2.MorphAnalyzer()
# загрузка csv файла из датасета
train_dataset = './data/X_train.csv'
test_dataset = './data/X_test.csv'

config = {'clean': {'status': False,
                    'nan': True,
                    'repeat': True},
          'misspelling': {'status': False, 'spellers': None},
          'lemma': {'status': True}}


class Vectorization():
    def __init__(self, config=None):
        if config is None:
            self.config = {'nan': True,
                           'repeat': True,
                           'speller': False,
                           'n-gram': False,
                           'lemma': False,
                           'count': False,
                           'tf-idf': True,
                           'fasttext': False,
                           'model': 'CNN'}
        else:
            self.config = config
        self.status = None

        if self.config['count']:
            self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, min_df=5)
            self.config['tokenization'] = False
        elif self.config['tf-idf']:
            self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer)
            self.config['tokenization'] = False
        elif self.config['fasttext']:
            self.vectorizer = fasttext.load_model()
            self.config['tokenization'] = True
        else:
            raise NotImplementedError('Not implemented vectorizer.')

    def clean(self, data):
        if self.config['nan']:
            data = data.dropna()
        if self.config['repeat']:
            data = data.drop_duplicates()
        return data

    def tokenizer(self, data):
        if not self.config['tokenization']:
            data = tokenization(data, morph=self.config['lemma'], ngram=self.config['n-gram'])
            new_data = list()
            for x in data:
                new_data.append(*map(' '.join, x))
                data = new_data
        else:
            data = tokenization(data, morph=self.config['lemma'], ngram=self.config['n-gram'])
        return data

    def run(self, data, train=False):
        # cleaning dataset from NaN and repeated requests
        data = self.clean(data)

        # fix misspelling
        if self.config['speller']:
            raise NotImplementedError('Speller is not implemented yet')

        # vectorization
        if not self.config['tokenization']:
            if train:
                vec = self.vectorizer.fit_transform(data)
            else:
                vec = self.vectorizer.transform(data)
        else:
            data = self.tokenizer(data)
            if self.config['model'] == 'CNN':
                vec = list()
                for x in data:
                    v = list()
                    for y in x:
                        v.append(self.vectorizer(y))
                    vec.append(np.asarray(v))
                vec = np.asarray(vec)

        self.status = 'Vectorization: {}'.format('Done')
        return vec

    def status(self):
        return self.status

    def config(self):
        return self.config



