import pandas as pd
import pymorphy2
import fasttext
import json
import os
import re

from dataset import Dataset
from utils import transform, logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from models.Classic.models import LinearRegression, GBM, SVM, RandomForest
from models.CNN.multiclass import KerasMulticlassModel
morph = pymorphy2.MorphAnalyzer()


# class IntentRecognition():
#     def __init__(self, Data, standartization_function=None, linear_only=False, neural_only=False):


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
    new_data.rename(columns={'Описание': 'request', 'Категория жалобы': 'class'}, inplace=True)
    new_data = new_data.dropna()  # dell nan
    if not duplicates:
        new_data = new_data.drop_duplicates()  # dell duplicates

    # как отдельную ветвь можно использовать
    if clean:
        delete_bad_symbols = lambda x: " ".join(re.sub('[^а-яa-zё0-9]', ' ', x.lower()).split())
        new_data['request'] = new_data['request'].apply(delete_bad_symbols)

    new_data = new_data.reset_index()
    new_data = new_data.drop('index', axis=1)

    return new_data


class Pipeline(object):
    def __init__(self, dataset, config=None):

        self.dataset = dataset

        if config is None:
            self.config = {'lemma': True,
                           'lower': True,
                           'ngramm': False,
                           'vectorization': {'count': True,
                                             'tf-idf': False,
                                             'fasttext': False},
                           'model': {'name': 'LR', 'model_config': None},
                           'fasttext_model': '../embeddings/ft_0.8.3_nltk_yalen_sg_300.bin'}
        else:
            self.config = config
        self.status = ''

        self.opt = None

        # vectorizers
        if self.config['vectorization']['count']:
            self.vectorizer = CountVectorizer(min_df=5)  # tokenizer=self.tokenizer,
        elif self.config['vectorization']['tf-idf']:
            self.vectorizer = TfidfVectorizer()  # tokenizer=self.tokenizer
        elif self.config['vectorization']['fasttext']:
            self.vectorizer = fasttext.load_model(self.config['fasttext_model'])
        else:
            raise NotImplementedError('Not implemented vectorizer.')

        # models
        if self.config['model']['name'] == 'LR':
            self.model = LinearRegression(self.vectorizer)
        elif self.config['model']['name'] == 'GBM':
            self.model = GBM(self.vectorizer)
        elif self.config['model']['name'] == 'SVM':
            self.model = SVM(self.vectorizer)
        elif self.config['model']['name'] == 'RF':
            self.model = RandomForest(self.vectorizer)
        elif self.config['model']['name'] == 'CNN':
            # Reading parameters of intent_model from json
            if os.path.isfile(self.config['model']['model_config']):
                with open(self.config['model']['model_config'], "r") as f:
                    self.opt = json.load(f)
                self.opt['classes'] = ' '.join([str(x) for x in dataset.data['classes']])
                self.model = KerasMulticlassModel(self.opt, self.vectorizer)
            else:
                raise FileExistsError('File {} is not exist.'.format(self.config['model']['model_config']))
        else:
            raise NotImplementedError('{} is not implemented'.format(self.config['model']['name']))

    def preprocessing(self, data):
        return transform(data, lower=self.config['lower'], lemma=self.config['lemma'], ngramm=self.config['ngramm'])

    def run(self):
        self.dataset.data['test']['mod'] = self.preprocessing(self.dataset.data['test']['base'])
        self.dataset.data['valid']['mod'] = self.preprocessing(self.dataset.data['valid']['base'])
        self.dataset.data['train']['mod'] = self.preprocessing(self.dataset.data['train']['base'])
        self.status += 'Data transformation: done\n'
        adres = self.model.fit(self.dataset, mode='train', stage='mod')
        results = self.model.test(self.dataset, stage='mod')
        self.status += 'Train: done\n'

        logging(results, self.config, self.opt, adres)

    def status(self):
        return self.status

    def config(self):
        return self.config


# testing
path = '/home/mks/projects/intent_classification_script/data/vkusvill_all_categories.csv'
global_data = read_dataset(path)
dataset = Dataset(global_data, seed=42)
conf = {'lemma': False,
        'lower': False,
        'ngramm': False,
        'vectorization': {'count': False,
                          'tf-idf': False,
                          'fasttext': True},
        'model': {'name': 'CNN', 'model_config': '../models/CNN/config.json'},  # '../models/CNN/config.json'
        'fasttext_model': '../embeddings/ft_0.8.3_nltk_yalen_sg_300.bin'}

pipe = Pipeline(dataset, conf)

pipe.run()

