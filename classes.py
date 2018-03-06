import pymorphy2
import fasttext
import json
import os

from utils import transform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from models.CNN.multiclass import KerasMulticlassModel
from models.Classic.models import LinearRegression, GBM, SVM, RandomForest
morph = pymorphy2.MorphAnalyzer()


# class IntentRecognition():
#     def __init__(self, Data, standartization_function=None, linear_only=False, neural_only=False):


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
                self.opt['classes'] = dataset.data['classes']
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
        self.model.fit(self.dataset, 'mod')
        self.status += 'Train: done\n'

    def status(self):
        return self.status

    def config(self):
        return self.config
