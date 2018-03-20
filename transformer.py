import json
import pandas as pd
import nltk
import pymorphy2
import fasttext
import numpy as np
from utils import labels2onehot_one

from deeppavlov.core.commands.infer import build_model_from_config


class BaseTransformer(object):

    def get_params(self):
        return self.params

    def set_params(self, params):
        # self.params = params
        self.__init__(params)
        return self


class Speller(BaseTransformer):
    def __init__(self, params=None):
        if params is None:
            self.conf_path = '/home/mks/projects/intent_classification_script/DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json'
        else:
            self.conf_path = params

        with open(self.conf_path) as config_file:
            self.config = json.load(config_file)

        self.speller = build_model_from_config(self.config)
        self.info = dict(type='transformer')

    def transform(self, dataset, name='base'):
        names = dataset.main_names
        data = dataset.data[name]

        refactor = list()
        for x in data[names[0]]:
            refactor.append(self.speller([x])[0])

        dataset.data[name] = pd.DataFrame({names[0]: refactor,
                                           names[1]: data[names[1]]})

        return dataset


class Tokenizer(BaseTransformer):
    def __init__(self, params=None):
        self.info = {}
        self.params = params
        self.info = dict(type='transformer')

    def transform(self, dataset, name='base'):
        names = dataset.main_names
        data = dataset.data[name][names[0]]

        tok_data = list()
        for x in data:
            sent_toks = nltk.sent_tokenize(x)
            word_toks = [nltk.word_tokenize(el) for el in sent_toks]
            tokens = [val for sublist in word_toks for val in sublist]
            tok_data.append(tokens)

        dataset.data[name] = pd.DataFrame({names[0]: tok_data,
                                           names[1]: dataset.data[name][names[1]]})

        return dataset


class Lemmatizer(BaseTransformer):
    def __init__(self, params=None):
        self.params = params
        self.morph = pymorphy2.MorphAnalyzer()
        self.info = dict(type='transformer')

    def transform(self, dataset, name='base'):
        names = dataset.main_names
        data = dataset.data[name][names[0]]

        morph_data = list()
        for x in data:
            mp_data = [self.morph.parse(el)[0].normal_form for el in x]
            morph_data.append(mp_data)

        dataset.data[name] = pd.DataFrame({names[0]: morph_data,
                                           names[1]: dataset.data[name][names[1]]})

        return dataset


class FasttextVectorizer(BaseTransformer):
    # TODO необходимо прописать логику того что векторизация может быть только после разделения датасета
    def __init__(self, params=None):
        if params is None:
            self.params = {'path_to_model': '/home/mks/projects/intent_classification_script/data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin',
                           'dimension': 300,
                           'file_type': 'bin'}

        self.vectorizer = fasttext.load_model(self.params['path_to_model'])
        self.info = dict(type='transformer')

    def transform(self, dataset, name='base'):
        names = dataset.main_names
        data = dataset.data[name][names[0]]

        vec_request = []
        for x in data:
            matrix_i = np.zeros((len(x), self.params['dimension']))
            for j, y in enumerate(x):
                matrix_i[j] = self.vectorizer[y]
            vec_request.append(matrix_i)

        vec_report = list(labels2onehot_one(dataset.data[name][names[1]], dataset.classes))

        dataset.data[name] = pd.DataFrame({names[0]: vec_request,
                                           names[1]: vec_report})

        return dataset


