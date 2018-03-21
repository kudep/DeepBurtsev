import json
import pandas as pd
import nltk
import pymorphy2
import fasttext
import numpy as np

from tqdm import tqdm
from utils import labels2onehot_one
from deeppavlov.core.commands.infer import build_model_from_config


class BaseTransformer(object):
    def __init__(self, info=None, request_names=None, new_names=None):
        # info resist
        if isinstance(info, list):
            self.info = {'op_type': info[0], 'name': info[1]}
        elif isinstance(info, dict):
            #             if set(['op_type', 'name']) in info.keys():
            if ('op_type' in info.keys()) and ('name' in info.keys()):
                self.info = info
            else:
                raise ValueError('Attribute info dict must contain fields "op_type" and "name",'
                                 'but {} was found.'.format(info.keys()))
        elif info is None:
            self.info = {'op_type': 'transformer', 'name': 'op_'}
        else:
            raise ValueError('Attribute info must be list, dict or None, but {} was found.'.format(type(info)))

        # named spaces
        self.new_names = new_names
        self.worked_names = request_names
        self.request_names = []

    def _validate_names(self, dataset):
        if self.worked_names is not None:
            if not isinstance(self.worked_names, list):
                raise ValueError('Request_names must be a list, but {} was found.'.format(type(self.worked_names)))

            for name in self.worked_names:
                if name not in dataset.data.keys():
                    raise KeyError('Key {} not found in dataset.'.format(name))
                else:
                    self.request_names.append(name)
        else:
            self.worked_names = ['base', 'train', 'valid', 'test']
            for name in self.worked_names:
                if name in dataset.data.keys():
                    self.request_names.append(name)
            if len(self.request_names) == 0:
                raise KeyError('Keys from {} not found in dataset.'.format(self.worked_names))

        if self.new_names is None:
            self.new_names = self.request_names

        return self

    def transform(self, dataset):
        self._validate_names(dataset)
        return self._transform(dataset)

    def get_params(self):
        return self.params

    def set_params(self, params):
        # self.params = params
        self.__init__(params)
        return self


class Speller(BaseTransformer):
    def __init__(self, params=None, info=None, request_names=None, new_names=None):
        super().__init__(info, request_names, new_names)

        if params is None:
            self.conf_path = '/home/mks/projects/intent_classification_script/DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json'
        else:
            if isinstance(params, dict):
                self.conf_path = params['path']
            else:
                raise ValueError('Attribute params must be dict, but {} was found.'.format(type(params)))

        with open(self.conf_path) as config_file:
            self.config = json.load(config_file)

        self.speller = build_model_from_config(self.config)

    def _transform(self, dataset):
        print('[ Speller start working ... ]')

        request, report = dataset.main_names
        for name, new_name in zip(self.request_names, self.new_names):
            data = dataset.data[name]
            refactor = list()

            for x in tqdm(data[request]):
                refactor.append(self.speller([x])[0])

            dataset.data[new_name] = pd.DataFrame({request: refactor,
                                                   report: data[report]})

        print('[ Speller done. ]')
        return dataset


class Tokenizer(BaseTransformer):
    def __init__(self, params=None, info=None, request_names=None, new_names=None):
        self.params = params
        super().__init__(info, request_names, new_names)

    def _transform(self, dataset):
        print('[ Starting tokenization ... ]')

        request, report = dataset.main_names
        for name, new_name in zip(self.request_names, self.new_names):
            data = dataset.data[name][request]
            tok_data = list()

            for x in tqdm(data):
                sent_toks = nltk.sent_tokenize(x)
                word_toks = [nltk.word_tokenize(el) for el in sent_toks]
                tokens = [val for sublist in word_toks for val in sublist]
                tok_data.append(tokens)

            dataset.data[new_name] = pd.DataFrame({request: tok_data,
                                                   report: dataset.data[name][report]})

        print('[ Tokenization was done. ]')
        return dataset


class Lemmatizer(BaseTransformer):
    def __init__(self, params=None, info=None, request_names=None, new_names=None):
        self.params = params
        self.morph = pymorphy2.MorphAnalyzer()
        super().__init__(info, request_names, new_names)

    def _transform(self, dataset):
        print('[ Starting lemmatization ... ]')
        request, report = dataset.main_names
        for name, new_name in zip(self.request_names, self.new_names):
            data = dataset.data[name][request]
            morph_data = list()

            for x in tqdm(data):
                mp_data = [self.morph.parse(el)[0].normal_form for el in x]
                morph_data.append(mp_data)

            dataset.data[new_name] = pd.DataFrame({request: morph_data,
                                                   report: dataset.data[name][report]})
        print('[ Ended lemmatization. ]')
        return dataset


class FasttextVectorizer(BaseTransformer):
    def __init__(self, params=None, info=None, request_names=None, new_names=None):
        super().__init__(info, request_names, new_names)
        self.info['op_type'] = 'vectorizer'

        #         print(type(self.new_names))
        #         for i, name in enumerate(self.new_names):
        #             name = name + '_' + 'vec'
        #             self.new_names[i] = name

        if params is None:
            self.params = {
                'path_to_model': '/home/mks/projects/intent_classification_script/data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin',
                'dimension': 300,
                'file_type': 'bin'}

        self.vectorizer = fasttext.load_model(self.params['path_to_model'])

    def _transform(self, dataset):
        print('[ Starting vectorization ... ]')
        request, report = dataset.main_names

        for name, new_name in zip(self.request_names, self.new_names):
            print('[ Vectorization of {} part of dataset ... ]'.format(name))
            data = dataset.data[name][request]
            vec_request = []

            for x in tqdm(data):
                matrix_i = np.zeros((len(x), self.params['dimension']))
                for j, y in enumerate(x):
                    matrix_i[j] = self.vectorizer[y]
                vec_request.append(matrix_i)

            vec_report = list(labels2onehot_one(dataset.data[name][report], dataset.classes))

            dataset.data[new_name] = pd.DataFrame({request: vec_request,
                                                   report: vec_report})

        print('[ Vectorization was ended. ]')
        return dataset
