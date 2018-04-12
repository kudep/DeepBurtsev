import json
import pandas as pd
import nltk
import pymorphy2
import fasttext
import numpy as np

from tqdm import tqdm
from DeepBurtsev.core.utils import labels2onehot_one, get_result, logging
from DeepPavlov.deeppavlov.core.commands.infer import build_model_from_config


class BaseTransformer(object):
    def __init__(self, config=None):
        # info resist
        if not isinstance(config, dict):
            raise ValueError('Input config must be dict or None, but {} was found.'.format(type(config)))

        # keys = ['op_type', 'name', 'request_names', 'new_names', 'input_x_type', 'input_y_type', 'output_x_type',
        #         'output_y_type']

        keys = ['op_type', 'name', 'request_names', 'new_names']

        self.info = dict()
        for x in keys:
            if x not in config.keys():
                raise ValueError('Input config must contain {} key.'.format(x))
            elif x == 'request_names' or x == 'new_names':
                if not isinstance(config[x], list):
                    raise ValueError('request_names and new_names in config must be list,'
                                     ' but {} was found.'.format(type(x)))
            self.info[x] = config[x]

        self.config = config

        # named spaces
        self.new_names = config['new_names']
        self.worked_names = config['request_names']
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

    def _transform(self, dataset):
        return None

    def transform(self, dataset):
        self._validate_names(dataset)
        return self._transform(dataset)

    def get_params(self):
        return self.config

    def set_params(self, params):
        # self.params = params
        self.__init__(params)
        return self


class Speller(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'Speller',
                           'request_names': ['base'],
                           'new_names': ['base'],
                           'path': './DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json'}
        else:
            need_names = ['path']
            for name in need_names:
                if name not in config.keys():
                    raise ValueError('Input config must contain {}.'.format(name))

            self.config = config

        super().__init__(self.config)

        self.conf_path = self.config['path']
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
    def __init__(self, config=None):
        if config is None:
            # self.config = {'op_type': 'transformer',
            #                'name': 'Tokenizer',
            #                'request_names': ['base'],
            #                'new_names': ['base'],
            #                'input_x_type': pd.core.series.Series,
            #                'input_y_type': pd.core.series.Series,
            #                'output_x_type': pd.core.series.Series,
            #                'output_y_type': pd.core.series.Series}

            self.config = {'op_type': 'transformer',
                           'name': 'Tokenizer',
                           'request_names': ['base'],
                           'new_names': ['base']}

        else:
            self.config = config

        super().__init__(self.config)

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
    def __init__(self, config=None):
        self.morph = pymorphy2.MorphAnalyzer()

        if config is None:
            # self.config = {'op_type': 'transformer',
            #                'name': 'Lemmatizer',
            #                'request_names': ['base'],
            #                'new_names': ['base'],
            #                'input_x_type': pd.core.series.Series,
            #                'input_y_type': pd.core.series.Series,
            #                'output_x_type': pd.core.series.Series,
            #                'output_y_type': pd.core.series.Series}

            self.config = {'op_type': 'transformer',
                           'name': 'Lemmatizer',
                           'request_names': ['base'],
                           'new_names': ['base']}

        else:
            self.config = config

        super().__init__(self.config)

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
    def __init__(self, config=None):

        if config is None:
            # self.config = {'op_type': 'vectorizer',
            #                'name': 'fasttext',
            #                'request_names': ['train', 'valid', 'test'],
            #                'new_names': ['train_vec', 'valid_vec', 'test_vec'],
            #                'input_x_type': pd.core.series.Series,
            #                'input_y_type': pd.core.series.Series,
            #                'output_x_type': pd.core.series.Series,
            #                'output_y_type': pd.core.series.Series,
            #                'path_to_model': './data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin',
            #                'dimension': 300,
            #                'file_type': 'bin'}

            self.config = {'op_type': 'vectorizer',
                           'name': 'fasttext',
                           'request_names': ['train', 'valid', 'test'],
                           'new_names': ['train_vec', 'valid_vec', 'test_vec'],
                           'path_to_model': './data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin',
                           'dimension': 300,
                           'file_type': 'bin'}

        else:
            need_names = ['path_to_model', 'dimension', 'file_type']
            for name in need_names:
                if name not in config.keys():
                    raise ValueError('Input config must contain {}.'.format(name))

            self.config = config

        super().__init__(self.config)

        self.vectorizer = fasttext.load_model(self.config['path_to_model'])

    def _transform(self, dataset):
        print('[ Starting vectorization ... ]')
        request, report = dataset.main_names

        for name, new_name in zip(self.request_names, self.new_names):
            print('[ Vectorization of {} part of dataset ... ]'.format(name))
            data = dataset.data[name][request]
            vec_request = []

            for x in tqdm(data):
                matrix_i = np.zeros((len(x), self.config['dimension']))
                for j, y in enumerate(x):
                    matrix_i[j] = self.vectorizer[y]
                vec_request.append(matrix_i)

            vec_report = list(labels2onehot_one(dataset.data[name][report], dataset.classes))

            dataset.data[new_name] = pd.DataFrame({request: vec_request,
                                                   report: vec_report})

        print('[ Vectorization was ended. ]')
        return dataset


class TextConcat(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'text_concatenator',
                           'request_names': ['base'],
                           'new_names': ['base']}
        else:
            need_names = []
            for name in need_names:
                if name not in config.keys():
                    raise ValueError('Input config must contain {}.'.format(name))

            self.config = config

        super().__init__(self.config)

    def _transform(self, dataset):
        print('[ Starting text merging ... ]')
        request, report = dataset.main_names

        for name, new_name in zip(self.request_names, self.new_names):
            data = dataset.data[name][request]
            text_request = []

            for x in tqdm(data):
                text_request.append(' '.join([z for z in x]))

            dataset.data[new_name] = pd.DataFrame({request: text_request,
                                                   report: dataset.data[name][report]})

        print('[ Text concatenation was ended. ]')
        return dataset


class GetResult(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'Resulter',
                           'request_names': ['predicted_test'],
                           'new_names': ['test']}
        else:
            self.config = config

        super().__init__(self.config)

    def _transform(self, dataset):
        request, report = dataset.main_names

        pred_name = self.config['request_names'][0]
        real_name = self.config['new_names'][0]
        pred_data = dataset.data[pred_name]
        real_data = np.array(dataset.data[real_name][report])

        preds = pred_data[0]
        for x in pred_data[1:]:
            preds = np.concatenate((preds, x), axis=0)

        preds = np.argmax(preds, axis=1)
        for i, x in enumerate(preds):
            preds[i] = x + 1

        preds = preds[:len(real_data)]

        results = get_result(preds, real_data)
        dataset.data['results'] = results

        conf = dataset.pipeline_config
        date = dataset.date

        # TODO fix dependencies
        logging(results, conf, date, language='russian', dataset_name='vkusvill')

        return dataset


class GetResultLinear(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'Resulter',
                           'request_names': ['predicted_test'],
                           'new_names': ['test']}
        else:
            self.config = config

        super().__init__(self.config)

    def _transform(self, dataset):
        request, report = dataset.main_names

        pred_name = self.config['request_names'][0]
        real_name = self.config['new_names'][0]
        pred_data = np.array(dataset.data[pred_name])

        real_data = np.array(dataset.data[real_name][report])
        results = get_result(pred_data, real_data)
        dataset.data['results'] = results
        return dataset


class GetResultLinear_W(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'Resulter',
                           'request_names': ['predicted_test'],
                           'new_names': ['test']}
        else:
            self.config = config

        super().__init__(self.config)

    def _transform(self, dataset):
        request, report = dataset.main_names

        pred_name = self.config['request_names'][0]
        real_name = self.config['new_names'][0]
        pred_data = np.array(dataset.data[pred_name])

        real_data = np.array(dataset.data[real_name][report])
        results = get_result(pred_data, real_data)
        dataset.data['results'] = results

        conf = dataset.pipeline_config
        date = dataset.date

        # TODO fix dependencies
        logging(results, conf, date, language='russian', dataset_name='vkusvill')

        return dataset
