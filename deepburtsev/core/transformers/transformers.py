import json
import pandas as pd
import nltk
import pymorphy2

import numpy as np

from tqdm import tqdm
from deepburtsev.core.utils import logging
# from DeepPavlov.deeppavlov.core.commands.infer import build_model_from_config

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

    def fit_transform(self, dataset):
        self._validate_names(dataset)
        return self._transform(dataset)

    def get_params(self):
        return self.config

    def set_params(self, params):
        # self.params = params
        self.__init__(params)
        return self


# class Speller(BaseTransformer):
#     def __init__(self, config=None):
#         if config is None:
#             self.config = {'op_type': 'transformer',
#                            'name': 'Speller',
#                            'request_names': ['base'],
#                            'new_names': ['base'],
#                            'path': './DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json'}
#         else:
#             need_names = ['path']
#             for name in need_names:
#                 if name not in config.keys():
#                     raise ValueError('Input config must contain {}.'.format(name))
#
#             self.config = config
#
#         super().__init__(self.config)
#
#         self.conf_path = self.config['path']
#         with open(self.conf_path) as config_file:
#             self.speller_config = json.load(config_file)
#
#         self.speller = build_model_from_config(self.speller_config)
#
#     def _transform(self, dataset):
#         print('[ Speller start working ... ]')
#
#         request, report = dataset.main_names
#         for name, new_name in zip(self.request_names, self.new_names):
#             data = dataset.data[name]
#             refactor = list()
#
#             for x in tqdm(data[request]):
#                 refactor.append(self.speller([x])[0])
#
#             dataset.data[new_name] = pd.DataFrame({request: refactor,
#                                                    report: data[report]})
#
#         print('[ Speller done. ]')
#         return dataset


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

        self.metrices = ['accuracy', 'f1_macro', 'f1_weighted', 'confusion_matrix']
        self.category_description = None
        self.root = None

        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'Resulter',
                           'request_names': ['predicted_test'],
                           'new_names': ['test'],
                           'metrics': ['accuracy', 'f1_macro', 'f1_weighted', 'confusion_matrix']}
        else:
            self.config = config

        for metr in self.config['metrics']:
            if metr not in self.metrices:
                raise ValueError('{} metrics is not implemented yet.'.format(metr))

        super().__init__(self.config)

    def _transform(self, dataset):
        request, report = dataset.main_names
        dataset_name = dataset.dataset_name
        language = dataset.language
        res_type = dataset.restype
        self.root = dataset.root

        self.category_description = dataset.classes_description
        if self.category_description is None:
            self.category_description = dataset.classes

        print(res_type)

        if res_type == 'neural':
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

            results = self.get_results(preds, real_data)
            dataset.data['results'] = results

            conf = dataset.pipeline_config
            date = dataset.date

            names_ = list(conf.keys())
            for u in conf[names_[-2]].keys():
                if isinstance(conf[names_[-2]][u], np.int64):
                    conf[names_[-2]][u] = int(conf[names_[-2]][u])

            logging(results, conf, date, language=language, dataset_name=dataset_name, root=self.root)

        elif res_type == 'linear':
            pred_name = self.config['request_names'][0]
            real_name = self.config['new_names'][0]
            pred_data = np.array(dataset.data[pred_name])

            real_data = np.array(dataset.data[real_name][report])
            results = self.get_results(pred_data, real_data)
            dataset.data['results'] = results

            conf = dataset.pipeline_config
            date = dataset.date

            names_ = list(conf.keys())
            for u in conf[names_[-2]].keys():
                if isinstance(conf[names_[-2]][u], np.int64):
                    conf[names_[-2]][u] = int(conf[names_[-2]][u])

            logging(results, conf, date, language=language, dataset_name=dataset_name, root=self.root)

        else:
            raise ValueError('Incorrect type: {}; need "neural" or "linear".'.format(res_type))

        return dataset

    def get_results(self, y_pred, y_true):

        results = dict()
        results['classes'] = {}
        for metr in self.config['metrics']:
            results[metr] = self.return_metric(metr, y_true, y_pred)

        for i in range(len(self.category_description)):
            y_bin_pred = np.zeros(y_pred.shape)
            y_bin_pred[y_pred == i] = 1
            y_bin_answ = np.zeros(y_pred.shape)
            y_bin_answ[y_true == i] = 1

            precision_tmp = precision_score(y_bin_answ, y_bin_pred)
            recall_tmp = recall_score(y_bin_answ, y_bin_pred)
            if recall_tmp == 0 and precision_tmp == 0:
                f1_tmp = 0.
            else:
                f1_tmp = 2 * recall_tmp * precision_tmp / (precision_tmp + recall_tmp)

            results['classes'][str(self.category_description[i])] = \
                {'number_test_objects': y_bin_answ[y_true == i].shape[0],
                 'precision': precision_tmp,
                 'recall': recall_tmp,
                 'f1': f1_tmp}

        # string_to_format = '{:7} number_test_objects: {:4}   precision: {:5.3}   recall: {:5.3}  f1: {:5.3}'
        # results['classes'].append(string_to_format.format(self.category_description[i],
        #                                                   y_bin_answ[y_true == i].shape[0],
        #                                                   precision_tmp,
        #                                                   recall_tmp,
        #                                                   f1_tmp))

        return results

    @staticmethod
    def return_metric(metr, y_true, y_pred):
        if metr == 'accuracy':
            res = accuracy_score(y_true, y_pred)
        elif metr == 'f1_macro':
            res = f1_score(y_true, y_pred, average='macro')
        elif metr == 'f1_weighted':
            res = f1_score(y_true, y_pred, average='weighted')
        elif metr == 'confusion_matrix':
            res = confusion_matrix(y_true, y_pred).tolist()
        else:
            raise ValueError

        return res
