import json
import pandas as pd
import nltk
import pymorphy2
import numpy as np

# only for python 3.3+
from inspect import signature

from tqdm import tqdm
from os.path import join
from collections import defaultdict

from deepburtsev.core.utils import logging
from DeepPavlov.deeppavlov.core.commands.infer import build_model_from_config

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class BaseTransformer(object):
    def __init__(self, request_names=None, new_names=None, op_type='transformer', op_name='base_transformer'):
        # named spaces
        self.op_type = op_type
        self.op_name = op_name
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
        raise AttributeError("Method 'transform' is not defined. Determine the method.")

    def fit_transform(self, dataset):
        self._validate_names(dataset)
        return self.transform(dataset)

    # def get_params(self):
    #     return self.config

    # def set_params(self, **params):
    #     self.__init__(**params)
    #     return self

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


class Speller(BaseTransformer):
    def __init__(self, request_names=None, new_names=None, op_type='transformer', op_name='Speller',
                 model_path='DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json',
                 root='/home/mks/projects/DeepBurtsev/'):

        super().__init__(request_names, new_names, op_type, op_name)

        self.model_path = join(root, model_path)
        with open(self.model_path) as config_file:
            self.speller_config = json.load(config_file)
            config_file.close()

        self.speller = build_model_from_config(self.speller_config)

    def transform(self, dataset, request_names=None, new_names=None):
        print('[ Speller start working ... ]')

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

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
    def transform(self, dataset, request_names=None, new_names=None):
        print('[ Starting tokenization ... ]')

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

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
    def __init__(self, request_names=None, new_names=None, op_type='transformer', op_name='Lemmatizer'):
        super().__init__(request_names, new_names, op_type, op_name)
        self.morph = pymorphy2.MorphAnalyzer()

    def transform(self, dataset, request_names=None, new_names=None):
        print('[ Starting lemmatization ... ]')

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

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
    def transform(self, dataset, request_names=None, new_names=None):
        print('[ Starting text merging ... ]')
        request, report = dataset.main_names

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

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
    def __init__(self, request_names=None, new_names=None, op_type='transformer', op_name='Results summarizator',
                 metrics=None):
        super().__init__(request_names, new_names, op_type, op_name)

        self.available_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'confusion_matrix']
        if metrics is None:
            self.metrics = self.available_metrics
        else:
            for metric in self.available_metrics:
                if metric not in self.metrics:
                    raise ValueError('{} metrics is not implemented yet.'.format(metric))
            self.metrics = self.available_metrics

        self.category_description = None
        self.root = None

    def transform(self, dataset, request_names=None, new_names=None):
        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

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
            pred_name = self.request_names[0]
            real_name = self.new_names[0]
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
            pred_name = self.request_names[0]
            real_name = self.new_names[0]
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
        for metr in self.metrics:
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
