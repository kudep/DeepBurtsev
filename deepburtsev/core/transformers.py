import nltk
import pymorphy2
import tensorflow_hub as hub
import tensorflow as tf
import fastText
import numpy as np
import random

# only for python 3.3+
from inspect import signature

from collections import defaultdict
from tqdm import tqdm

from deepburtsev.models.spellers.utils import RussianWordsVocab
from deepburtsev.models.spellers.error_model import ErrorModel

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class BaseClass(object):
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


class BaseTransformer(BaseClass):
    def __init__(self, request_names=None, new_names=None, op_type=None, op_name=None):
        # named spaces
        self.request_names = []

        if self._check_names(request_names):
            self.worked_names = request_names
        else:
            self.worked_names = [request_names]

        if self._check_names(new_names):
            self.new_names = new_names
        else:
            self.new_names = [new_names]

        if len(self.new_names) != len(self.worked_names):
            raise ValueError('The number of requested and new names must match.')

        self.op_type = op_type
        self.op_name = op_name

    @staticmethod
    def _check_names(names):
        if isinstance(names, list):
            for i, x in enumerate(names):
                if not isinstance(x, str):
                    raise ValueError('A list of names must contain strings,'
                                     ' but {0} was found in position {1}.'.format(type(x), i))
        elif isinstance(names, str):
            return False
        else:
            raise ValueError('Input parameters must be string or a list of strings.')
        return True

    def _fill_names(self, x_name, y_name):
        if x_name is not None:
            self.worked_names = x_name
        if y_name is not None:
            self.new_names = y_name

        if len(self.new_names) != len(self.worked_names):
            raise ValueError('The number of requested and new names must match.')

        return self

    def _validate_names(self, dictionary):
        if self.worked_names is not None:
            if not isinstance(self.worked_names, list):
                raise ValueError('Request_names must be a list, but {} was found.'.format(type(self.worked_names)))

            for name in self.worked_names:
                if name not in dictionary.keys():
                    raise KeyError('Key {} not found in input dictionary.'.format(name))
                else:
                    self.request_names.append(name)
        else:
            self.worked_names = ['base', 'train', 'valid', 'test']
            for name in self.worked_names:
                if name in dictionary.keys():
                    self.request_names.append(name)
            if len(self.request_names) == 0:
                raise KeyError('Keys from {} not found in input dictionary.'.format(self.worked_names))

        if self.new_names is None:
            self.new_names = self.request_names

        return self

    def _transform(self, dictionary):
        raise AttributeError("Method 'transform' is not defined. Determine the method.")

    def transform(self, dictionary):
        self._validate_names(dictionary)
        return self._transform(dictionary)

    def fit_transform(self, dictionary):
        self._validate_names(dictionary)
        return self.transform(dictionary)


class Tokenizer(BaseTransformer):
    def __init__(self, request_names='base', new_names='base', op_type='transformer', op_name='Tokenizer'):
        super().__init__(request_names, new_names, op_type, op_name)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Starting tokenization ... ]')
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            data = dictionary[name]['x']
            tok_data = list()

            for x in tqdm(data):
                sent_toks = nltk.sent_tokenize(x)
                word_toks = [nltk.word_tokenize(el) for el in sent_toks]
                tokens = [val for sublist in word_toks for val in sublist]
                tok_data.append(tokens)

            dictionary[new_name] = {'x': tok_data, 'y': dictionary[name]['y']}

        print('[ Tokenization was done. ]')
        return dictionary


class Lemmatizer(BaseTransformer):
    def __init__(self, request_names='base', new_names='base', op_type='transformer', op_name='Lemmatizer'):
        super().__init__(request_names, new_names, op_type, op_name)
        self.morph = pymorphy2.MorphAnalyzer()

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Starting lemmatization ... ]')
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            data = dictionary[name]['x']
            morph_data = list()

            for x in tqdm(data):
                mp_data = [self.morph.parse(el)[0].normal_form for el in x]
                morph_data.append(mp_data)

                dictionary[new_name] = {'x': morph_data, 'y': dictionary[name]['y']}
        print('[ Ended lemmatization. ]')
        return dictionary


class TextConcat(BaseTransformer):
    def __init__(self, request_names='base', new_names='base', op_type='transformer', op_name='Concatenator'):
        super().__init__(request_names, new_names, op_type, op_name)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Starting text merging ... ]')
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            data = dictionary[name]['x']
            text_request = []

            for x in tqdm(data):
                text_request.append(' '.join([z for z in x]))

                dictionary[new_name] = {'x': text_request, 'y': dictionary[name]['y']}

        print('[ Text concatenation was ended. ]')
        return dictionary


class ResultsCollector(BaseTransformer):
    def __init__(self, request_names='pred_test', new_names='results', y_pred='y_pred', y_true='y_true',
                 op_type='transformer', op_name='ResultsCollector', metrics=None):
        super().__init__(request_names, new_names, op_type, op_name)

        self.y_pred = y_pred
        self.y_true = y_true
        self.category_description = None

        self.available_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        if metrics is None:
            self.metrics = self.available_metrics
        else:
            for metric in metrics:
                if metric not in self.available_metrics:
                    raise ValueError('Sorry {0} metrics is not implemented yet.'.format(metric))
            self.metrics = metrics

    def _check_data_format(self, data):
        if isinstance(data, np.ndarray):
            if len(np.shape(data)) != 1:
                raise ValueError('y_pred and y_true must be numpy.ndarray with rang=1 or list.')
            else:
                return self
        elif isinstance(data, list):
            return self
        else:
            raise ValueError('y_pred and y_true must be numpy.ndarray with rang=1 or list')

    def _transform(self, dictionary, request_names=None, new_names=None):
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            pred_data = dictionary[name][self.y_pred]
            real_data = dictionary[name][self.y_true]

            self._check_data_format(pred_data)
            self._check_data_format(real_data)

            # get amount of classes as list
            # self.category_description = [i for i in range(len(dictionary[self.request_names[1]][0]))]
            self.category_description = list(set(dictionary['train']['y']))

            # protector from excess batch elements
            pred_data = pred_data[:len(real_data)]

            results = self.get_results(pred_data, real_data)
            dictionary[new_name] = results

        return dictionary

    def get_results(self, y_pred, y_true):
        y_true = np.array(y_true)
        results = dict()
        results['classes'] = {}
        for metr in self.metrics:
            results[metr] = self.return_metric(metr, y_true, y_pred)

        for i in range(len(self.category_description)):
            y_bin_pred = np.zeros(y_pred.shape)
            y_bin_pred[y_pred == i] = 1

            y_bin_answ = np.zeros(y_true.shape)
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


class FasttextVectorizer(BaseTransformer):
    def __init__(self, request_names=None, new_names=None, classes_name='classes', op_type='vectorizer',
                 op_name='fasttext', dimension=300, file_type='bin',
                 model_path='./data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin'):
        super().__init__(request_names, new_names, op_type, op_name)
        self.file_type = file_type
        self.classes_name = classes_name
        self.dimension = dimension
        self.model_path = model_path
        self.vectorizer = fastText.load_model(self.model_path)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Starting vectorization ... ]')
        self._fill_names(request_names, new_names)

        # if dictionary.get(self.classes_name) is None:
        #     raise ValueError("The inbound dictionary should contain list of classes for one-hot"
        #                      " vectorization of y_true values.")

        for name, new_name in zip(self.request_names, self.new_names):
            print('[ Vectorization of {} part of dataset ... ]'.format(name))
            data = dictionary[name]['x']
            vec_request = []

            for x in tqdm(data):
                matrix_i = np.zeros((len(x), self.dimension))
                for j, y in enumerate(x):
                    matrix_i[j] = self.vectorizer.get_word_vector(y)
                vec_request.append(matrix_i)

            # vec_report = list(labels2onehot_one(dictionary[name]['y'], dictionary[self.classes_name]))

            dictionary[new_name] = {'x': vec_request, 'y': dictionary[name]['y']}

        print('[ Vectorization was ended. ]')
        return dictionary


class SentEmbedder(BaseTransformer):
    def __init__(self, request_names=None, new_names=None, op_type='vectorizer', op_name='GoogleSentEmbedder',
                 model_path="https://tfhub.dev/google/universal-sentence-encoder/1"):
        super().__init__(request_names, new_names, op_type, op_name)
        self.model_path = model_path
        self.model = hub.Module(self.model_path)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ SentEmbedder start working ... ]')
        self._fill_names(request_names, new_names)

        with tf.Session() as session:
            for name, new_name in zip(self.request_names, self.new_names):
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                message_embeddings = session.run(self.model(list(dictionary[name]['x'])))

                dictionary[new_name]['x'] = message_embeddings
                dictionary[new_name]['y'] = list(dictionary[name]['y'])

        print('[ SentEmbedder and work. ]')
        return dictionary


# TODO fix
class Splitter(BaseTransformer):
    def __init__(self, split_names='base', new_names=['train', 'test'], op_type='transformer', op_name='Splitter',
                 splitting_proportions=[0.9, 0.1], delete_parent=True, classes_name='classes'):
        super().__init__(split_names, new_names, op_type, op_name)
        self.splitting_proportions = splitting_proportions
        self.delete_parent = delete_parent
        self.classes_name = classes_name

    def _transform(self, dictionary, splitting_proportions=None, delete_parent=True):

        # self._fill_names(request_names, new_names)

        if dictionary.get(self.classes_name) is None:
            raise ValueError("The inbound dictionary should contain list of classes for one-hot"
                             " vectorization of y_true values.")

        dd = dict()
        cd = dictionary[self.classes_name]
        train = list()
        valid = list()
        test = list()

        if splitting_proportions is None:
            splitting_proportions = [0.1, 0.1]

        for name, new_names in zip(self.request_names, self.new_names):
            dataset = dictionary[name]

            for x, y in zip(dataset['x'], dataset['y']):
                if y not in dd.keys():
                    dd[y] = list()
                    dd[y].append((x, y))
                else:
                    dd[y].append((x, y))

            if type(splitting_proportions) is list:
                assert len(splitting_proportions) == 2
                assert type(splitting_proportions[0]) is float

                valid_ = dict()
                test_ = dict()

                for x in dd.keys():
                    num = int(cd[x] * splitting_proportions[0])
                    valid_[x] = random.sample(dd[x], num)
                    [dd[x].remove(t) for t in valid_[x]]

                for x in dd.keys():
                    num = int(cd[x] * splitting_proportions[1])
                    test_[x] = random.sample(dd[x], num)
                    [dd[x].remove(t) for t in test_[x]]
            else:
                raise ValueError('Split proportion must be list of floats, with length = 2')

            train_ = dd

            for x in train_.keys():
                for z_, z in zip([train_, valid_, test_], [train, valid, test]):
                    z.extend(z_[x])

            del train_, valid_, test_, dd, cd, dataset

            for z in [train, valid, test]:
                z = random.shuffle(z)

            utrain, uvalid, utest, ctrain, cvalid, ctest = list(), list(), list(), list(), list(), list()
            for z, n, c in zip([train, valid, test], [utrain, uvalid, utest], [ctrain, cvalid, ctest]):
                for x in z:
                    n.append(x[0])
                    c.append(x[1])

            dictionary[name]['train'] = {'x': utrain, 'y': ctrain}
            dictionary[name]['valid'] = {'x': uvalid, 'y': cvalid}
            dictionary[name]['test'] = {'x': utest, 'y': ctest}

            if delete_parent:
                a = dictionary[name].pop('base', [])
                del a

        return self


class Speller(BaseTransformer):
    def __init__(self, request_names='base', new_names='base', op_type='transformer', op_name='Speller',
                 dict_path='./downloads/error_model/',
                 model_save_path='./downloads/error_model/',
                 model_load_path='./downloads/error_model/'):

        super().__init__(request_names, new_names, op_type, op_name)

        self.save_path = model_save_path
        self.load_path = model_load_path
        self.dict_path = dict_path

        self.russian_word_dictionary = RussianWordsVocab(data_dir=self.dict_path)
        self.speller = ErrorModel(self.russian_word_dictionary, lm_file='',
                                  save_path=self.save_path, load_path=self.load_path)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Speller start working ... ]')
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            data = dictionary[name]
            refactor = list()

            for x in tqdm(data['x']):
                refactor.append(self.speller([x])[0])

            dictionary[new_name] = {'x': refactor, 'y': data['y']}

        print('[ Speller done. ]')
        return dictionary


class Lower(BaseTransformer):
    def __init__(self, request_names='base', new_names='base', op_type='transformer', op_name='Lowercase'):
        super().__init__(request_names, new_names, op_type, op_name)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Lowercase ]')
        self._fill_names(request_names, new_names)

        for name, new_name in zip(self.request_names, self.new_names):
            lower = []
            for x in dictionary[name]['x']:
                lower.append(x.lower())
            dictionary[new_name] = {'x': lower, 'y': dictionary[name]['y']}

        return dictionary
