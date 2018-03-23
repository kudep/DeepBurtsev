import pandas as pd
import pymorphy2
import fasttext
import json
import os
import re

from dataset import Dataset
from utils import transform, logging, results_summarization
from dataset import Dataset
from transformers import Speller, Tokenizer, Lemmatizer, FasttextVectorizer
morph = pymorphy2.MorphAnalyzer()
 

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
    new_data.rename(columns={'Описание': 'request', 'Категория жалобы': 'report'}, inplace=True)
    new_data = new_data.dropna()  # dell nan
    if not duplicates:
        new_data = new_data.drop_duplicates()  # dell duplicates

    # как отдельную ветвь можно использовать
    if clean:
        delete_bad_symbols = lambda x: " ".join(re.sub('[^а-яa-zё0-9]', ' ', x.lower()).split())
        new_data['request'] = new_data['request'].apply(delete_bad_symbols)

    new_data = new_data.reset_index()

    return new_data


class Pipeline(object):
    def __init__(self, pipe, config=None):
        self.pipe = []
        self.config = None

        if config is not None:
            assert len(pipe) == len(config), ('List of operations and configurations has different length:'
                                              'pipe={0}, config={1}'.format(len(pipe), len(config)))
            for i, x, y in enumerate(zip(pipe, config)):
                assert len(y) == 2, ('Config of operation must be tuple of len two:'
                                     'len of {0} element in configs={1};'.format(i, len(y[i])))
                assert x[0] == y[0], ('Names of operations and configurations must be the same:'
                                      'pipe_{0}={1}, config_{0}={2};'.format(i, x[0], y[0]))

            for i, x, y in enumerate(zip(pipe, config)):
                self.pipe.append(x[1](y[1]))

            self.config = config
        else:
            for op in pipe:
                self.pipe.append(op[1]())

        self._validate_steps()

    # def _validate_names(self, names):
    #     if len(set(names)) != len(names):
    #         raise ValueError('Names provided are not unique: '
    #                          '{0!r}'.format(list(names)))
    #     invalid_names = set(names).intersection(self.get_params(deep=False))
    #     if invalid_names:
    #         raise ValueError('Estimator names conflict with constructor '
    #                          'arguments: {0!r}'.format(sorted(invalid_names)))
    #     invalid_names = [name for name in names if '__' in name]
    #     if invalid_names:
    #         raise ValueError('Estimator names must not contain __: got '
    #                          '{0!r}'.format(invalid_names))

    def _validate_steps(self):
        # validate names
        # self._validate_names(names)

        # validate models and transformers
        transformers = []
        models = []
        for op in self.pipe:
            if op.info['type'] == 'transformer':
                transformers.append(op)
            elif op.info['type'] == 'model':
                models.append(op)
            else:
                raise TypeError("All operations should be transformers or models and implement fit and transform,"
                                "but {0} operation has type: {1}".format(op, op.info['type']))

        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
            hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        for m in models:
            if m is not None and not (hasattr(m, "fit") or hasattr(m, "predict")):
                raise TypeError("Models should implement fit or predict. "
                                "'%s' (type %s) doesn't"
                                % (m, type(m)))

    def fit(self, dataset, **fit_params):
        for op in self.pipe:
            if op is not None:
                if op.info['type'] == 'transformer':
                    dataset = op.transform(dataset, name='base')
                elif op.info['type'] == 'model':
                    op.init(dataset)
                    op.fit()
            else:
                pass

        return self

    def predict(self, dataset):
        prediction = None

        for op in self.pipe:
            if op is not None:
                if op.info['type'] == 'transformer':
                    dataset = op.transform(dataset, name='base')
                elif op.info['type'] == 'model':
                    op.init(dataset)
                    prediction = op.predict(dataset)
            else:
                pass

        return prediction


# testing
path = '/home/mks/projects/intent_classification_script/data/russian/data/vkusvill_all_categories.csv'
global_data = read_dataset(path)
dataset = Dataset(global_data, seed=42)
p = [('Speller', Speller), ('Tokenizer', Tokenizer), ('Lemmatizer', Lemmatizer), ('Vectorizer', FasttextVectorizer)]


pipe = Pipeline(pipe=p)
pipe.fit(dataset)












































