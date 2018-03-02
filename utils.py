import numpy as np
import nltk
import pymorphy2
import random
import pandas as pd

from copy import deepcopy
from time import time
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()


def transform(data, lower=True, lemma=True, ngramm=False):
    Tokens = list()
    for x in data['request']:
        sent_toks = nltk.sent_tokenize(x)
        word_toks = [nltk.word_tokenize(el) for el in sent_toks]
        tokens = [val for sublist in word_toks for val in sublist]
        tokens = [el for el in tokens if el != '']
        if lower:
            tokens = [el.lower() for el in tokens]
        if lemma:
            tokens = [morph.parse(el)[0].normal_form for el in tokens]
        if ngramm:
            bigram = list(nltk.bigrams(tokens))
            bigram = ['_'.join(x) for x in bigram]
            Tokens.append(' '.join(bigram))
        else:
            Tokens.append(' '.join(tokens))

    df = pd.DataFrame({'request': Tokens,
                       'class': data['class']})
    return df


class HyperPar:
    def __init__(self, **kwargs):
        np.random.seed(int(time()))
        self.params = kwargs

    def sample_params(self):
        params = deepcopy(self.params)
        params_sample = dict()
        for param, param_val in params.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = bool(np.random.choice([True, False]))
                elif 'range' in param_val:
                    # Generate number of smaples
                    if 'n_samples' in param_val:
                        if param_val['n_samples'] > 1 and param_val.get('increasing', False):

                            sample_1 = self._sample_from_ranges(param_val)
                            sample_2 = self._sample_from_ranges(param_val)
                            start_stop = sorted([sample_1, sample_2])
                            sample = [s for s in np.linspace(start_stop[0], start_stop[1], param_val['n_samples'])]
                            if param_val.get('discrete', False):
                                sample = [int(s) for s in sample]
                        else:
                            sample = [self._sample_from_ranges(param_val) for _ in range(param_val['n_samples'])]
                    else:
                        sample = self._sample_from_ranges(param_val)
                params_sample[param] = sample
            else:
                params_sample[param] = param_val
        return params_sample

    def _sample_from_ranges(self, opts):
        from_ = opts['range'][0]
        to_ = opts['range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_, to_):
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)


def split(data, prop):
    dd = dict()
    cd = dict()
    train = list()
    valid = list()
    test = list()

    for x, y in zip(data['request'], data['class']):
        if y not in dd.keys():
            dd[y] = list()
            cd[y] = 0
            dd[y].append((x, y))
            cd[y] += 1
        else:
            dd[y].append((x, y))
            cd[y] += 1

    if type(prop) is list:
        assert len(prop) == 2
        assert type(prop[0]) is float

        valid_ = dict()
        test_ = dict()

        for x in dd.keys():
            num = int(cd[x] * prop[0])
            valid_[x] = random.sample(dd[x], num)
            [dd[x].remove(t) for t in valid_[x]]

        for x in dd.keys():
            num = int(cd[x] * prop[1])
            test_[x] = random.sample(dd[x], num)
            [dd[x].remove(t) for t in test_[x]]
    else:
        raise ValueError('Split proportion must be list of floats, with length = 2')

    train_ = dd

    for x in train_.keys():
        for z_, z in zip([train_, valid_, test_], [train, valid, test]):
            z.extend(z_[x])

    del train_, valid_, test_, dd, cd

    for z in [train, valid, test]:
        z = random.shuffle(z)

    utrain, uvalid, utest, ctrain, cvalid, ctest = list(), list(), list(), list(), list(), list()
    for z, n, c in zip([train, valid, test], [utrain, uvalid, utest], [ctrain, cvalid, ctest]):
        for x in z:
            n.append(x[0])
            c.append(x[1])

    train = pd.DataFrame({'request': utrain,
                          'class': ctrain})
    valid = pd.DataFrame({'request': uvalid,
                          'class': cvalid})
    test = pd.DataFrame({'request': utest,
                         'class': ctest})

    return train, valid, test
