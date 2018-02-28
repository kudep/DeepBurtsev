import numpy as np
from copy import deepcopy
from time import time
import nltk
import pymorphy2

from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()


def tokenize(sen):
    sent_toks = nltk.sent_tokenize(sen)
    word_toks = [nltk.word_tokenize(el) for el in sent_toks]
    tokens = [val for sublist in word_toks for val in sublist]
    tokens = [el for el in tokens if el != '']
    tokens = [el.lower() for el in tokens]
    tokens = [morph.parse(el)[0].normal_form for el in tokens]
    return tokens


def tokenization(data, morph=True, ngram=True):
    dataset = list()
    ans = list()

    for x, y in tqdm(zip(data['req'], data['cat'])):
        if morph:
            tokens = tokenize(x)
        else:
            sent = nltk.sent_tokenize(x)
            word_toks = [nltk.word_tokenize(el) for el in sent]
            tokens = [val for sublist in word_toks for val in sublist]

        if ngram:
            bigrm = nltk.bigrams(tokens)
            tokens = tokens.extend(bigrm)

        dataset.append(tokens)
        ans.append(y)

    return dataset, ans


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
