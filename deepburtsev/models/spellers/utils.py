# from deeppavlov.core.models.estimator import Estimator
# from deeppavlov.vocabs.typos import StaticDictionary


import json
import pickle
import logging
import shutil
import requests

from . import paths
from pathlib import Path
from typing import Union
from collections import defaultdict
from lxml import html

_MARK_DONE = '.done'
logger = logging.getLogger(__name__)


def read_json(fpath):
    with open(fpath) as fin:
        return json.load(fin)


def save_json(data, fpath):
    with open(fpath, 'w') as fout:
        return json.dump(data, fout)


def save_pickle(data, fpath):
    with open(fpath, 'wb') as fout:
        pickle.dump(data, fout)


def load_pickle(fpath):
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)


def set_deeppavlov_root(config: dict):
    """
    Make a serialization user dir.
    """
    try:
        deeppavlov_root = Path(config['deeppavlov_root'])
    except KeyError:
        deeppavlov_root = Path(__file__, "..", "..", "..", "download").resolve()

    deeppavlov_root.mkdir(exist_ok=True)

    paths.deeppavlov_root = deeppavlov_root


def get_deeppavlov_root() -> Path:
    return paths.deeppavlov_root


def expand_path(path: Union[str, Path]) -> Path:
    return get_deeppavlov_root() / Path(path).expanduser()


def mark_done(path):
    mark = Path(path) / _MARK_DONE
    mark.touch(exist_ok=True)


def is_done(path):
    mark = Path(path) / _MARK_DONE
    return mark.is_file()


class ConfigError(Exception):
    """
    Any configuration error.
    """

    def __init__(self, message):
        super(ConfigError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class StaticDictionary:
    dict_name = None

    def __init__(self, data_dir=None, *args, **kwargs):
        data_dir = expand_path(data_dir or '')
        if self.dict_name is None:
            self.dict_name = args[0] if args else kwargs.get('dictionary_name', 'dictionary')

        data_dir = data_dir / self.dict_name

        alphabet_path = data_dir / 'alphabet.pkl'
        words_path = data_dir / 'words.pkl'
        words_trie_path = data_dir / 'words_trie.pkl'

        if not is_done(data_dir):
            if data_dir.is_dir():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True)

            words = self._get_source(data_dir, *args, **kwargs)
            words = {self._normalize(word) for word in words}

            alphabet = {c for w in words for c in w}
            alphabet.remove('⟬')
            alphabet.remove('⟭')

            save_pickle(alphabet, alphabet_path)
            save_pickle(words, words_path)

            words_trie = defaultdict(set)
            for word in words:
                for i in range(len(word)):
                    words_trie[word[:i]].add(word[:i+1])
                words_trie[word] = set()
            words_trie = {k: sorted(v) for k, v in words_trie.items()}

            save_pickle(words_trie, words_trie_path)

            mark_done(data_dir)
            print('built')
        else:
            print('Loading a dictionary from {}'.format(data_dir))

        self.alphabet = load_pickle(alphabet_path)
        self.words_set = load_pickle(words_path)
        self.words_trie = load_pickle(words_trie_path)

    @staticmethod
    def _get_source(*args, **kwargs):
        raw_path = args[2] if len(args) > 2 else kwargs.get('raw_dictionary_path', None)
        if not raw_path:
            raise RuntimeError('raw_path for StaticDictionary is not set')
        raw_path = expand_path(raw_path)
        with open(raw_path, newline='') as f:
            data = [line.strip().split('\t')[0] for line in f]
        return data

    @staticmethod
    def _normalize(word):
        return '⟬{}⟭'.format(word.strip().lower().replace('ё', 'е'))


class RussianWordsVocab(StaticDictionary):
    dict_name = 'russian_words_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        print('Downloading russian vocab from https://github.com/danakt/russian-words/')
        url = 'https://github.com/danakt/russian-words/raw/master/russian.txt'
        page = requests.get(url)
        return [word.strip() for word in page.content.decode('cp1251').split('\n')]


class Wiki100KDictionary(StaticDictionary):
    dict_name = 'wikipedia_100K_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        words = []
        print('Downloading english vocab from Wiktionary')
        for i in range(1, 100000, 10000):
            k = 10000 + i - 1
            url = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/{}-{}'.format(i, k)
            page = requests.get(url)
            tree = html.fromstring(page.content)
            words += tree.xpath('//div[@class="mw-parser-output"]/p/a/text()')
        return words
