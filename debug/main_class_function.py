import pandas as pd
import re
import random
from typing import Generator
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, data, seed=None, split=True, splitting_proportions=None,
                 *args, **kwargs):

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        if splitting_proportions is None:
            self.splitting_proportions = [0.9, 0.1]
        else:
            self.splitting_proportions = splitting_proportions

        if not split:
            self.train = data.get('train', [])
            self.test = data.get('test', [])
            try:
                self.valid = data.get('valid', [])
                self.data = {'train': self.train,
                             'test': self.test,
                             'valid': self.valid,
                             'all': self.train + self.test}
            except KeyError:
                self.data = {'train': self.train,
                             'test': self.test,
                             'all': self.train + self.test}
        else:
            self.train, self.valid, self.test = self.split_data(data)
            self.data = {'train': self.train,
                         'test': self.test,
                         'valid': self.valid,
                         'all': self.train + self.test}

        self.data['classes'] = data['class'].unique()  # np.array

    # TODO rewrite for pandas format
    def batch_generator(self, batch_size: int, data_type: str = 'train') -> Generator:
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        for i in range((data_len - 1) // batch_size + 1):
            yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))

    # TODO rewrite for pandas format
    def iter_all(self, data_type: str = 'train') -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in data:
            yield (x, y)

    def _split_data(self, splitting_proportions, field_to_split, splitted_fields):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(splitted_fields) - 1):
            self.data[splitted_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                          test_size=
                                                                          len(data_to_div) -
                                                                          int(data_size * splitting_proportions[i]))
        self.data[splitted_fields[-1]] = data_to_div
        return True

    # TODO same classes distributions
    def split_data(self, dataset):
        train, valid, test = None, None, None
        return train, valid, test

    def _merge_data(self, fields_to_merge):
        data = self.data.copy()
        new_name = [s + '_' for s in fields_to_merge]
        data[new_name] = []
        for name in fields_to_merge:
            data[new_name] += self.data[name]
        self.data = data
        return True


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
    new_data.rename(columns={'Описание': 'request', 'Категория жалобы': 'class'}, inplace=True)
    new_data = new_data.dropna()  # dell nan
    if not duplicates:
        new_data = new_data.drop_duplicates()  # dell duplicates

    # как отдельную ветвь можно использовать
    if clean:
        delete_bad_symbols = lambda x: " ".join(re.sub('[^а-яa-zё0-9]', ' ', x.lower()).split())
        new_data['request'] = new_data['request'].apply(delete_bad_symbols)

    new_data = new_data.reset_index()
    new_data = new_data.drop('index', axis=1)

    return new_data


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


path = '../data/vkusvill_all_categories.csv'
global_data = read_dataset(path)
train, valid, test = split(global_data, [0.1, 0.1])
print(test)
