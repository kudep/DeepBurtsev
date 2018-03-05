import random
import pandas as pd
from typing import Generator
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, data, seed=None, split=True, splitting_proportions=None, *args, **kwargs):

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        if splitting_proportions is None:
            self.splitting_proportions = [0.1, 0.1]
        else:
            self.splitting_proportions = splitting_proportions

        if not split:
            self.train = data.get('train', [])
            self.test = data.get('test', [])
            try:
                self.valid = data.get('valid', [])
                self.data = {'train': {'base': self.train},
                             'test': {'base': self.test},
                             'valid': {'base': self.valid},
                             'all': {'base': self.train + self.test}}
            except KeyError:
                self.data = {'train': {'base': self.train},
                             'test': {'base': self.test},
                             'all': {'base': self.train + self.test}}
        else:
            self.train, self.valid, self.test = self.split_data(data)
            self.data = {'train': {'base': self.train},
                         'test': {'base': self.test},
                         'valid': {'base': self.valid},
                         'all': {'base': self.train + self.test}}

        self.data['classes'] = data['class'].unique()  # np.array

    def batch_generator(self, batch_size: int, data_type: str = 'train', stage: str = 'base') -> Generator:
        """This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            stage (str): can be either 'base', 'mod1', etc
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type][stage]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        # for i in range((data_len - 1) // batch_size + 1):
        #     yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))
        for i in range((data_len - 1) // batch_size + 1):
            o = order[i * batch_size:(i + 1) * batch_size]
            yield list((list(data['request'][o]), list(data['class'][o])))

    def iter_all(self, data_type: str = 'train', stage: str = 'base') -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
            stage (str): can be either 'base', 'mod1', etc
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type][stage]
        for x, y in zip(data['request'], data['class']):
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

    def split_data(self, dataset):

        dd = dict()
        cd = dict()
        train = list()
        valid = list()
        test = list()

        for x, y in zip(dataset['request'], dataset['class']):
            if y not in dd.keys():
                dd[y] = list()
                cd[y] = 0
                dd[y].append((x, y))
                cd[y] += 1
            else:
                dd[y].append((x, y))
                cd[y] += 1

        if type(self.splitting_proportions) is list:
            assert len(self.splitting_proportions) == 2
            assert type(self.splitting_proportions[0]) is float

            valid_ = dict()
            test_ = dict()

            for x in dd.keys():
                num = int(cd[x] * self.splitting_proportions[0])
                valid_[x] = random.sample(dd[x], num)
                [dd[x].remove(t) for t in valid_[x]]

            for x in dd.keys():
                num = int(cd[x] * self.splitting_proportions[1])
                test_[x] = random.sample(dd[x], num)
                [dd[x].remove(t) for t in test_[x]]
        else:
            raise ValueError('Split proportion must be list of floats, with length = 2')

        train_ = dd

        for x in train_.keys():
            for z_, z in zip([train_, valid_, test_], [train, valid, test]):
                z.extend(z_[x])

        del train_, valid_, test_, dd, cd, dataset  # really need ?

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

    def _merge_data(self, fields_to_merge):
        data = self.data.copy()
        new_name = [s + '_' for s in fields_to_merge]
        data[new_name] = []
        for name in fields_to_merge:
            data[new_name] += self.data[name]
        self.data = data
        return True
