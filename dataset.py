import random
import pandas as pd
from typing import Generator
from sklearn.model_selection import train_test_split


class Dataset(object):

    def __init__(self, data, seed=None, classes_description=None, *args, **kwargs):

        self.main_names = ['request', 'report']

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.classes_description = classes_description
        self.data = dict()

        if data.get('train') is not None:
            self.data['train'] = data.get('train')
        elif data.get('test') is not None:
            self.data['test'] = data.get('test')
        elif data.get('valid') is not None:
            self.data['valid'] = data.get('valid')
        else:
            self.data['base'] = data

        self.classes = self.get_classes()
        self.classes_distribution = self.get_distribution()

    def simple_split(self, splitting_proportions, field_to_split, splitted_fields, delete_parent=True):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(splitted_fields) - 1):
            self.data[splitted_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                          test_size=
                                                                          len(data_to_div) -
                                                                          int(data_size * splitting_proportions[i]))
        self.data[splitted_fields[-1]] = data_to_div

        if delete_parent:
            a = self.data.pop(field_to_split)
            del a

        return self

    def split(self, splitting_proportions=None, delete_parent=True):

        dd = dict()
        cd = self.classes_distribution
        train = list()
        valid = list()
        test = list()

        if splitting_proportions is None:
            splitting_proportions = [0.1, 0.1]

        if self.data.get('base', []) is not None:
            dataset = self.data['base']
        else:
            raise ValueError("You dataset don't contains 'base' key. If You want to split a specific part dataset,"
                             "please use .simple_split method.")

        for x, y in zip(dataset[self.main_names[0]], dataset[self.main_names[1]]):
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

        self.data['train'] = pd.DataFrame({self.main_names[0]: utrain, self.main_names[1]: ctrain})
        self.data['valid'] = pd.DataFrame({self.main_names[0]: uvalid, self.main_names[1]: cvalid})
        self.data['test'] = pd.DataFrame({self.main_names[0]: utest, self.main_names[1]: ctest})

        if delete_parent:
            a = self.data.pop('base', [])
            del a

        return self

    def iter_batch(self, batch_size: int, data_type: str = 'base') -> Generator:
        """This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
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

        # for i in range((data_len - 1) // batch_size + 1):
        #     yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))
        for i in range((data_len - 1) // batch_size + 1):
            o = order[i * batch_size:(i + 1) * batch_size]
            yield list((list(data[self.main_names[0]][o]), list(data[self.main_names[1]][o])))

    def iter_all(self, data_type: str = 'base') -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in zip(data[self.main_names[0]], data[self.main_names[1]]):
            yield (x, y)

    def merge_data(self, fields_to_merge, delete_parent=True, new_name=None):
        if new_name is None:
            new_name = '_'.join([s for s in fields_to_merge])

        if set(fields_to_merge) <= set(self.data.keys()):
            fraims_to_merge = [self.data[s] for s in fields_to_merge]
            self.data[new_name] = pd.concat(fraims_to_merge)
        else:
            raise KeyError('In dataset no such parts {}'.format(fields_to_merge))

        if delete_parent:
            a = [self.data.pop(x) for x in fields_to_merge]
            del a

        return self

    def del_data(self, fields_to_del):
        for name in fields_to_del:
            a = self.data.pop(name)
            del a
        return self

    def get_classes(self):
        if self.data.get('base') is not None:
            classes = self.data['base'][self.main_names[1]].unique()
        else:
            classes = self.data['train'][self.main_names[1]].unique()
        return classes

    def get_distribution(self):
        try:
            classes_distribution = self.data['base'].groupby(self.main_names[1])[self.main_names[0]].nunique()
        except KeyError:
            classes_distribution = self.data['train'].groupby(self.main_names[1])[self.main_names[0]].nunique()
        return classes_distribution

    def info(self):
        information = dict(data_keys=list(self.data.keys()),
                           classes_description=self.classes_description)

        return information
