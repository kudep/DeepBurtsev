import random
import pandas as pd
import json
import secrets
from os.path import join, isfile, isdir
import os
from collections import OrderedDict
from typing import Generator
from sklearn.model_selection import train_test_split


class Dataset(object):

    def __init__(self, data, seed=None, classes_description=None, *args, **kwargs):

        self.main_names = ['request', 'report']
        self.pipeline_config = OrderedDict()

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.data = dict()

        if data.get('train') is not None:
            self.data['train'] = data.get('train')
        elif data.get('test') is not None:
            self.data['test'] = data.get('test')
        elif data.get('valid') is not None:
            self.data['valid'] = data.get('valid')
        else:
            self.data['base'] = data

        self.classes_description = classes_description

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

    def iter_batch(self, batch_size: int, data_type: str = 'base', shuffle: bool = True,
                   only_request: bool = False) -> Generator:
        """This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): shuffle trigger
            only_request (bool): trigger that told what data will be returned
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        if shuffle:
            random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        # for i in range((data_len - 1) // batch_size + 1):
        #     yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))
        if not only_request:
            for i in range((data_len - 1) // batch_size + 1):
                o = order[i * batch_size:(i + 1) * batch_size]
                yield list((list(data[self.main_names[0]][o]), list(data[self.main_names[1]][o])))
        else:
            for i in range((data_len - 1) // batch_size + 1):
                o = order[i * batch_size:(i + 1) * batch_size]
                yield list((list(data[self.main_names[0]][o]), ))

    def iter_all(self, data_type: str = 'base', only_request: bool = False) -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
            only_request (bool): trigger that told what data will be returned
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in zip(data[self.main_names[0]], data[self.main_names[1]]):
            if not only_request:
                yield (x, y)
            else:
                yield (x, )

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

    def add_config(self, conf):
        name = conf['name']
        op_type = conf['op_type']
        self.pipeline_config[name + '_' + op_type] = conf
        return self


class Watcher(Dataset):
    def __init__(self, data, date, language, dataset_name, restype, seed=None, classes_description=None, root=None,
                 *args, **kwargs):

        super().__init__(data, seed, classes_description, *args, **kwargs)

        self.date = '{}-{}-{}'.format(date.year, date.month, date.day)
        self.language = language
        self.dataset_name = dataset_name
        self.restype = restype

        if root is None:
            root = '/home/mks/projects/DeepBurtsev/'

        self.conf_dict = join(root, 'data', language, dataset_name, 'log_data')
        self.save_path = join(self.conf_dict, 'data')

    def test_config(self, conf):
        self.add_config(conf)
        status = self.check_config(self.pipeline_config)

        if isinstance(status, bool):
            if status:
                # self.save_data(self.pipeline_config)
                return False
        elif isinstance(status, str):
            # self.load_data(status)
            return status
        else:
            print(type(status))
            raise ValueError('Incorrect')

        return self

    def check_config(self, conf):
        # check file
        if not isdir(self.conf_dict):
            os.makedirs(self.conf_dict)
        if not isfile(join(self.conf_dict, 'pipe_conf_dict.json')):
            with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
                d.write('{}')
                d.close()

        # read config file
        with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r+') as d:
            conf_ = json.load(d)

            if len(list(conf_.keys())) == 0:
                d.close()
                return True
            else:
                coincidence = False
                for name in conf_.keys():
                    if conf_[name] == conf:
                        coincidence = True
                        d.close()
                        return name
                if not coincidence:
                    d.close()
                    return True
        return None

    def save_data(self):
        names = self.data.keys()
        dataframes = []
        datanames = []
        for name in names:
            if isinstance(self.data[name], pd.DataFrame):
                dataframes.append(self.data[name])
                datanames.append(name)
        data = pd.concat(dataframes, keys=datanames)

        # saving in file
        secret_name = secrets.token_hex(nbytes=16)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        path = join(self.save_path, secret_name + '.csv')  # + '.csv'
        data.to_csv(path)

        # write in conf_dict.json
        if isfile(join(self.conf_dict, 'pipe_conf_dict.json')):
            with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r') as d:
                conf_ = json.load(d)
                d.close()

            conf_[secret_name] = self.pipeline_config
            with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
                line = json.dumps(conf_)
                d.write(line)
                d.close()

        else:
            conf_ = dict()
            conf_[secret_name] = self.pipeline_config
            with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
                line = json.dumps(conf_)
                d.write(line)
                d.close()

        return self

    def load_data(self, name):
        filepath = join(self.save_path, name + '.csv')
        file = open(filepath, 'r')
        data = pd.read_csv(file)
        file.close()

        request, report = self.main_names

        with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r') as f:
            conf = json.load(f)
            f.close()

        config = conf[name]

        keys = list(data['Unnamed: 0'].unique())
        data_keys = list(self.data.keys())
        sam = lambda s: [x[1:-1] for x in s[1:-1].split(', ')]

        for key in keys:
            if key not in data_keys:
                self.data[key] = {}

            if 'Tokenizator_transformer' in config.keys() and 'TextConcatenator_transformer' not in config.keys():
                self.data[key][request] = data[data['Unnamed: 0'] == key][request].apply(sam)
            else:
                self.data[key][request] = data[data['Unnamed: 0'] == key][request]
            self.data[key][report] = data[data['Unnamed: 0'] == key][report]
            self.data[key].dropna(inplace=True)
            # self.data[key].reset_index(inplace=True)

        for key in data_keys:
            if key not in keys:
                self.del_data([key])

        del data

        #################################################
        # changes
        # for key in keys:
        #     self.data[key] = self.data[key].dropna().reset_index()

        return self
