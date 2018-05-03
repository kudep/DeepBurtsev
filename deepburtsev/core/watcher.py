import pandas as pd
import json
import secrets
import os

from os.path import join, isfile, isdir
from collections import OrderedDict


class Watcher(object):
    def __init__(self, date, language, dataset_name, seed=None, root=None):

        self.date = '{}-{}-{}'.format(date.year, date.month, date.day)
        self.language = language
        self.dataset_name = dataset_name
        self.pipeline_config = OrderedDict()
        self.seed = seed

        # TODO fix
        if root is None:
            self.root = '/home/mks/projects/deepburtsev/'
        else:
            self.root = root

        self.conf_dict = join(self.root, 'log_data')
        self.save_path = join(self.conf_dict, 'data')

    def add_config(self, conf):
        name = conf['name']
        op_type = conf['op_type']
        self.pipeline_config[name + '_' + op_type] = conf
        return self

    def del_data(self, fields_to_del):
        for name in fields_to_del:
            a = self.data.pop(name)
            del a
        return self

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

        return self
