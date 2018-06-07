# import pandas as pd
import json
import secrets
import os

from os.path import join, isfile, isdir
from collections import OrderedDict


# class Watcher(object):
#     def __init__(self, root, seed=None):
#         self.root = root
#         self.seed = seed
#         self.pipeline_config = OrderedDict()
#         self.conf_dict = join(self.root, 'log_data')
#         self.save_path = join(self.conf_dict, 'data')
#
#     def add_config(self, conf):
#         name = conf['op_name']
#         op_type = conf['op_type']
#         self.pipeline_config[name + '_' + op_type] = conf
#         return name, op_type
#
#     def test_config(self, conf, dictionary):
#         name, op_type = self.add_config(conf)
#         if name == 'ResultsCollector':
#             return True
#         elif op_type == 'model' or op_type == 'vectorizer':
#             return True
#         else:
#             status = self.check_config(self.pipeline_config)
#
#             if isinstance(status, bool):
#                 if status:
#                     # self.save_data(self.pipeline_config)
#                     return False
#             elif isinstance(status, str):
#                 d = self.load_data(status, dictionary)
#                 return d
#             else:
#                 print(type(status))
#                 raise ValueError('Incorrect')
#
#         return self
#
#     def check_config(self, conf):
#         # check file
#         if not isdir(self.conf_dict):
#             os.makedirs(self.conf_dict)
#         if not isfile(join(self.conf_dict, 'pipe_conf_dict.json')):
#             with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
#                 d.write('{}')
#                 d.close()
#
#         # read config file
#         with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r+') as d:
#             conf_ = json.load(d)
#
#             if len(list(conf_.keys())) == 0:
#                 d.close()
#                 return True
#             else:
#                 coincidence = False
#                 for name in conf_.keys():
#                     if conf_[name] == conf:
#                         coincidence = True
#                         d.close()
#                         return name
#                 if not coincidence:
#                     d.close()
#                     return True
#         return None
#
#     def save_data(self, dictionary):
#         dataframes = []
#         names = []
#         main_names = ['train', 'test', 'valid']
#
#         for name in main_names:
#             if name in dictionary.keys():
#                 dataframes.append({name: {'x': dictionary[name]['x'], 'y': dictionary[name]['y']}})
#                 names.append(name)
#             else:
#                 pass
#
#         for i, x in enumerate(dataframes):
#             dataframes[i] = pd.DataFrame(x)
#
#         data = pd.concat(dataframes, keys=names)
#
#         # saving in file
#         secret_name = secrets.token_hex(nbytes=16)
#
#         if not os.path.isdir(self.save_path):
#             os.makedirs(self.save_path)
#
#         path = join(self.save_path, secret_name + '.csv')  # + '.csv'
#         data.to_csv(path)
#
#         # write in conf_dict.json
#         if isfile(join(self.conf_dict, 'pipe_conf_dict.json')):
#             with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r') as d:
#                 conf_ = json.load(d)
#                 d.close()
#
#             conf_[secret_name] = self.pipeline_config
#             with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
#                 line = json.dumps(conf_)
#                 d.write(line)
#                 d.close()
#
#         else:
#             conf_ = dict()
#             conf_[secret_name] = self.pipeline_config
#             with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'w') as d:
#                 line = json.dumps(conf_)
#                 d.write(line)
#                 d.close()
#
#         return self
#
#     def load_data(self, name, dictionary):
#         filepath = join(self.save_path, name + '.csv')
#         data = pd.read_csv(filepath)
#
#         with open(join(self.conf_dict, 'pipe_conf_dict.json'), 'r') as f:
#             conf = json.load(f)
#             f.close()
#
#         config = conf[name]
#
#         sam = lambda s: [x[1:-1] for x in s[1:-1].split(', ')]
#
#         keys = list(data['Unnamed: 0'].unique())
#
#         for key in keys:
#             if 'NLTK_Tokenizer' in config.keys() and 'Concatenator_transformer' not in config.keys():
#                 dictionary[key]['x'] = data[data['Unnamed: 0'] == key]['x'].apply(sam)
#             else:
#                 dictionary[key]['x'] = data[data['Unnamed: 0'] == key]['x']
#
#             dictionary[key]['y'] = data[data['Unnamed: 0'] == key]['y']
#             dictionary[key].dropna(inplace=True)
#             # dictionary[key].reset_index(inplace=True)
#
#         del data
#
#         return dictionary


class Watcher(object):
    def __init__(self, root, seed=None):
        self.root = root
        self.seed = seed
        self.pipeline_config = OrderedDict()
        self.conf_dict = join(self.root, 'log_data')
        self.save_path = join(self.conf_dict, 'data')

    def add_config(self, conf):
        name = conf['op_name']
        op_type = conf['op_type']
        self.pipeline_config[name + '_' + op_type] = conf
        return name, op_type

    def test_config(self, conf, dictionary):
        name, op_type = self.add_config(conf)
        if name == 'ResultsCollector':
            return True
        elif op_type == 'model' or op_type == 'vectorizer':
            return True
        else:
            status = self.check_config(self.pipeline_config)

            if isinstance(status, bool):
                if status:
                    # self.save_data(self.pipeline_config)
                    return False
            elif isinstance(status, str):
                d = self.load_data(status, dictionary)
                return d
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

    def save_data(self, dictionary):
        # saving in file
        secret_name = secrets.token_hex(nbytes=16)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        path = join(self.save_path, secret_name + '.json')
        with open(path, 'w') as f:
            json.dump(dictionary, f)

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

    def load_data(self, name, dictionary):
        filepath = join(self.save_path, name + '.json')
        with open(filepath, 'r') as f:
            data = json.load(f)

        for key, item in data.items():
            dictionary[key] = item

        return dictionary
