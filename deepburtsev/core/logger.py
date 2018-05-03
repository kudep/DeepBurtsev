import json
import os

from collections import OrderedDict
from os.path import join, isdir


class Logger(object):
    def __init__(self, exp_name, root, language, dataset_name, date, hp_search):
        self.exp_name = exp_name
        self.root = root
        self.language = language
        self.dataset_name = dataset_name
        self.date = date
        self.hp_search = hp_search

        # tmp parameters
        self.model = None
        self.pipe_ind = 0
        self.pipe_conf = None
        self.metrics = None
        self.pipe_start_time = None
        self.pipe_end_time = None

        # build folder dependencies
        self.log_path = join(self.root, 'results', self.language, self.dataset_name,
                             '{0}-{1}-{2}'.format(date.year, date.month, date.day),
                             self.exp_name)
        self.log_file = join(self.log_path, self.exp_name + '.json')

        if not isdir(self.log_path):
            os.makedirs(self.log_path)
            os.makedirs(join(self.log_path, 'images'))

        self.log = OrderedDict(experiment_parameters=OrderedDict(date='{0}-{1}-{2}'.format(date.year,
                                                                                           date.month,
                                                                                           date.day),
                                                                 exp_name=self.exp_name,
                                                                 language=self.language,
                                                                 dataset_name=self.dataset_name,
                                                                 hp_search=self.hp_search,
                                                                 root=self.root),
                               dataset={'init_time': None},
                               experiments=OrderedDict())

    def save(self):
        with open(self.log_file, 'w') as log_file:
            json.dump(self.log, log_file)

    def pipe_log(self, conf, time):
        last_op_name = list(conf.keys())[-1]
        last_conf = conf.pop(last_op_name)
        del last_conf

        model = list(conf.keys())[-1].split('_')[0]

        self.model = model
        self.pipe_ind += 1
        self.pipe_conf = conf
        self.pipe_start_time = time

        pipe_name = '-->'.join([x.split('_')[0] for x in list(conf.keys())])

        if model not in self.log['experiments'].keys():
            self.pipe_ind = 1
            self.log['experiments'][model] = OrderedDict()
            self.log['experiments'][model][self.pipe_ind] = OrderedDict({'pipeline_config': conf,
                                                                         'light_config': pipe_name,
                                                                         'results': None,
                                                                         'time': time,
                                                                         'ops_time': {}})
        else:
            self.log['experiments'][model][self.pipe_ind] = OrderedDict({'config': conf,
                                                                         'light_config': pipe_name,
                                                                         'results': None,
                                                                         'time': time,
                                                                         'ops_time': {}})
        return self

    def hipe_log(self, conf, time):
        self.pipe_ind += 1
        self.pipe_conf[list(self.pipe_conf.keys())[-1]] = conf

        pipe_name = '-->'.join([x.split('_')[0] for x in list(self.pipe_conf.keys())])

        if self.model not in self.log['experiments'].keys():
            self.log['experiments'][self.model] = OrderedDict()
            self.log['experiments'][self.model][self.pipe_ind] = OrderedDict({'config': self.pipe_conf,
                                                                              'light_config': pipe_name,
                                                                              'results': None,
                                                                              'time': time,
                                                                              'ops_time': {}})
        else:
            self.log['experiments'][self.model][self.pipe_ind] = OrderedDict({'config': self.pipe_conf,
                                                                              'light_config': pipe_name,
                                                                              'results': None,
                                                                              'time': time,
                                                                              'ops_time': {}})
        return self

    def get_res(self, res, time, metrics):
        self.metrics = metrics
        self.log['experiments'][self.model][self.pipe_ind]['results'] = res
        self.log['experiments'][self.model][self.pipe_ind]['time'] = time
        return self
