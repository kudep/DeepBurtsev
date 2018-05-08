import json
import os

from collections import OrderedDict
from os.path import join, isdir


class Logger(object):
    def __init__(self, name, root, info, date, target_metric='f1_weighted'):
        self.exp_name = name
        self.exp_inf = info
        self.root = root
        self.date = date
        self.target_metric = target_metric
        self.metrics = []

        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.ops = {}

        # build folder dependencies
        self.log_path = join(self.root, 'results', '{0}-{1}-{2}'.format(date.year, date.month, date.day), self.exp_name)
        self.log_file = join(self.log_path, self.exp_name + '.json')

        if not isdir(self.log_path):
            os.makedirs(self.log_path)
            os.makedirs(join(self.log_path, 'images'))

        self.log = OrderedDict(experiment_info=OrderedDict(date='{0}-{1}-{2}'.format(date.year, date.month, date.day),
                                                           exp_name=self.exp_name,
                                                           root=self.root,
                                                           info=self.exp_inf),
                               dataset={},
                               experiments=OrderedDict())

    def tmp_reset(self):
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.ops = {}

    def save(self):
        with open(self.log_file, 'w') as log_file:
            json.dump(self.log, log_file)

    def add_metrics(self, metrics):
        if isinstance(metrics, str):
            if len(self.metrics) == 0:
                self.metrics.append(metrics)
            else:
                if metrics not in self.metrics:
                    self.metrics.append(metrics)
                else:
                    pass
        elif isinstance(metrics, list):
            if len(self.metrics) == 0:
                self.metrics.extend(metrics)
            else:
                for x in metrics:
                    if x not in self.metrics:
                        self.metrics.append(x)
                    else:
                        pass

        return self

    def get_pipe_log(self):
        ops_times = {}
        self.pipe_conf = OrderedDict()
        for i in range(len(self.ops.keys())):
            time = self.ops[str(i)].pop('time')
            name = self.ops[str(i)]['op_name']

            # add metrics
            if name == 'ResultsCollector':
                self.add_metrics(self.ops[str(i)]['metrics'])
            # find main model
            if self.ops[str(i)]['op_type'] == 'model':
                self.model = name

            self.pipe_conf[name] = {'conf': self.ops[str(i)]}
            ops_times[name] = time

        pipe_name = '-->'.join([x.split('_')[0] for x in list(self.pipe_conf.keys())])

        if self.model not in self.log['experiments'].keys():
            self.log['experiments'][self.model] = OrderedDict()
            self.log['experiments'][self.model][self.pipe_ind] = {'config': self.pipe_conf,
                                                                  'light_config': pipe_name,
                                                                  'time': self.pipe_time,
                                                                  'ops_time': ops_times,
                                                                  'results': self.pipe_res}
        else:
            self.log['experiments'][self.model][self.pipe_ind] = {'config': self.pipe_conf,
                                                                  'light_config': pipe_name,
                                                                  'time': self.pipe_time,
                                                                  'ops_time': ops_times,
                                                                  'results': self.pipe_res}

        self.tmp_reset()
        return self

    def results_analysis(self):

        pass
