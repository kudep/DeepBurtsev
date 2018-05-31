import json
import os

from collections import OrderedDict
from os.path import join, isdir, isfile
from deepburtsev.core.utils import normal_time


class Logger(object):
    def __init__(self, name, root, info, date):
        self.exp_name = name
        self.exp_inf = info
        self.root = root
        self.date = date

        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.ops = {}

        # build folder dependencies
        self.log_path = join(self.root, '{0}-{1}-{2}'.format(date.year, date.month, date.day), self.exp_name)
        self.log_file = join(self.log_path, self.exp_name + '.json')

        if not isdir(self.log_path):
            os.makedirs(self.log_path)
        if not isdir(join(self.log_path, 'results', 'images')):
            os.makedirs(join(self.log_path, 'results', 'images'))

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
        if not isfile(self.log_file):
            with open(self.log_file, 'w') as log_file:
                json.dump(self.log, log_file)
        else:
            with open(self.log_file, 'r') as old_file:
                old_log = json.load(old_file)
                old_file.close()

            self.log = self.merge_logs(old_log, self.log)
            with open(self.log_file, 'w') as log_file:
                json.dump(self.log, log_file)

    @staticmethod
    def merge_logs(old_log, new_log):
        new_models_names = list(new_log['experiments'].keys())

        for name in new_models_names:
            if name not in old_log['experiments'].keys():
                old_log['experiments'][name] = new_log['experiments'][name]
            else:
                old_npipe = len(old_log['experiments'][name]) - 1
                for nkey, nval in new_log['experiments'][name].items():
                    match = False
                    for okey, oval in old_log['experiments'][name].items():
                        if nval['config'] == oval['config']:
                            old_log['experiments'][name][okey] = new_log['experiments'][name][nkey]
                            match = True
                        else:
                            pass

                    if not match:
                        old_log['experiments'][name][str(old_npipe+1)] = new_log['experiments'][name][nkey]

        # addition time
        t_old = old_log['experiment_info']['full_time'].split(':')
        t_new = new_log['experiment_info']['full_time'].split(':')
        sec = int(t_old[2]) + int(t_new[2]) + (int(t_old[1]) + int(t_new[1]))*60 + (int(t_old[0]) + int(t_new[0]))*3600

        old_log['experiment_info']['full_time'] = normal_time(sec)

        return old_log

    def get_pipe_log(self):
        ops_times = {}
        self.pipe_conf = OrderedDict()
        for i in range(len(self.ops.keys())):
            time = self.ops[str(i)].pop('time')
            name = self.ops[str(i)]['op_name']

            # find main model
            if self.ops[str(i)]['op_type'] == 'model':
                self.model = self.ops[str(i)]['op_name']
            else:
                pass

            self.pipe_conf[name + '_' + self.ops[str(i)]['op_type']] = {'conf': self.ops[str(i)]}
            ops_times[name] = time

        pipe_name = '-->'.join([x for x in list(self.pipe_conf.keys())[:-1]])

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
