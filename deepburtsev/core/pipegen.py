import numpy as np
from itertools import product
from deepburtsev.core.utils import HyperPar
from deepburtsev.core.pipeline import Pipeline


class PipelineGenerator(object):
    def __init__(self, structure, n=10, dtype='list', search='grid'):
        self.structure = structure
        self.dtype = dtype
        self.check_struct()
        self.N = n
        self.search = search
        self.pipes = []

        # self.length = self.get_len()
        self.generator = self.pipeline_gen()

    def get_len(self):
        k = 0
        for x in self.pipeline_gen():
            k += 1
        return k

    def __call__(self, *args, **kwargs):
        return self.generator

    def check_struct(self):
        if not isinstance(self.structure, list):
            raise ValueError("Input must be a list.")

    # generation
    def conf_gen(self):
        for i, x in enumerate(self.structure):
            if isinstance(x, list):
                self.pipes.append(x)
            else:
                self.pipes.append([x])

        for lst in self.pipes:
            for x in lst:
                if not isinstance(x, tuple):
                    pass
                else:
                    assert len(x) == 2
                    if not isinstance(x[1], dict):
                        raise ValueError("Configuration of operation or search must have a dict type.")

                    if 'search' not in x[1].keys():
                        pass
                    else:
                        if self.search == 'random':
                            conf_gen = self.rand_param_gen(x[1])
                            op = x[0]
                            lst.remove(x)
                            for conf in conf_gen:
                                #######################################
                                for key in conf.keys():
                                    if isinstance(conf[key], np.int64):
                                        conf[key] = int(conf[key])
                                #######################################
                                lst.append((op, conf))
                        elif self.search == 'grid':
                            conf_gen = self.grid_param_gen(x[1])
                            op = x[0]
                            lst.remove(x)
                            for conf in conf_gen:
                                lst.append((op, conf))
                        else:
                            raise NotImplementedError("'{}' search are not implement.".format(self.search))

        return product(*self.pipes)

    def pipeline_gen(self):
        pipe_gen = self.conf_gen()
        for pipe in pipe_gen:
            yield Pipeline(list(pipe))

    def rand_param_gen(self, conf):
        search_conf = conf
        del search_conf['search']

        param_gen = HyperPar(**search_conf)
        for i in range(self.N):
            yield param_gen.sample_params()

    @staticmethod
    def grid_param_gen(conf):
        search_conf = conf
        del search_conf['search']

        values = list()
        keys = list()

        static_keys = list()
        static_values = list()
        for key in search_conf.keys():
            if isinstance(search_conf[key], list):
                values.append(search_conf[key])
                keys.append(key)
            elif isinstance(search_conf[key], dict):
                raise ValueError("Grid search are not supported 'dict', that contain values of parameters.")
            elif isinstance(search_conf[key], tuple):
                raise ValueError("Grid search are not supported 'tuple', that contain values of parameters.")
            else:
                static_values.append(search_conf[key])
                static_keys.append(key)

        valgen = product(*values)

        config = {}
        for i in range(len(static_keys)):
            config[static_keys[i]] = static_values[i]

        for val in valgen:
            for i in range(len(keys)):
                config[keys[i]] = val[i]

            yield config
