import numpy as np
from itertools import product
from copy import deepcopy

from deepburtsev.core.utils import HyperPar
from deepburtsev.core.pipeline import Pipeline


class PipelineGenerator(object):
    def __init__(self, structure, n=10, dtype='list', search='grid'):
        self.structure = structure
        self.dtype = dtype
        self.N = n
        self.search = search

        if self.search == 'grid':
            self.generator = GridGenerator(self.structure)
            self.length = self.generator.len
        elif self.search == 'random':
            self.generator = RandomGenerator(self.structure, self.N)
            self.length = self.generator.len
        else:
            raise ValueError("{} search type not implemented.".format(self.search))

    def __call__(self, *args, **kwargs):
        return self.generator


class RandomGenerator(object):
    def __init__(self, structure, n=10):
        self.structure = structure
        self.N = n
        self.pipes = []
        self.len = 0

        self.get_len()
        self.generator = self.pipeline_gen()

    def __call__(self, *args, **kwargs):
        return self.generator

    def get_len(self):
        test = []
        lst = []

        for x in self.structure:
            if not isinstance(x, list):
                if not isinstance(x, tuple):
                    test.append([False])
                else:
                    assert len(x) == 2
                    assert isinstance(x[1], dict)
                    if "search" not in x[1].keys():
                        test.append([False])
                    else:
                        test.append([True])
            else:
                ln = []
                for y in x:
                    if not isinstance(y, tuple):
                        ln.append(False)
                    else:
                        assert len(y) == 2
                        assert isinstance(y[1], dict)
                        if "search" not in y[1].keys():
                            ln.append(False)
                        else:
                            ln.append(True)
                test.append(ln)

        zgen = product(*test)
        for x in zgen:
            lst.append(x)

        ks = 0
        k = 0
        for x in lst:
            if True not in x:
                k += 1
            else:
                ks += 1

        self.len = k + ks * self.N

        del test, lst, zgen

        return self

    # generation
    def conf_gen(self):
        for i, x in enumerate(self.structure):
            if isinstance(x, list):
                self.pipes.append(x)
            else:
                self.pipes.append([x])

        lgen = product(*self.pipes)
        for pipe in lgen:
            search = False
            pipe = list(pipe)

            for op in pipe:
                if isinstance(op, tuple) and "search" in op[1].keys():
                    search = True
                    break

            if search:
                for i in range(self.N):
                    for j, op in enumerate(pipe):
                        if isinstance(op, tuple) and "search" in op[1].keys():
                            search_conf = deepcopy(op[1])
                            del search_conf['search']

                            conf = HyperPar(**search_conf).sample_params()
                            # fix dtype for json dump
                            for key in conf.keys():
                                if isinstance(conf[key], np.int64):
                                    conf[key] = int(conf[key])

                            pipe[j] = (op[0], conf)

                    yield pipe
            else:
                yield pipe

    def pipeline_gen(self):
        pipe_gen = self.conf_gen()
        for pipe in pipe_gen:
            yield Pipeline(list(pipe))


class GridGenerator(object):
    def __init__(self, structure):
        self.structure = structure
        self.pipes = []
        self.len = 1

        self.get_len()
        self.generator = self.pipeline_gen()

    @staticmethod
    def get_p(z):
        assert len(z) == 2
        assert isinstance(z[1], dict)
        if 'search' in z[1].keys():
            l_ = list()
            for key, it in z[1].items():
                if key == 'search':
                    pass
                else:
                    if isinstance(it, list):
                        l_.append(len(it))
                    else:
                        pass
            p = 1
            for q in l_:
                p *= q
            return p
        else:
            return 1

    def get_len(self):
        leng = []

        for x in self.structure:
            if not isinstance(x, list):
                if not isinstance(x, tuple):
                    leng.append(1)
                else:
                    leng.append(self.get_p(x))
            else:
                k = 0
                for y in x:
                    if not isinstance(y, tuple):
                        k += 1
                    else:
                        k += self.get_p(y)
                leng.append(k)

        for x in leng:
            self.len *= x

        return self

    # generation
    def conf_gen(self):

        def update(el):
            lst = []
            if not isinstance(el, tuple):
                lst.append(el)
            else:
                if 'search' not in el[1].keys():
                    lst.append(el)
                else:
                    lst.extend(self.grid_param_gen(el))
            return lst

        for i, x in enumerate(self.structure):
            if not isinstance(x, list):
                self.pipes.append(update(x))
            else:
                ln = []
                for y in x:
                    ln.extend(update(y))
                self.pipes.append(ln)

        return product(*self.pipes)

    def pipeline_gen(self):
        pipe_gen = self.conf_gen()
        for pipe in pipe_gen:
            yield Pipeline(list(pipe))

    def __call__(self, *args, **kwargs):
        return self.generator

    @staticmethod
    def grid_param_gen(element):
        op = element[0]
        search_conf = deepcopy(element[1])
        list_of_var = []

        # delete "search" key and element
        del search_conf['search']

        values = list()
        keys = list()

        static_keys = list()
        static_values = list()
        for key, item in search_conf.items():
            if isinstance(search_conf[key], list):
                values.append(item)
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
            cop = deepcopy(config)
            for i, v in enumerate(val):
                cop[keys[i]] = v
            list_of_var.append((op, cop))

        return list_of_var
