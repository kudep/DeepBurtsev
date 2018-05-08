from itertools import product
from deepburtsev.core.utils import HyperPar
from deepburtsev.core.pipeline import Pipeline


class PipelineGenerator(object):
    def __init__(self, structure, n=10, dtype='list'):
        self.structure = structure
        self.dtype = dtype
        self.check_struct()
        self.N = n
        self.pipes = []

        self.generator = self.pipeline_gen()
        self.length = self.get_len()

    def get_len(self):
        k = 0
        for x in self.generator:
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
                        conf_gen = self.rand_param_gen(x[1])
                        op = x[0]
                        lst.remove(x)
                        for conf in conf_gen:
                            lst.append((op, conf))

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
