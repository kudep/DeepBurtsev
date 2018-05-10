from time import time
from datetime import datetime
from .pipegen import PipelineGenerator
from .logger import Logger
from .watcher import Watcher
from .utils import normal_time


class PipelineManager(object):
    def __init__(self, dataset, structure, exp_name, info=None, root='./experiments/', analitic_func=None, k_fold=None, k_num=1,
                 seed=42, hyper_search='random', sample_num=10, add_watcher=True):
        self.dataset = dataset
        self.structure = structure
        self.exp_name = exp_name
        self.info = info
        self.data_func = analitic_func
        self.k_fold = k_fold
        self.k_num = k_num
        self.seed = seed
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.date = datetime.now()
        self.add_watcher = add_watcher

        self.root = root
        self.pipeline_generator = None

        if isinstance(self.structure, list):
            self.structure_type = 'list'
        elif isinstance(self.structure, dict):
            # self.structure_type = 'dict'
            raise ValueError("Dict structure as input parameter not implemented yet.")
        else:
            raise ValueError("Structure parameter must be a list or dict")

        self.logger = Logger(exp_name, root, self.info, self.date)
        self.start_exp = time()

    def check_dataset(self):
        if isinstance(self.dataset, dict):
            if not ('train' in self.dataset.keys() and 'test' in self.dataset.keys()):
                raise ValueError("Input dataset must contain 'train' and 'test' keys with data.")
            elif len(self.dataset['train']) == 0 or len(self.dataset['test']) == 0:
                raise ValueError("Input dict is empty.")
        else:
            raise ValueError("Input dataset must be a dict.")
        return self

    def run(self):
        self.check_dataset()

        # analytics of dataset
        if self.data_func is not None:
            an_start = time()
            data_info = self.data_func(self.dataset)
            self.logger.log['dataset']['time'] = normal_time(time() - an_start)
            self.logger.log['dataset'].update(**data_info)

        # TODO make grid_search and fix it
        # create PipelineGenerator
        # if self.hyper_search == 'random':
        #     self.pipeline_generator = PipelineGenerator(self.structure, n=self.sample_num, dtype='list')
        # if self.hyper_search == 'grid':
        #     # self.pipeline_generator = PipelineGenerator(self.structure, n=self.sample_num, dtype='list')
        #     raise ValueError("Grid search not implemented yet.")
        # else:
        #     raise ValueError("{} search not implemented.".format(self.hyper_search))

        # it just zaglushka
        self.pipeline_generator = PipelineGenerator(self.structure, n=self.sample_num, dtype='list')

        # Start generating pipelines configs
        for i, pipe in enumerate(self.pipeline_generator()):
            self.logger.pipe_ind = i
            pipe_start = time()
            dataset_i = self.dataset

            # add watcher if need
            if self.add_watcher:
                watcher = Watcher(self.root, self.seed)

            for j in range(pipe.length):
                try:
                    op_start = time()
                    conf = pipe.get_op_config(j)
                    self.logger.ops[str(j)] = conf

                    if self.add_watcher:
                        test = watcher.test_config(conf, dataset_i)
                        if test is False:
                            dataset_i = pipe.step(j, dataset_i)
                            watcher.save_data(dataset_i)
                        else:
                            dataset_i = test

                    else:
                        dataset_i = pipe.step(j, dataset_i)

                    t = {'time': normal_time(time() - op_start)}
                    self.logger.ops[str(j)].update(**t)
                except:
                    print('Operation with number {0};'.format(i + 1))
                    raise

            self.logger.pipe_time = normal_time(time() - pipe_start)
            self.logger.pipe_res = dataset_i.data['results']
            self.logger.get_pipe_log()

        self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        self.logger.save()

        # TODO refactor this function
        # results_summarization()

        return None
