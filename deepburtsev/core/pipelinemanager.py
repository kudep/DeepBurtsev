from .pipegen import PipelineGenerator
from .utils import *


class PipelineManager(object):
    def __init__(self, dataset, structure, info, root='./experiments/', analitic_func=None, k_fold=None, k_num=1,
                 seed=42, hyper_search='random', sample_num=10):
        self.dataset = dataset
        self.structure = structure
        self.info = info
        self.data_func = analitic_func
        self.k_fold = k_fold
        self.k_num = k_num
        self.seed = seed
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.date = datetime.now()

        self.root = root
        self.pipeline_generator = None

        if isinstance(self.structure, list):
            self.structure_type = 'list'
        elif isinstance(self.structure, dict):
            # self.structure_type = 'dict'
            raise ValueError("Dict structure as input parameter not implemented yet.")
        else:
            raise ValueError("Structure parameter must be a list or dict")

        # self.time = {}
        # self.logger = Logger(exp_name, root, language, dataset_name, self.date, self.hyper_search)
        # self.logger.log['experiment_parameters']['full_tume'] = time()

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
            data_info = self.data_func(self.dataset)

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
            dataset_i = self.dataset
            for j in range(pipe.length):
                try:
                    conf = pipe.get_op_config(j)
                    dataset_i = pipe.step(j, dataset_i)
                except:
                    print('Operation with number {0};'.format(i + 1))
                    raise

        return None
