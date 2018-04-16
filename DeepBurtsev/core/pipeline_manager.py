from time import time

from DeepBurtsev.core.datasets.dataset import Watcher
from DeepBurtsev.core.datasets.dataset_readers import *
from DeepBurtsev.core.pipelines.pipeline import Pipeline, PrepPipeline
from DeepBurtsev.core.transformers.transformers import *
from .pipeline_generator import PipelineGenerator
from .utils import *


class PipelineManager(object):
    def __init__(self, language, dataset_name, emb_name, emb_dim, hyper_search=False, n=20, seed=42,
                 root='/home/mks/projects/DeepBurtsev/'):
        self.seed = seed
        self.hyper_search = hyper_search
        self.N = n
        self.language = language
        self.dataset_name = dataset_name
        self.emb_name = emb_name
        self.emb_dim = emb_dim
        self.root = root
        self.time_log_file = join(self.root, 'data', self.language, self.dataset_name, 'log_data', 'time_log.json')
        self.data_root = join(self.root, 'data', self.language)
        self.date = datetime.now()
        self.start_dataset = None
        self.time = dict()
        self.pipeline_generator = None  # pipegen.pipeline_gen()

    def init_dataset(self, pure_data, res_type, test_mode=False):
        self.start_dataset = Watcher(pure_data, self.date, self.language, self.dataset_name, res_type,
                                     seed=self.seed)

        if test_mode:
            dataset = self.start_dataset.split([0.1, 0.1])
            data = dataset.data['test']
            self.start_dataset = Watcher(data, self.date, self.language, self.dataset_name, res_type, seed=self.seed)

        return self

    def run(self, pipe, structure, res_type, pure_data, test_mode=False):

        print(res_type)

        pipegen = PipelineGenerator(pipe, structure, self.data_root, self.dataset_name, self.emb_name,
                                    self.emb_dim, res_type=res_type)
        self.pipeline_generator = pipegen.pipeline_gen()

        # Start generating pipelines configs
        for x in self.pipeline_generator:
            prer_pipe = x[0][:-3]  # 2
            model_pipe = x[0][-3:]  # 2
            pipe_conf = x[1]

            # time meshure
            model_name = list(pipe_conf.keys())[-2].split('_')[0]
            pipe_name = '___'.join(list(pipe_conf.keys()))
            self.time[model_name] = dict()

            self.time[model_name][pipe_name] = dict()

            self.time[model_name][pipe_name]['start'] = time()

            prer_pipeline = PrepPipeline(prer_pipe, mode='infer', output='dataset')
            # prer_pipeline = Pipeline(prer_pipe, mode='infer', output='dataset')

            # initialize new dataset
            self.init_dataset(pure_data, res_type, test_mode=test_mode)

            self.time[model_name][pipe_name]['dataset_init'] = time() - self.time[model_name][pipe_name]['start']
            d_ = prer_pipeline.run(self.start_dataset)

            self.time[model_name][pipe_name]['end_of_preprocess'] = time() - \
                                                                    self.time[model_name][pipe_name]['dataset_init']

            if not self.hyper_search:
                self.time[model_name][pipe_name]['hyper'] = False
                self.time[model_name][pipe_name]['start_model_pipe'] = time()

                model_pipeline = Pipeline(model_pipe, mode='infer', output='dataset')
                end_dataset = model_pipeline.run(d_)

                self.time[model_name][pipe_name]['end_model_pipe'] = \
                    time() - self.time[model_name][pipe_name]['start_model_pipe']

                self.time[model_name][pipe_name]['full_time'] = time() - self.time[model_name][pipe_name]['start']
            else:
                self.time[model_name][pipe_name]['hyper'] = True
                self.time[model_name][pipe_name]['start_model_pipe'] = time()

                model_conf_name = list(pipe_conf.keys())[-2].split('_')[0] + '_params.json'
                model_conf = pipe_conf[list(pipe_conf.keys())[-2]]
                path_to_model_conf = join(self.root, 'configs', 'models', model_conf_name)

                if isfile(path_to_model_conf):
                    with open(path_to_model_conf, 'r') as file:
                        search_conf = json.load(file)
                        file.close()
                else:
                    raise FileExistsError('File {0} is not exist in folder: '
                                          '{1} .'.format(path_to_model_conf.split('/')[-1],
                                                         path_to_model_conf))

                params_generator = ConfGen(model_conf, search_conf, seed=self.seed)
                for params in params_generator.generator(self.N):
                    try:
                        # TODO refactor
                        for u in params.keys():
                            if isinstance(params[u], np.int64):
                                params[u] = int(params[u])

                        model_op = model_pipe[0][0]
                        mod = (model_op, params)
                        model_pipe[0] = mod

                        model_pipeline = Pipeline(model_pipe, mode='infer', output='dataset')
                        end_dataset = model_pipeline.run(d_)

                        self.time[model_name][pipe_name]['end_model_pipe'] = \
                            time() - self.time[model_name][pipe_name]['start_model_pipe']

                        self.time[model_name][pipe_name]['full_time'] = time() - self.time[model_name][pipe_name]['start']
                    except:
                        print(params)
                        raise

        results_summarization(self.root, self.date, self.language, self.dataset_name)
        self.time_log()

        return None

    def time_log(self):
        if not isdir(join(self.root, 'data', self.language, self.dataset_name, 'log_data')):
            os.makedirs(join(self.root, 'data', self.language, self.dataset_name, 'log_data'))

        with open(self.time_log_file, 'w') as log:
            json.dump(self.time, log)
        return self
