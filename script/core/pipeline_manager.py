from .pipe_gen import PipelineGenerator
from .pipeline import Pipeline, PrepPipeline
from .dataset import Watcher
from .utils import *
from .transformers import *


class PipelineManager(object):
    def __init__(self, language, dataset_name, file_name, hyper_search=False, n=20, seed=42):
        self.seed = seed
        self.hyper_search = hyper_search
        self.N = n
        self.language = language
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.root = '/home/mks/projects/intent_classification_script/'
        self.file_path = join(self.root, 'data', self.language, self.dataset_name, 'data', self.file_name)
        self.date = datetime.datetime.now()
        pipegen = PipelineGenerator()
        self.pipeline_generator = pipegen.pipeline_gen()
        self.start_dataset = None

    def init_dataset(self):
        pure_data = read_dataset(self.file_path, True, True)  # It not default meanings!!!
        self.start_dataset = Watcher(pure_data, self.date, self.language, self.dataset_name,
                                     seed=self.seed)  # classes_descriptions = {} we can do it

        return self

    def init_dataset_tiny(self):
        pure_data = read_dataset(self.file_path, True, True)  # It not default meanings!!!
        self.start_dataset = Watcher(pure_data, self.date, self.language, self.dataset_name,
                                     seed=self.seed)  # classes_descriptions = {} we can do it

        ######################################################################################
        dataset = self.start_dataset.split([0.1, 0.1])
        data = dataset.data['test']
        self.start_dataset = Watcher(data, self.date, self.language, self.dataset_name, seed=self.seed)
        ######################################################################################

        return self

    def run(self):
        # Start generating pipelines configs
        for x in self.pipeline_generator:
            prer_pipe = x[0][:-2]
            model_pipe = x[0][-2:]
            pipe_conf = x[1]

            prer_pipeline = PrepPipeline(prer_pipe, mode='infer', output='dataset')

            # initialize new dataset
            self.init_dataset_tiny()
            d_ = prer_pipeline.run(self.start_dataset)

            if not self.hyper_search:
                model_pipeline = Pipeline(model_pipe, mode='infer', output='dataset')
                end_dataset = model_pipeline.run(d_)
            else:
                model_name = list(pipe_conf.keys())[-1].split('_')[0] + '_params.json'
                model_conf = pipe_conf[list(pipe_conf.keys())[-1]]
                path_to_model_conf = join(self.root, 'configs', 'models', model_name)

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
                    model_pipe[0][1] = params
                    model_pipeline = Pipeline(model_pipe, mode='infer', output='dataset')
                    end_dataset = model_pipeline.run(d_)

        results_summarization(self.date, self.language, self.dataset_name)

        return None
