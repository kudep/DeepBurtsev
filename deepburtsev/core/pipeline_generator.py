from collections import OrderedDict
from itertools import product

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from deepburtsev.core.models.BaseModel import *
from deepburtsev.core.transformers.transformers import *
from deepburtsev.core.vectorizers.vectorizers import *
from deepburtsev.core.utils import ConfGen
from deepburtsev.core.wrappers.skwrappers import *
from deepburtsev.models.cnn import *


class GetCNN(BaseModel):
    def init_model(self, dataset):
        classes = dataset.get_classes()
        classes = ' '.join([str(x) for x in classes])
        self.config['classes'] = classes

        super().init_model(dataset)

        return self


def get_config(path):
    with open(path, 'r') as conf:
        config = json.load(conf)
        conf.close()
    return config


class PipeGen(object):
    def __init__(self):
        self.vec_default = {'op_type': 'vectorizer', 'name': 'tf-idf vectorizer',
                            'request_names': ['train', 'valid', 'test'],
                            'new_names': ['train_vec', 'valid_vec', 'test_vec']}

        # Concatenator
        self.ops_dict = {'Speller': Speller,
                         'Tokenizer': Tokenizer,
                         'Lemmatizer': Lemmatizer,
                         'Textconcatenator': TextConcat,
                         'FasttextVectorizer': FasttextVectorizer,
                         'tf-idf': TfidfVectorizer,
                         'count': CountVectorizer,
                         'CNN': CNN,
                         'RandomForestClassifier': RandomForestClassifier,
                         'LinearSVC': LinearSVC,
                         'LogisticRegression': LogisticRegression,
                         'LGBMClassifier': LGBMClassifier}

        # self.neural_struct = {'Speller': {'bool': True}, 'Lemmatizer': {'bool': True}, 'model': ['CNN']}

        ################################################################################################
        self.neural_struct = {'Lemmatizer': {'bool': True}, 'model': ['CNN']}
        ################################################################################################

        self.neural_pipe = OrderedDict(Speller=True,
                                       Tokenizer=True,
                                       Lemmatizer=True,
                                       vectorizer='FasttextVectorizer',
                                       model='CNN')
        self.neural = (self.neural_pipe, self.neural_struct)

        # self.linear_struct = {'Speller': {'bool': True}, 'Lemmatizer': {'bool': True},
        #                       'vectorizer': ['tf-idf', 'count'],
        #                       'model': ['LogisticRegression',
        #                                 'RandomForestClassifier',
        #                                 'LGBMClassifier',
        #                                 'LinearSVC']}

        ################################################################################################
        self.linear_struct = {'Lemmatizer': {'bool': True},
                              'vectorizer': ['tf-idf', 'count'],
                              'model': ['LogisticRegression',
                                        'RandomForestClassifier',
                                        'LGBMClassifier',
                                        'LinearSVC']}
        ################################################################################################

        self.linear_pipe = OrderedDict(Speller=True,
                                       Tokenizer=True,
                                       Lemmatizer=True,
                                       Textconcatenator=True,
                                       vectorizer='tf-idf',
                                       model='LogisticRegression')
        self.linear = (self.linear_pipe, self.linear_struct)

    def pipeline_gen(self, model_type):
        # generation
        resulter = GetResult
        if model_type == 'neural':
            pipe_conf = ConfGen(self.neural[0], self.neural[1]).sample_params()
        elif model_type == 'linear':
            pipe_conf = ConfGen(self.linear[0], self.linear[1]).sample_params()
            pipe_conf['Tokenizer'] = pipe_conf['Lemmatizer']
            pipe_conf['Textconcatenator'] = pipe_conf['Lemmatizer']
        else:
            raise ValueError('')

        pipeline_config = OrderedDict()
        pipe = []
        for key in pipe_conf.keys():
            if isinstance(pipe_conf[key], bool):
                path = './deepburtsev/configs/ops/' + key + '.json'
                conf = get_config(path)
                pipeline_config[str(key) + '_transformer'] = conf
                pipe.append((self.ops_dict[key], conf))
            elif isinstance(pipe_conf[key], str):
                if key == 'vectorizer':
                    if pipe_conf[key] == 'FasttextVectorizer':
                        conf = get_config('./deepburtsev/configs/ops/FasttextVectorizer.json')
                        pipeline_config['FasttextVectorizer_vectorizer'] = conf
                        pipe.append((self.ops_dict[pipe_conf[key]], conf))
                    elif pipe_conf[key] == 'tf-idf':
                        conf = self.vec_default
                        conf['name'] = 'tf-idf'
                        pipeline_config['tf-idf_vectorizer'] = conf
                        vec = sktransformer(self.ops_dict[pipe_conf[key]], conf)
                        pipe.append((vec,))
                    elif pipe_conf[key] == 'count':
                        conf = self.vec_default
                        conf['name'] = 'count'
                        pipeline_config['count_vectorizer'] = conf
                        vec = sktransformer(self.ops_dict[pipe_conf[key]], conf)
                        pipe.append((vec,))
                    else:
                        raise AttributeError('Vectorizer {} is not implemented yet.'.format(pipe_conf[key]))
                elif key == 'model':
                    if pipe_conf[key] in ['LogisticRegression', 'LGBMClassifier',
                                          'RandomForestClassifier', 'LinearSVC']:
                        path = './deepburtsev/configs/models/' + pipe_conf[key] + '.json'
                        conf = get_config(path)
                        model = skmodel(self.ops_dict[pipe_conf[key]], conf)
                        pipeline_config[pipe_conf[key] + '_model'] = conf
                        pipe.append((model,))
                    elif pipe_conf[key] == 'CNN':
                        path = './deepburtsev/configs/models/CNN.json'
                        conf = get_config(path)
                        WCNN = GetCNN(self.ops_dict[pipe_conf[key]], conf)
                        pipeline_config['WCNN_model'] = conf
                        pipe.append((WCNN,))
                    else:
                        raise ValueError('Model {} is not implemented yet.'.format(pipe_conf[key]))
                else:
                    raise ValueError('Unexpected key value {}'.format(key))
            else:
                raise TypeError('It wrong dict, attribute of dicts must have a bool type or str,'
                                'but {} was found.'.format(type(pipe_conf[key])))
        pipe.append((resulter,))

        return pipe, pipeline_config

    def sample_config(self):
        model_type = np.random.choice(['neural', 'linear'])
        pipe = self.pipeline_gen(model_type=model_type)
        return pipe


def genc(pipe, var):
    keys_values = []
    keys = list(var.keys())
    for key in var.keys():
        if key not in pipe.keys():
            raise KeyError('Key "{}" not in  pipe.'.format(key))
        else:
            keys_values.append(var[key])

    params_gen = product(*keys_values)
    for params in params_gen:
        for i, x in enumerate(params):
            pipe[keys[i]] = x
        yield pipe


class PipelineGeneratorOld(object):
    def __init__(self):
        self.vec_default = {'op_type': 'vectorizer', 'name': 'tf-idf vectorizer',
                            'request_names': ['train', 'valid', 'test'],
                            'new_names': ['train_vec', 'valid_vec', 'test_vec']}

        # Concatenator
        self.ops_dict = {'Speller': Speller,
                         'Tokenizer': Tokenizer,
                         'Lemmatizer': Lemmatizer,
                         'Textсoncatenator': TextConcat,
                         'FasttextVectorizer': FasttextVectorizer,
                         'tf-idf': TfidfVectorizer,
                         'count': CountVectorizer,
                         'CNN': CNN,
                         'RandomForestClassifier': RandomForestClassifier,
                         'LinearSVC': LinearSVC,
                         'LogisticRegression': LogisticRegression,
                         'LGBMClassifier': LGBMClassifier}

        # self.neural_struct = {'Speller': [False, True], 'Lemmatizer': [False, True], 'model': ['CNN']}
        # self.neural_pipe = OrderedDict(Speller=True,
        #                                Tokenizer=True,
        #                                Lemmatizer=True,
        #                                vectorizer='FasttextVectorizer',
        #                                model='CNN')

        ###############################################################################################
        self.neural_struct = {'Lemmatizer': [False, True], 'model': ['CNN']}
        self.neural_pipe = OrderedDict(Tokenizer=True,
                                       Lemmatizer=True,
                                       vectorizer='FasttextVectorizer',
                                       model='CNN',
                                       Resulter='Resulter')
        ################################################################################################

        # self.linear_struct = {'Speller': [False, True], 'Lemmatizer': [False, True],
        #                       'vectorizer': ['tf-idf', 'count'],
        #                       'model': ['LogisticRegression',
        #                                 'RandomForestClassifier',
        #                                 'LGBMClassifier',
        #                                 'LinearSVC']}
        # self.linear_pipe = OrderedDict(Speller=True,
        #                                Tokenizer=True,
        #                                Lemmatizer=True,
        #                                Textconcatenator=True,
        #                                vectorizer='tf-idf',
        #                                model='LogisticRegression')

        ###############################################################################################
        self.linear_struct = {'Lemmatizer': [False, True],
                              'vectorizer': ['tf-idf', 'count'],
                              'model': ['LogisticRegression',
                                        'RandomForestClassifier',
                                        # 'LGBMClassifier',
                                        'LinearSVC']}
        self.linear_pipe = OrderedDict(Tokenizer=True,
                                       Lemmatizer=True,
                                       Textсoncatenator=True,
                                       vectorizer='tf-idf',
                                       model='LogisticRegression',
                                       Resulter='Resulter')
        ###############################################################################################

    # generation
    def pipeline_gen(self):
        model_types = ['neural', 'linear']
        resulter = GetResult

        for type_ in model_types:
            if type_ == 'neural':
                gen = genc(self.neural_pipe, self.neural_struct)
            elif type_ == 'linear':
                gen = genc(self.linear_pipe, self.linear_struct)
            else:
                raise ValueError('')

            for conf in gen:
                if type_ == 'linear':
                    conf['Tokenizer'] = conf['Lemmatizer']
                    conf['Textсoncatenator'] = conf['Lemmatizer']

                pipeline_config = OrderedDict()
                pipe = []
                for key in conf.keys():
                    if isinstance(conf[key], bool):
                        path = './deepburtsev/configs/ops/' + key + '.json'
                        config = get_config(path)
                        pipeline_config[str(key) + '_transformer'] = config
                        pipe.append((self.ops_dict[key], config))
                    elif isinstance(conf[key], str):
                        if key == 'vectorizer':
                            if conf[key] == 'FasttextVectorizer':
                                config = get_config('./deepburtsev/configs/ops/FasttextVectorizer.json')
                                pipeline_config['FasttextVectorizer_vectorizer'] = config
                                pipe.append((self.ops_dict[conf[key]], config))
                            elif conf[key] == 'tf-idf':
                                config = self.vec_default
                                config['name'] = 'tf-idf'
                                pipeline_config['tf-idf_vectorizer'] = config
                                vec = sktransformer(self.ops_dict[conf[key]], config)
                                pipe.append((vec,))
                            elif conf[key] == 'count':
                                config = self.vec_default
                                config['name'] = 'count'
                                pipeline_config['count_vectorizer'] = config
                                vec = sktransformer(self.ops_dict[conf[key]], config)
                                pipe.append((vec,))
                            else:
                                raise AttributeError('Vectorizer {} is not implemented yet.'.format(conf[key]))
                        elif key == 'model':
                            if conf[key] in ['LogisticRegression', 'LGBMClassifier',
                                             'RandomForestClassifier', 'LinearSVC']:
                                path = './deepburtsev/configs/models/' + conf[key] + '.json'
                                config = get_config(path)
                                model = skmodel(self.ops_dict[conf[key]], config)
                                pipeline_config[conf[key] + '_model'] = config
                                pipe.append((model,))
                            elif conf[key] == 'CNN':
                                path = './deepburtsev/configs/models/CNN.json'
                                config = get_config(path)
                                WCNN = GetCNN(self.ops_dict[conf[key]], config)
                                pipeline_config['WCNN_model'] = config
                                pipe.append((WCNN,))
                            else:
                                raise ValueError('Model {} is not implemented yet.'.format(conf[key]))
                        elif key == 'Resulter':
                            path = './deepburtsev/configs/ops/'+key+'.json'
                            config = get_config(path)
                            pipeline_config['Resulter_transformer'] = config
                            pipe.append((resulter, config))

                        else:
                            raise ValueError('Unexpected key value {}'.format(key))
                    else:
                        raise TypeError('It wrong dict, attribute of dicts must have a bool type or str,'
                                        'but {} was found.'.format(type(conf[key])))
                yield (pipe, pipeline_config)


class PipelineGenerator(object):
    def __init__(self, pipe, structure, root, dataset_name, emb_name, emb_dim, res_type, vec_default=None,
                 ops_dict=None):
        self.pipe = pipe
        self.structure = structure
        self.res_type = res_type
        self.root = root
        self.dataset_name = dataset_name
        self.emb_name = emb_name
        self.emb_dim = emb_dim

        if vec_default is None:
            self.vec_default = {'op_type': 'vectorizer', 'name': 'tf-idf vectorizer',
                                'request_names': ['train', 'valid', 'test'],
                                'new_names': ['train_vec', 'valid_vec', 'test_vec']}
        else:
            self.vec_default = vec_default

        if ops_dict is None:
            self.ops_dict = {'Speller': Speller,
                             'Tokenizer': Tokenizer,
                             'Lemmatizer': Lemmatizer,
                             'Textсoncatenator': TextConcat,
                             'FasttextVectorizer': FasttextVectorizer,
                             'tf-idf': TfidfVectorizer,
                             'count': CountVectorizer,
                             'CNN': CNN,
                             'RandomForestClassifier': RandomForestClassifier,
                             'LinearSVC': LinearSVC,
                             'LogisticRegression': LogisticRegression,
                             'LGBMClassifier': LGBMClassifier}
        else:
            self.ops_dict = ops_dict

    # generation
    def pipeline_gen(self):
        gen = genc(self.pipe, self.structure)
        resulter = GetResult

        for conf in gen:
            if self.res_type == 'linear':
                conf['Tokenizer'] = conf['Lemmatizer']
                conf['Textсoncatenator'] = conf['Lemmatizer']
                if conf['model'] == 'LGBMClassifier' and conf['vectorizer'] == 'count':
                    continue

            pipeline_config = OrderedDict()
            pipe = []
            for key in conf.keys():
                if isinstance(conf[key], bool):
                    if conf[key]:
                        path = './deepburtsev/configs/ops/' + key + '.json'
                        config = get_config(path)
                        pipeline_config[str(key) + '_transformer'] = config
                        pipe.append((self.ops_dict[key], config))
                    else:
                        pass
                elif isinstance(conf[key], str):
                    if key == 'vectorizer':
                        if conf[key] == 'FasttextVectorizer':
                            config = get_config('./deepburtsev/configs/ops/FasttextVectorizer.json')
                            config['path_to_model'] = join(self.root, 'embeddings', self.emb_name)
                            config['dimension'] = int(self.emb_dim)
                            pipeline_config['FasttextVectorizer_vectorizer'] = config
                            pipe.append((self.ops_dict[conf[key]], config))
                        elif conf[key] == 'tf-idf':
                            config = self.vec_default
                            config['name'] = 'tf-idf'
                            pipeline_config['tf-idf_vectorizer'] = config
                            vec = sktransformer(self.ops_dict[conf[key]], config)
                            pipe.append((vec,))
                        elif conf[key] == 'count':
                            config = self.vec_default
                            config['name'] = 'count'
                            pipeline_config['count_vectorizer'] = config
                            vec = sktransformer(self.ops_dict[conf[key]], config)
                            pipe.append((vec,))
                        else:
                            raise AttributeError('Vectorizer {} is not implemented yet.'.format(conf[key]))
                    elif key == 'model':
                        if conf[key] in ['LogisticRegression', 'LGBMClassifier',
                                         'RandomForestClassifier', 'LinearSVC']:
                            path = './deepburtsev/configs/models/' + conf[key] + '.json'
                            config = get_config(path)
                            model = skmodel(self.ops_dict[conf[key]], config)
                            pipeline_config[conf[key] + '_model'] = config
                            pipe.append((model,))
                        elif conf[key] == 'CNN':
                            path = './deepburtsev/configs/models/CNN.json'
                            config = get_config(path)

                            config['checkpoint_path'] = join(self.root, self.dataset_name, 'checkpoints', 'CNN')

                            WCNN = GetCNN(self.ops_dict[conf[key]], config)
                            pipeline_config['WCNN_model'] = config
                            pipe.append((WCNN,))
                        else:
                            raise ValueError('Model {} is not implemented yet.'.format(conf[key]))
                    elif key == 'Resulter':
                        path = './deepburtsev/configs/ops/'+key+'.json'
                        config = get_config(path)
                        pipeline_config['Resulter_transformer'] = config
                        pipe.append((resulter, config))

                    else:
                        raise ValueError('Unexpected key value {}'.format(key))
                else:
                    raise TypeError('It wrong dict, attribute of dicts must have a bool type or str,'
                                    'but {} was found.'.format(type(conf[key])))
            yield (pipe, pipeline_config)
