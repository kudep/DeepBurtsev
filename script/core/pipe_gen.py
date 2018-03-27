from collections import OrderedDict
from script.core.utils import ConfGen
from script.core.transformers import *
from script.core.models import *
from script.models.cnn import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import json


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
                         'TextConcatenator': TextConcat,
                         'FasttextVectorizer': FasttextVectorizer,
                         'tf-idf': TfidfVectorizer,
                         'count': CountVectorizer,
                         'CNN': CNN,
                         'RandomForestClassifier': RandomForestClassifier,
                         'LinearSVC': LinearSVC,
                         'LogisticRegression': LogisticRegression,
                         'LGBMClassifier': LGBMClassifier}

        self.neural_struct = {'Speller': {'bool': True}, 'Lemmatizer': {'bool': True}, 'model': ['CNN']}
        self.neural_pipe = OrderedDict(Speller=True,
                                       Tokenizer=True,
                                       Lemmatizer=True,
                                       vectorizer='FasttextVectorizer',
                                       model='CNN')
        self.neural = (self.neural_pipe, self.neural_struct)

        self.linear_struct = {'Speller': {'bool': True}, 'Lemmatizer': {'bool': True},
                              'vectorizer': ['tf-idf', 'count'],
                              'model': ['LogisticRegression',
                                        'RandomForestClassifier',
                                        'LGBMClassifier',
                                        'LinearSVC']}
        self.linear_pipe = OrderedDict(Speller=True,
                                       Tokenizer=True,
                                       Lemmatizer=True,
                                       TextConcatenator=True,
                                       vectorizer='tf-idf',
                                       model='LogisticRegression')
        self.linear = (self.linear_pipe, self.linear_struct)

    def pipeline_gen(self, model_type):
        # generation
        if model_type == 'neural':
            pipe_conf = ConfGen(self.neural[0], self.neural[1]).sample_params()
            resulter = GetResult
        elif model_type == 'linear':
            pipe_conf = ConfGen(self.linear[0], self.linear[1]).sample_params()
            pipe_conf['Tokenizer'] = pipe_conf['Lemmatizer']
            pipe_conf['TextConcatenator'] = pipe_conf['Lemmatizer']
            resulter = GetResultLinear
        else:
            raise ValueError('')

        pipeline_config = OrderedDict()
        pipe = []
        for key in pipe_conf.keys():
            if isinstance(pipe_conf[key], bool):
                path = './configs/ops/' + key + '.json'
                conf = get_config(path)
                pipeline_config[str(key) + '_transformer'] = conf
                pipe.append((self.ops_dict[key], conf))
            elif isinstance(pipe_conf[key], str):
                if key == 'vectorizer':
                    if pipe_conf[key] == 'FasttextVectorizer':
                        conf = get_config('./configs/ops/FasttextVectorizer.json')
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
                        path = './configs/models/Linear/' + pipe_conf[key] + '.json'
                        conf = get_config(path)
                        model = skmodel(self.ops_dict[pipe_conf[key]], conf)
                        pipeline_config[pipe_conf[key] + '_model'] = conf
                        pipe.append((model,))
                    elif pipe_conf[key] == 'CNN':
                        path = './configs/models/CNN/WCNN.json'
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
