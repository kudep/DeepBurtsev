import sys
from collections import OrderedDict
from os.path import join

from DeepBurtsev.core.pipeline_manager import PipelineManager
from DeepBurtsev.datasets.dataset_readers import *

language = sys.argv[1]
dataset_name = sys.argv[2]
file_name = sys.argv[3]
emb_name = sys.argv[4]

neural_struct = {'Lemmatizer': [False, True], 'model': ['CNN']}
neural_pipe = OrderedDict(Tokenizer=True,
                          Lemmatizer=True,
                          vectorizer='FasttextVectorizer',
                          model='CNN',
                          Resulter='Resulter')

linear_struct = {'Lemmatizer': [False, True],
                 'vectorizer': ['tf-idf', 'count'],
                 'model': ['LogisticRegression']}
# 'model': ['LogisticRegression',
#           'RandomForestClassifier',
#           'LGBMClassifier',
#           'LinearSVC']}
linear_pipe = OrderedDict(Tokenizer=True,
                          Lemmatizer=True,
                          Textсoncatenator=True,
                          vectorizer='tf-idf',
                          model='LogisticRegression',
                          Resulter='Resulter')


root = '/home/mks/projects/DeepBurtsev/'
file_path = join(root, 'data', language, dataset_name, 'data', file_name)
pure_data = read_sber_dataset(file_path)

Manager = PipelineManager(language, dataset_name, emb_name, hyper_search=False)
# Manager.run(linear_pipe, linear_struct, 'linear', pure_data, test_mode=False)
Manager.run(neural_pipe, neural_struct, 'neural', pure_data)
