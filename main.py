import json
from os.path import join

from deepburtsev.core.pipelinemanager import PipelineManager
from deepburtsev.core.transformers import FasttextVectorizer, ResultsCollector
from deepburtsev.models.intent_classification.WCNN import WCNN
from deepburtsev.models.skmodels.linear_models import LinearRegression, LinearSVM, RandomForest
from deepburtsev.core.sktransformers import Tfidf
from deepburtsev.core.sktransformers import Count


# data prepare
root = '/home/mks/projects/DeepBurtsev/'
file_path = join(root, 'data', 'english', 'new_group', 'dataset.json')
with open(file_path, 'r') as f:
    dataset = json.load(f)
    f.close()

# create structure for pipeline manager
fasttext = FasttextVectorizer(request_names=['train', 'valid', 'test'],
                              new_names=['train', 'valid', 'test'],
                              dimension=100,
                              model_path='./embeddings/wordpunct_tok_reddit_comments_2017_11_100.bin')

tfidf = Tfidf(request_names=['train', 'valid', 'test'], new_names=['train', 'valid', 'test'])
count = Count(request_names=['train', 'valid', 'test'], new_names=['train', 'valid', 'test'])

neural_struct = [fasttext, (WCNN, {'search': True, 'batch_size': 32, 'epochs': [3, 5, 8, 10, 12, 14, 16, 18, 20]}),
                 ResultsCollector]

linear_struct = [[tfidf, count],
                 [LinearRegression, LinearSVM, RandomForest],
                 ResultsCollector]

# neural_man = PipelineManager(dataset, neural_struct, 'skill_manager', target_metric='f1_macro')
# neural_man.run()

linear_man = PipelineManager(dataset, linear_struct, 'skill_manager', target_metric='f1_macro')
linear_man.run()
