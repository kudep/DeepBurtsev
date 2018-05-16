import json
from os.path import join

from deepburtsev.core.pipelinemanager import PipelineManager
from deepburtsev.core.transformers import FasttextVectorizer, ResultsCollector
from deepburtsev.models.intent_classification.WCNN import WCNN
from deepburtsev.models.skmodels.linear_models import LinearRegression
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

neural_struct = [fasttext, (WCNN, {'batch_size': 32}), ResultsCollector]

linear_struct = [[tfidf, count], LinearRegression, ResultsCollector]

# neural_man = PipelineManager(dataset, neural_struct, 'skill_manager')
# neural_man.run()

linear_man = PipelineManager(dataset, linear_struct, 'skill_manager')
linear_man.run()
