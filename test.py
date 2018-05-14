from deepburtsev.core.pipegen import PipelineGenerator
from deepburtsev.core.transformers import FasttextVectorizer, ResultsCollector
from deepburtsev.models.intent_classification.WCNN import WCNN
import json
from os.path import join


# data prepare
root = '/home/mks/projects/DeepBurtsev/'
file_path = join(root, 'data', 'english', 'new_group', 'dataset.json')
with open(file_path, 'r') as f:
    dataset = json.load(f)
    f.close()

# create structure for pipeline manager
neural_struct = [(FasttextVectorizer, {'dimension': 100,
                                       'model_path': 'wordpunct_tok_reddit_comments_2017_11_100.bin'}),
                 WCNN, ResultsCollector]

pipeline_generator = PipelineGenerator(neural_struct, n=10, dtype='list')
gen = pipeline_generator.generator

for i, pipe in enumerate(gen):
    print('fuck', i)
