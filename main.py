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
file_path = join(root, 'data', 'russian', 'sber_faq', 'sber_faq.json')
with open(file_path, 'r') as f:
    dataset = json.load(f)
    f.close()

# create structure for pipeline manager
names = ['train', 'val', 'test']
fasttext = FasttextVectorizer(request_names=names,
                              new_names=names,
                              dimension=300,
                              model_path='./embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')

tfidf = Tfidf(request_names=names, new_names=names)
count = Count(request_names=names, new_names=names)

neural_struct = [fasttext,
                 (WCNN(new_names=["pred_test", "pred_val"], predict_names=["test", "val"]),
                  {'search': True, 'batch_size': [32, 64], 'epochs': 1}),
                 ResultsCollector]

# (DCNN(new_names=["pred_test", "pred_val"], predict_names=["test", "val"]),
#                    {'batch_size': 32, 'epochs': 1, 'op_name': 'DCNN'}),


linear_struct = [[tfidf, count],
                 [LinearRegression, LinearSVM, RandomForest],
                 ResultsCollector]

neural_man = PipelineManager(dataset, neural_struct, 'skill_manager', target_metric='f1_macro')
neural_man.run()

# linear_man = PipelineManager(dataset, linear_struct, 'skill_manager', target_metric='f1_macro')
# linear_man.run()
