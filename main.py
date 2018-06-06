import json
from os.path import join

from deepburtsev.core.pipelinemanager import PipelineManager
from deepburtsev.core.sktransformers import Count
from deepburtsev.core.sktransformers import Tfidf
from deepburtsev.core.transformers import FasttextVectorizer, ResultsCollector, Tokenizer
from deepburtsev.models.intent_classificators import WCNN, DCNN
from deepburtsev.models.skmodels import LinearRegression, LinearSVM, RandomForest

# data prepare
root = '/home/mks/projects/DeepBurtsev/'
file_path = join(root, 'data', 'russian', 'sber_faq', 'sber_faq.json')
with open(file_path, 'r') as f:
    dataset = json.load(f)
    f.close()

# create structure for pipeline manager
fasttext = FasttextVectorizer(dimension=300,
                              model_path='./embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')

tfidf = Tfidf()
count = Count()

neural_struct = [Tokenizer, fasttext,
                 [(WCNN(), {'search': True, 'batch_size': [32, 64], 'epochs': 5, 'val_every_n_epochs': 2}),
                  (DCNN(), {"search": True, 'batch_size': [32, 64], 'epochs': 5, 'op_name': 'DCNN'})],
                 ResultsCollector]

linear_struct = [[tfidf, count],
                 [(LinearRegression, {"search": True, "max_iter": [100, 150, 200]}),
                  RandomForest,
                  (LinearSVM, {"search": True, "loss": ["squared_hinge", "hinge"]})],
                 ResultsCollector(metrics=['accuracy', 'f1_macro', 'f1_weighted'])]

neural_man = PipelineManager(dataset, neural_struct, 'test', target_metric='f1_macro', hyper_search='grid',
                             add_watcher=False)
neural_man.run()

linear_man = PipelineManager(dataset, linear_struct, 'test', target_metric='f1_macro', hyper_search='grid',
                             add_watcher=False)
linear_man.run()  # skill_manager
