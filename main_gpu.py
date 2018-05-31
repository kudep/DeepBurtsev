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
fasttext = FasttextVectorizer(request_names=['train', 'val', 'test'],
                              new_names=['train', 'val', 'test'],
                              dimension=300,
                              model_path='./embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')

tfidf = Tfidf(request_names=['train', 'val', 'test'], new_names=['train', 'val', 'test'])
count = Count(request_names=['train', 'val', 'test'], new_names=['train', 'val', 'test'])

neural_struct = [fasttext, (WCNN,
                            {"search": True,
                             "new_names": "pred_test",
                             "predict_names": "test",
                             "epochs": 1,  # [3, 5, 7, 10],
                             "kernel_sizes_cnn": {"range": [1, 5],
                                                  "discrete": True,
                                                  "n_samples": 3,
                                                  "increasing": True},
                             "filters_cnn": {"range": [32, 512],
                                             "discrete": True},
                             "confident_threshold": {"range": [0.4, 0.6]},
                             "lear_rate": {"range": [1e-2, 10],
                                           "scale": "log"},
                             "lear_rate_decay": {"range": [1e-1, 9e-1]},
                             "text_size": [25, 30, 40],
                             "coef_reg_cnn": {"range": [1e-5, 1e-3],
                                              "scale": "log"},
                             "coef_reg_den": {"range": [1e-5, 1e-3],
                                              "scale": "log"},
                             "dropout_rate": {"range": [1e-1, 7e-1]},
                             "dense_size": {"range": [32, 128],
                                            "discrete": True},
                             "batch_size": {"range": [32, 64, 128],
                                            "discrete": True}}),
                 ResultsCollector]

linear_svm = [[tfidf, count], (LinearSVM, {"search": True,
                                           "loss": ["squared_hinge", "hinge"],
                                           "tol": {"range": [0.0001, 0.1]},
                                           "C": {"range": [0.5, 2.0]},
                                           "fit_intercept": {"bool": True},
                                           "max_iter": [100, 150, 200],
                                           "multi_class": ["ovr", "crammer_singer"]}),

              ResultsCollector]

linear_rf = [[tfidf, count], (RandomForest, {"search": True,
                                             "n_estimators": [10, 12, 14, 16, 18, 20],
                                             "criterion": ["gini", "entropy"],
                                             "max_features": ["auto", "sqrt", "log2", None],
                                             "oob_score": {"bool": True}}),
             ResultsCollector]

linear_lr = [[tfidf, count], (LinearRegression, {"search": True,
                                                 "tol": {"range": [0.0001, 0.1]},
                                                 "C": {"range": [0.5, 2.0]},
                                                 "fit_intercept": {"bool": True},
                                                 "max_iter": [100, 150, 200],
                                                 "warm_start": {"bool": True}}),
             ResultsCollector]


neural_man = PipelineManager(dataset, neural_struct, 'sber_l', target_metric='f1_macro', hyper_search='random',
                             sample_num=1)
neural_man.run()

linear_sv = PipelineManager(dataset, linear_svm, 'sber_l', target_metric='f1_macro', hyper_search='random',
                            sample_num=1)
linear_sv.run()

linear_rfm = PipelineManager(dataset, linear_rf, 'sber_l', target_metric='f1_macro', hyper_search='random',
                             sample_num=1)
linear_rfm.run()

linear_man = PipelineManager(dataset, linear_lr, 'sber_l', target_metric='f1_macro', hyper_search='random',
                             sample_num=1)
linear_man.run()
