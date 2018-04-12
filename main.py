import sys
from collections import OrderedDict
from DeepBurtsev.core.pipeline_manager import PipelineManager


language = sys.argv[1]
dataset_name = sys.argv[2]
file_name = sys.argv[3]

# neural_struct = {'Speller': [False, True], 'Lemmatizer': [False, True], 'model': ['CNN']}
# neural_pipe = OrderedDict(Speller=True,
#                                Tokenizer=True,
#                                Lemmatizer=True,
#                                vectorizer='FasttextVectorizer',
#                                model='CNN')

###############################################################################################
neural_struct = {'Lemmatizer': [False, True], 'model': ['CNN']}
neural_pipe = OrderedDict(Tokenizer=True,
                          Lemmatizer=True,
                          vectorizer='FasttextVectorizer',
                          model='CNN',
                          Resulter='Resulter')
################################################################################################

# linear_struct = {'Speller': [False, True], 'Lemmatizer': [False, True],
#                       'vectorizer': ['tf-idf', 'count'],
#                       'model': ['LogisticRegression',
#                                 'RandomForestClassifier',
#                                 'LGBMClassifier',
#                                 'LinearSVC']}
# linear_pipe = OrderedDict(Speller=True,
#                                Tokenizer=True,
#                                Lemmatizer=True,
#                                Textconcatenator=True,
#                                vectorizer='tf-idf',
#                                model='LogisticRegression')

###############################################################################################
linear_struct = {'Lemmatizer': [False, True],
                 'vectorizer': ['tf-idf', 'count'],
                 'model': ['LinearSVC']}

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
###############################################################################################

Manager = PipelineManager(language, dataset_name, file_name, hyper_search=True)
# Manager.run(linear_pipe, linear_struct, 'linear')
Manager.run(neural_pipe, neural_struct, 'neural')


# if __name__ == '__main__':
#     language = 'russian'
#     dataset_name = 'vkusvill'
#     file_name = 'vkusvill_all_categories.csv'
#
#     Manager.run(linear_pipe, linear_struct, 'linear')
#     Manager.run(neural_pipe, neural_struct, 'neural')
