import fastText
import pandas as pd
import numpy as np

from tqdm import tqdm

from deepburtsev.core.utils import labels2onehot_one
from deepburtsev.core.transformers.transformers import BaseTransformer


# Old fasttext
# class FasttextVectorizer(BaseTransformer):
#     def __init__(self, config=None):
#
#         if config is None:
#             self.config = {'op_type': 'vectorizer',
#                            'name': 'fasttext',
#                            'request_names': ['train', 'valid', 'test'],
#                            'new_names': ['train_vec', 'valid_vec', 'test_vec'],
#                            'path_to_model': './data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin',
#                            'dimension': 300,
#                            'file_type': 'bin'}
#
#         else:
#             need_names = ['path_to_model', 'dimension', 'file_type']
#             for name in need_names:
#                 if name not in config.keys():
#                     raise ValueError('Input config must contain {}.'.format(name))
#
#             self.config = config
#
#         super().__init__(self.config)
#
#         self.vectorizer = fasttext.load_model(self.config['path_to_model'])
#
#     def _transform(self, dataset):
#         print('[ Starting vectorization ... ]')
#         request, report = dataset.main_names
#
#         for name, new_name in zip(self.request_names, self.new_names):
#             print('[ Vectorization of {} part of dataset ... ]'.format(name))
#             data = dataset.data[name][request]
#             vec_request = []
#
#             for x in tqdm(data):
#                 matrix_i = np.zeros((len(x), self.config['dimension']))
#                 for j, y in enumerate(x):
#                     matrix_i[j] = self.vectorizer[y]
#                 vec_request.append(matrix_i)
#
#             vec_report = list(labels2onehot_one(dataset.data[name][report], dataset.classes))
#
#             dataset.data[new_name] = pd.DataFrame({request: vec_request,
#                                                    report: vec_report})
#
#         print('[ Vectorization was ended. ]')
#         return dataset


class FasttextVectorizer(BaseTransformer):
    def __init__(self, request_names=None, new_names=None, op_type='vectorizer', op_name='fasttext',
                 dimension=300, file_type='bin', model_path='./data/russian/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin'):
        super().__init__(request_names, new_names, op_type, op_name)
        self.file_type = file_type
        self.dimension = dimension
        self.model_path = model_path
        self.vectorizer = fastText.load_model(self.model_path)

    def _transform(self, dataset, request_names=None, new_names=None):
        print('[ Starting vectorization ... ]')

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

        request, report = dataset.main_names

        for name, new_name in zip(self.request_names, self.new_names):
            print('[ Vectorization of {} part of dataset ... ]'.format(name))
            data = dataset.data[name][request]
            vec_request = []

            for x in tqdm(data):
                matrix_i = np.zeros((len(x), self.config['dimension']))
                for j, y in enumerate(x):
                    matrix_i[j] = self.vectorizer.get_word_vector(y)
                vec_request.append(matrix_i)

            vec_report = list(labels2onehot_one(dataset.data[name][report], dataset.classes))

            dataset.data[new_name] = pd.DataFrame({request: vec_request,
                                                   report: vec_report})

        print('[ Vectorization was ended. ]')
        return dataset
