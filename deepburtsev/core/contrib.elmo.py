#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''
src:
https://github.com/allenai/bilm-tf

Installing:
    Install python version 3.5 or later, tensorflow version 1.2 and h5py:
        pip install tensorflow-gpu==1.2 h5py # Working with other versions of tensorflow-gpu, for example 1.4.
        python setup.py install
    Ensure the tests pass in your environment by running:
        python -m unittest discover tests/
'''

import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

from deepburtsev.core.transformers import BaseTransformer

class ELMOVectorizer(BaseTransformer):
    def __init__(self, request_names=['train', 'valid', 'test'], new_names=['train', 'valid', 'test'],
                 classes_name='classes', op_type='vectorizer', op_name='elmo', dimension=1024,
                 file_type='bin', #TODO: ?
                 options_file='./embeddingsruwiki_pp_1.0_elmo/options.json',  #TODO: ?
                 weights_file='./embeddingsruwiki_pp_1.0_elmo/weights.hdf5', #TODO: ?
                 vocab_file = './embeddingsruwiki_pp_1.0_elmo/vocab.txt' #TODO: ?
                 ):
        super().__init__(request_names, new_names, op_type, op_name)
        self.file_type = file_type
        self.classes_name = classes_name
        self.dimension = dimension
        # Location of pretrained LM.
        self.options_file = options_file
        self.weights_file = weights_file
        self.vocab_file = vocab_file
        # Create a Batcher to map text to character ids.
        char_per_token = 50
        self.batcher = Batcher(self.vocab_file, char_per_token)
        # Input placeholders to the biLM.
        self.character_ids = tf.placeholder('int32', shape=(None, None, char_per_token))
        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(self.options_file, self.weights_file)

        # Get ops to compute the LM embeddings.
        embeddings_op = bilm(character_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        self.elmo_output = weight_layers('elmo_output', embeddings_op, l2_coef=0.0)

    def _transform(self, dictionary, request_names=None, new_names=None):
        print('[ Starting vectorization ... ]')
        self._fill_names(request_names, new_names)
        # if dictionary.get(self.classes_name) is None:
        #     raise ValueError("The inbound dictionary should contain list of classes for one-hot"
        #                      " vectorization of y_true values.")
        with tf.Session() as sess:
            for name, new_name in zip(self.request_names, self.new_names):
                print('[ Vectorization of {} part of dataset ... ]'.format(name))
                data = dictionary[name]['x']
                # It is necessary to initialize variables once before running inference.
                sess.run(tf.global_variables_initializer())

                # Create batches of data.
                data_ids = batcher.batch_sentences(data)
                # Compute ELMo representations (here for the input only, for simplicity).
                elmo_output_,  = sess.run(
                    [self.elmo_output['weighted_op']],
                    feed_dict={self.character_ids: data_ids}
                )
                dictionary[new_name] = {'x': elmo_output_, 'y': dictionary[name]['y']}
        print('[ Vectorization was ended. ]')
        return dictionary
