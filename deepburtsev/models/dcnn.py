# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from os.path import isdir, join
import os

import keras.metrics
import keras.optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from deepburtsev.core import metrics as metrics_file
from deepburtsev.core.utils import log_metrics


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))


class DCNN(object):
    """
    Class builds keras intent_model
    """

    def __init__(self, opt, *args, **kwargs):
        """
        Method initializes intent_model using parameters from opt
        Args:
            opt: dictionary of parameters
            *args:
            **kwargs:
        """
        self.opt = copy.deepcopy(opt)

        self.model_path_ = self.opt["checkpoint_path"]
        self.confident_threshold = self.opt['confident_threshold']

        if self.opt['model_from_saved']:
            self.model = self.restore(model_name=self.opt['model_name'],
                                      fname=self.model_path_,
                                      optimizer_name=self.opt['optimizer'],
                                      lr=self.opt['lear_rate'],
                                      decay=self.opt['lear_rate_decay'],
                                      loss_name=self.opt['loss'],
                                      metrics_names=self.opt['lear_metrics'],
                                      add_metrics_file=metrics_file)
        else:
            self.classes = np.array(self.opt['classes'].split(' '))
            self.n_classes = self.classes.shape[0]
            self.model = self.init_model_from_scratch(model_name=self.opt['model_name'],
                                                      optimizer_name=self.opt['optimizer'],
                                                      lr=self.opt['lear_rate'],
                                                      decay=self.opt['lear_rate_decay'],
                                                      loss_name=self.opt['loss'],
                                                      metrics_names=self.opt['lear_metrics'],
                                                      add_metrics_file=metrics_file)

        self.metrics_names = self.model.metrics_names
        self.metrics_values = len(self.metrics_names) * [0.]

    def model(self, params):
        """
        Method builds uncompiled intent_model of deep CNN
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled intent_model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                            self.opt['kernel_sizes_cnn'].split(' ')]

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = [int(x) for x in
                                       self.opt['filters_cnn'].split(' ')]

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = inp

        for i in range(len(params['kernel_sizes_cnn'])):
            output = Conv1D(params['filters_cnn'][i], kernel_size=params['kernel_sizes_cnn'][i],
                            activation=None,
                            kernel_regularizer=l2(params['coef_reg_cnn']),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation(params['last_activation'])(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def init_model_from_scratch(self, model_name, optimizer_name,
                                lr, decay, loss_name, metrics_names=None, add_metrics_file=None,
                                loss_weights=None,
                                sample_weight_mode=None,
                                weighted_metrics=None,
                                target_tensors=None):
        """
        Method initializes intent_model from scratch with given params
        Args:
            model_name: name of intent_model function described as a method of this class
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)
            metrics_names: names of metrics (from keras.metrics) as one string
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.intent_model.compile
            sample_weight_mode: optional parameter as in keras.intent_model.compile
            weighted_metrics: optional parameter as in keras.intent_model.compile
            target_tensors: optional parameter as in keras.intent_model.compile

        Returns:
            compiled intent_model with given network and learning parameters
        """
        print('[ Initializing intent_model from scratch ]')
        model = self.model(params=self.opt)
        optimizer_func = getattr(keras.optimizers, optimizer_name, None)

        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(add_metrics_file, metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
        return model

    def restore(self, model_name, fname, optimizer_name,
                lr, decay, loss_name, metrics_names=None, add_metrics_file=None, loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        """
        Method initiliazes intent_model from saved params and weights
        Args:
            model_name: name of intent_model function described as a method of this class
            fname: path and first part of name of intent_model
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)
            metrics_names: names of metrics (from keras.metrics) as one string
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.intent_model.compile
            sample_weight_mode: optional parameter as in keras.intent_model.compile
            weighted_metrics: optional parameter as in keras.intent_model.compile
            target_tensors: optional parameter as in keras.intent_model.compile

        Returns:
            intent_model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        print('___Initializing intent_model from saved___'
              '\nModel weights file is %s.h5'
              '\nNetwork parameters are from %s_opt.json' % (fname, fname))

        weights_fname = model_name + '.h5'
        opt_fname = model_name + '_opt.json'

        opt_path = join(fname, opt_fname)
        weights_path = join(fname, weights_fname)

        if Path(opt_path).is_file():
            with open(opt_path, 'r') as opt_file:
                self.opt = json.load(opt_file)
        else:
            raise IOError("Error: config file %s_opt.json of saved intent_model does not exist" % fname)

        self.classes = np.array(self.opt['classes'].split(" "))
        self.n_classes = self.classes.shape[0]

        model = self.model(params=self.opt)

        print("Loading wights from `{}`".format(fname + self.opt['model_name'] + '.h5'))
        model.load_weights(weights_path)

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(add_metrics_file, metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
        return model

    def save(self, fname=None):
        """
        Method saves the intent_model parameters into <<fname>>_opt.json (or <<model_file>>_opt.json)
        and intent_model weights into <<fname>>.h5 (or <<model_file>>.h5)
        Args:
            fname: file_path to save intent_model. If not explicitly given seld.opt["model_file"] will be used

        Returns:
            nothing
        """
        fname = self.model_path_ if fname is None else fname
        opt_path = join(fname, self.opt['model_name'] + '_opt.json')
        weights_path = join(fname, self.opt['model_name'] + '.h5')

        if isdir(fname):
            pass
        else:
            os.makedirs(fname)

        # opt_path = Path.joinpath(self.model_path_, opt_fname)
        # weights_path = Path.joinpath(self.model_path_, weights_fname)
        # print("[ saving intent_model: {} ]".format(str(opt_path)))
        self.model.save_weights(weights_path)

        with open(opt_path, 'w') as outfile:
            json.dump(self.opt, outfile)

        return str(weights_path)

    def reset(self):
        tf.reset_default_graph()
        return self

    def train_on_batch(self, batch):
        """
        Method pipelines the intent_model on the given batch
        Args:
            batch - list of tuples (preprocessed text, labels)

        Returns:
            loss and metrics values on the given batch
        """
        features = batch[0]
        onehot_labels = batch[1]
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    def infer_on_batch(self, batch, labels=None):
        """
        Method infers the model on the given batch
        Args:
            batch - list of texts

        Returns:
            loss and metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        if labels:
            features = batch
            onehot_labels = labels
            metrics_values = self.model.test_on_batch(features, onehot_labels.reshape(-1, self.n_classes))
            return metrics_values
        else:
            predictions = self.model.predict(batch)
            return predictions

    def batch_reformat(self, batch):
        vectors = list(batch[0])
        labels_vec = np.array(batch[1])
        if len(labels_vec) != self.opt['batch_size']:
            shape = (self.opt['batch_size'] - len(labels_vec), len(labels_vec[0]))
            labels_vec = np.concatenate((labels_vec, np.zeros(shape)), axis=0)

        matrix = np.zeros((self.opt['batch_size'], self.opt['text_size'], self.opt['embedding_size']))
        for i, x in enumerate(vectors):
            for j, y in enumerate(x):
                if j < self.opt['text_size']:
                    for k, d in enumerate(y):
                            matrix[i][j][k] = d

        batch = (matrix, labels_vec)

        return batch

    def train(self, dataset, name, *args, **kwargs):
        """
        Method pipelines the intent_model using batches and validation
        Args:
            dataset: instance of class Dataset

        Returns: None

        """
        updates = 0

        val_loss = 1e100
        val_increase = 0
        epochs_done = 0

        n_train_samples = len(dataset.data[name])
        print('\n____Training over {} samples____\n\n'.format(n_train_samples))

        while epochs_done < self.opt['epochs']:
            batch_gen = dataset.iter_batch(batch_size=self.opt['batch_size'],
                                           data_type=name)

            for step, batch in enumerate(batch_gen):

                ##############################################
                batch = self.batch_reformat(batch)
                ##############################################

                metrics_values = self.train_on_batch(batch)
                updates += 1

                if self.opt['verbose'] and step % 500 == 0:
                    log_metrics(names=self.metrics_names,
                                values=metrics_values,
                                updates=updates,
                                mode='train')

            epochs_done += 1
            if epochs_done % self.opt['val_every_n_epochs'] == 0:
                if 'valid' in dataset.data.keys():

                    valid_batch_gen = dataset.iter_batch(batch_size=self.opt['batch_size'],
                                                         data_type='valid')
                    valid_metrics_values = []
                    for valid_step, valid_batch in enumerate(valid_batch_gen):
                        ##############################################
                        valid_batch = self.batch_reformat(valid_batch)
                        ##############################################

                        valid_metrics_values.append(self.infer_on_batch(valid_batch[0],
                                                                        labels=valid_batch[1]))

                    valid_metrics_values = np.mean(np.asarray(valid_metrics_values), axis=0)
                    log_metrics(names=self.metrics_names,
                                values=valid_metrics_values,
                                mode='valid')
                    if valid_metrics_values[0] > val_loss:
                        val_increase += 1
                        print("__Validation impatience {} out of {}".format(
                            val_increase, self.opt['val_patience']))
                        if val_increase == self.opt['val_patience']:
                            print("___Stop training: validation is out of patience___")
                            break
                    else:
                        val_increase = 0
                        val_loss = valid_metrics_values[0]
            print('epochs_done: {}'.format(epochs_done))

        self.save()

    def predict(self, dataset, name, *args):
        """
        Method returns predictions on the given data
        Args:
            data: sentence or list of sentences
            *args:

        Returns:
            Predictions for the given data
        """
        request, report = dataset.main_names
        data = dataset.data[name][request]

        if type(data) is str:
            preds = self.infer_on_batch([data])[0]
            preds = np.array(preds)
        elif (type(data) is list) or isinstance(data, pd.Series):
            if len(data) > self.opt['batch_size']:
                batch_gen = dataset.iter_batch(batch_size=self.opt['batch_size'],
                                               data_type=name)
                predictions = []
                for batch in batch_gen:
                    ##############################################
                    pred_batch = self.batch_reformat(batch)
                    ##############################################
                    preds = self.infer_on_batch(pred_batch[0])
                    preds = np.array(preds)
                    predictions.append(preds)

                preds = predictions
            else:
                preds = self.infer_on_batch(data)
                preds = np.array(preds)
        else:
            raise ValueError("Not understand data type for inference")
        return preds
