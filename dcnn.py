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
from os.path import isdir, join
import os

import keras.metrics
import keras.optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import metrics as metrics_file
from utils import log_metrics, labels2onehot_one, get_result


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

        self.model_path_ = join(self.opt["checkpoint_path"], self.opt['model_name'] + '.h5')
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

    def dcnn_model(self, params):
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
        # print('[ Initializing intent_model from scratch ]')

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

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

        weights_fname = fname
        opt_fname = fname + '_opt.json'

        opt_path = Path.joinpath(self.model_path_, opt_fname)
        weights_path = Path.joinpath(self.model_path_, weights_fname)

        if Path(opt_path).is_file():
            with open(opt_path, 'r') as opt_file:
                self.opt = json.load(opt_file)
        else:
            raise IOError("Error: config file %s_opt.json of saved intent_model does not exist" % fname)

        self.classes = np.array(self.opt['classes'].split(" "))
        self.n_classes = self.classes.shape[0]

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        # print("Loading wights from `{}`".format(fname + '.h5'))
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
        fname = self.model_path_.name if fname is None else fname
        opt_fname = str(fname) + '_opt.json'
        weights_fname = str(fname) + '.h5'

        if isdir(fname):
            pass
        else:
            os.makedirs(fname)

        opt_path = Path.joinpath(self.model_path_, opt_fname)
        weights_path = Path.joinpath(self.model_path_, weights_fname)
        # print("[ saving intent_model: {} ]".format(str(opt_path)))
        self.model.save_weights(weights_path)

        with open(opt_path, 'w') as outfile:
            json.dump(self.opt, outfile)

        return str(weights_path)

    def reset(self):
        pass
