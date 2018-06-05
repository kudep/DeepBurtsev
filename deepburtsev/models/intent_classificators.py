import json
import os
import random
from os.path import join, isdir
from pathlib import Path
from typing import Generator

import keras.metrics
import keras.optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Input, concatenate, Activation, Concatenate, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2

from deepburtsev.core import metrics as metrics_file
from deepburtsev.core.keras_layers import multiplicative_self_attention, additive_self_attention
from deepburtsev.core.utils import log_metrics, labels2onehot_one
from deepburtsev.wrappers.model_wrappers import BaseModel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

SEED = 42


class Dataset(object):

    def __init__(self, data, seed=None, classes_description=None, *args, **kwargs):

        self.main_names = ['x', 'y']
        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.data = data

        self.classes_description = classes_description

    def iter_batch(self, batch_size: int, data_type: str = 'base', shuffle: bool = True,
                   only_request: bool = False) -> Generator:
        """This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): shuffle trigger
            only_request (bool): trigger that told what data will be returned
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data['x'])
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        if shuffle:
            random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        # for i in range((data_len - 1) // batch_size + 1):
        #     yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))
        if not only_request:
            for i in range((data_len - 1) // batch_size + 1):
                # o = order[i * batch_size: (i + 1) * batch_size]
                # print(type(o))
                # print(o)

                yield list((list(data[self.main_names[0]][i * batch_size: (i + 1) * batch_size]),
                            list(data[self.main_names[1]][i * batch_size: (i + 1) * batch_size])))
        else:
            for i in range((data_len - 1) // batch_size + 1):
                o = order[i * batch_size:(i + 1) * batch_size]
                yield list((list(self.data[self.main_names[0]][o]),))

    def iter_all(self, data_type: str = 'base', only_request: bool = False) -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
            only_request (bool): trigger that told what data will be returned
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in zip(data[self.main_names[0]], data[self.main_names[1]]):
            if not only_request:
                yield (x, y)
            else:
                yield (x,)


class WCNN(BaseModel):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='WCNN',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 kernel_sizes_cnn="1 2 3",
                 filters_cnn=256,
                 embedding_size=300,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=25,
                 coef_reg_cnn=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure precision_K recall_K'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.kernel_sizes_cnn = kernel_sizes_cnn
        self.filters_cnn = filters_cnn
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_cnn = coef_reg_cnn
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    # model graph
    def cnn_model(self):
        """
        Method builds uncompiled intent_model of shallow-and-wide CNN
        Args:
        Returns:
            Uncompiled intent_model
        """
        if type(self.kernel_sizes_cnn) is str:
            self.kernel_sizes_cnn = [int(x) for x in self.kernel_sizes_cnn.split(' ')]

        inp = Input(shape=(self.text_size, self.embedding_size))

        outputs = []
        for i in range(len(self.kernel_sizes_cnn)):
            output_i = Conv1D(self.filters_cnn, kernel_size=self.kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(self.coef_reg_cnn),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def init_model_from_scratch(self, add_metrics_file=None, loss_weights=None, sample_weight_mode=None):
        """
        Method initializes intent_model from scratch with given params
        Args:
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.intent_model.compile
            sample_weight_mode: optional parameter as in keras.intent_model.compile

        Returns:
            compiled intent_model with given network and learning parameters
        """
        print('[ Initializing intent_model from scratch ]')
        model = self.model
        optimizer_func = getattr(keras.optimizers, self.optimizer, None)

        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=self.lear_rate, decay=self.lear_rate_decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(self.optimizer))

        loss_func = getattr(keras.losses, self.loss, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(self.loss))

        if self.metrics_names is None:
            metrics_funcs = getattr(keras.metrics, self.lear_metrics, None)
        else:
            if isinstance(self.metrics_names, str):
                self.metrics_names = self.metrics_names.split(' ')

            metrics_funcs = []
            for i in range(len(self.metrics_names)):
                metrics_func = getattr(keras.metrics, self.metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    metrics_func = getattr(add_metrics_file, self.metrics_names[i], None)
                    if callable(metrics_func):
                        metrics_funcs.append(metrics_func)
                    else:
                        raise AttributeError("Metric {} is not defined".format(self.metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode)
        return model

    def restore(self, add_metrics_file=None, loss_weights=None, sample_weight_mode=None):
        """
        Method initiliazes intent_model from saved params and weights
        Args:
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.intent_model.compile
            sample_weight_mode: optional parameter as in keras.intent_model.compile

        Returns:
            intent_model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        print('___Initializing intent_model from saved___'
              '\nModel weights file is %s.h5'
              '\nNetwork parameters are from %s_opt.json' % (self.checkpoint_path, self.checkpoint_path))

        weights_fname = self.model_name + '.h5'
        opt_fname = self.model_name + '_opt.json'

        opt_path = join(self.checkpoint_path, opt_fname)
        weights_path = join(self.checkpoint_path, weights_fname)

        if Path(opt_path).is_file():
            with open(opt_path, 'r') as opt_file:
                opt = json.load(opt_file)
        else:
            raise IOError("Error: config file %s_opt.json of saved intent_model does not exist" % self.checkpoint_path)

        self.set_params(**opt)
        model = self.cnn_model()

        print("Loading wights from `{}`".format(self.checkpoint_path + opt['model_name'] + '.h5'))
        model.load_weights(weights_path)

        optimizer_func = getattr(keras.optimizers, self.optimizer, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=self.lear_rate, decay=self.lear_rate_decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(self.optimizer))

        loss_func = getattr(keras.losses, self.loss, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(self.loss))

        self.metrics_names = self.metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(self.metrics_names)):
            metrics_func = getattr(keras.metrics, self.metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(add_metrics_file, self.metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(self.metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode)
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
        fname = self.checkpoint_path if fname is None else fname
        opt_path = join(fname, self.model_name + '_opt.json')
        weights_path = join(fname, self.model_name + '.h5')

        if isdir(fname):
            pass
        else:
            os.makedirs(fname)

        # opt_path = Path.joinpath(self.model_path_, opt_fname)
        # weights_path = Path.joinpath(self.model_path_, weights_fname)
        # print("[ saving intent_model: {} ]".format(str(opt_path)))
        self.model.save_weights(weights_path)

        opt = self.get_params()
        with open(opt_path, 'w') as outfile:
            json.dump(opt, outfile)

        return str(weights_path)

    def reset(self):
        tf.reset_default_graph()
        return self

    def batch_reformat(self, batch):
        vectors = list(batch[0])
        labels = list(batch[1])

        onehot_labels = labels2onehot_one(labels, self.n_classes, self.batch_size)

        # if len(labels_vec) != self.batch_size:
        #     shape = (self.batch_size - len(labels_vec), len(labels_vec[0]))
        #     labels_vec = np.concatenate((labels_vec, np.zeros(shape)), axis=0)

        matrix = np.zeros((self.batch_size, self.text_size, self.embedding_size))
        for i, x in enumerate(vectors):
            for j, y in enumerate(x):
                if j < self.text_size:
                    for k, d in enumerate(y):
                            matrix[i][j][k] = d

        batch = (matrix, onehot_labels)

        return batch

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
        if labels is not None:
            features = batch
            onehot_labels = labels
            metrics_values = self.model.test_on_batch(features, onehot_labels.reshape(-1, self.n_classes))
            return metrics_values
        else:
            predictions = self.model.predict(batch)
            return predictions

    def fit(self, dictionary, fit_name=None):
        """
        Method pipelines the intent_model using batches and validation
        Args:
            dictionary: dictionary
            fit_name: str name of key in input dictionary

        Returns: None

        """

        self._fill_names(fit_name, None, None)

        updates = 0

        val_loss = 1e100
        val_increase = 0
        epochs_done = 0

        n_train_samples = len(dictionary[self.fit_name]['x'])
        print('\n____Training over {} samples____\n\n'.format(n_train_samples))

        # get classes amount and init model
        if not self.model_init:
            self.n_classes = len(list(set(dictionary[self.fit_name]['y'])))
            self.model = self.cnn_model()
            self.init_model_from_scratch(add_metrics_file=metrics_file)
            self.model_init = True

        # init dataset object
        dataset = Dataset(dictionary, seed=SEED)

        while epochs_done < self.epochs:
            batch_gen = dataset.iter_batch(batch_size=self.batch_size,
                                           data_type=self.fit_name)

            for step, batch in enumerate(batch_gen):
                batch = self.batch_reformat(batch)
                metrics_values = self.train_on_batch(batch)
                updates += 1

                if self.verbose and step % 500 == 0:
                    names = ['loss']
                    if isinstance(self.metrics_names, str):
                        names.append(self.metrics_names)
                    elif isinstance(self.metrics_names, list):
                        for x in self.metrics_names:
                            names.append(x)

                    log_metrics(names=names,
                                values=metrics_values,
                                updates=updates,
                                mode='train')

            epochs_done += 1
            if epochs_done % self.val_every_n_epochs == 0:
                if 'valid' in dataset.data.keys():

                    valid_batch_gen = dataset.iter_batch(batch_size=self.batch_size,
                                                         data_type='valid')
                    valid_metrics_values = []
                    for valid_step, valid_batch in enumerate(valid_batch_gen):
                        valid_batch = self.batch_reformat(valid_batch)
                        valid_metrics_values.append(self.infer_on_batch(valid_batch[0],
                                                                        labels=valid_batch[1]))

                    valid_metrics_values = np.mean(np.asarray(valid_metrics_values), axis=0)

                    names = ['loss']
                    if isinstance(self.metrics_names, str):
                        names.append(self.metrics_names)
                    elif isinstance(self.metrics_names, list):
                        for x in self.metrics_names:
                            names.append(x)

                    log_metrics(names=names,
                                values=valid_metrics_values,
                                mode='valid')
                    if valid_metrics_values[0] > val_loss:
                        val_increase += 1
                        print("__Validation impatience {} out of {}".format(
                            val_increase, self.val_patience))
                        if val_increase == self.val_patience:
                            print("___Stop training: validation is out of patience___")
                            break
                    else:
                        val_increase = 0
                        val_loss = valid_metrics_values[0]
            print('epochs_done: {}'.format(epochs_done))

        self.trained = True
        # self.save()

        return self

    def predict(self, dictionary, pred_names=None, new_names=None):
        """
        Method returns predictions on the given data
        Args:
            dictionary: sentence or list of sentences
            pred_names: str or list of keys of input dictionary
            new_names: str or list of keys of input dictionary

        Returns:
            Predictions for the given data
        """

        self._fill_names(None, pred_names, new_names)

        if not self.model_init:
            raise ValueError('Model is not initialized.')
        if not self.trained:
            raise ValueError('Model is not trained.')

        # init dataset object
        dataset = Dataset(dictionary, seed=SEED)

        for name, new_name in zip(self.request_names, self.new_names):

            if isinstance(dictionary[name], dict):
                data = dictionary[name]['x']
            else:
                data = dictionary[name]

            if type(data) is str:
                preds = self.infer_on_batch([data])[0]
                preds = np.array(preds)
            elif (type(data) is list) or isinstance(data, pd.Series):
                if len(data) > self.batch_size:
                    batch_gen = dataset.iter_batch(batch_size=self.batch_size,
                                                   data_type=name, shuffle=False)
                    predictions = []
                    k = 0
                    for batch in batch_gen:
                        pred_batch = self.batch_reformat(batch)
                        prediction = self.infer_on_batch(pred_batch[0])
                        prediction = np.array(prediction)
                        predictions.append(prediction)
                        k += 1

                    # print(k)

                    # create one list of predictions from list of batches
                    preds = predictions[0]
                    for x in predictions[1:]:
                        preds = np.concatenate((preds, x), axis=0)

                    preds = np.argmax(preds, axis=1)
                    for i, x in enumerate(preds):
                        preds[i] = x + 1

                else:
                    # TODO fix big batch
                    batch_gen = dataset.iter_batch(batch_size=self.batch_size,
                                                   data_type=name, shuffle=False)
                    predictions = []
                    for batch in batch_gen:
                        pred_batch = self.batch_reformat(batch)

                        preds = self.infer_on_batch(pred_batch[0])
                        preds = np.array(preds)
                        predictions.append(preds)

                    preds = predictions
            else:
                raise ValueError("Not understand data type for inference")

            if isinstance(dictionary[name], dict):
                if 'y' in dictionary[name].keys():
                    dictionary[new_name] = {'y_pred': preds, 'y_true': dictionary[name]['y']}
                else:
                    dictionary[new_name] = {'y_pred': preds}
            else:
                dictionary[new_name] = {'y_pred': preds}

        return dictionary

    def fit_transform(self, dictionary, fit_name=None, pred_names=None, new_names=None):
        self._fill_names(fit_name, pred_names, new_names)

        self.fit(dictionary)
        out = self.predict(dictionary)

        return out

    def set_params(self, **params):
        super().set_params(**params)
        self.model_init = False
        if self.classes is not None:
            self.n_classes = np.array(self.classes.split(" ")).shape[0]
            self.model = self.cnn_model()
            self.init_model_from_scratch(add_metrics_file=metrics_file)
            self.model_init = True
        return self


class DCNN(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='DCNN',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 kernel_sizes_cnn="1 2 3",
                 filters_cnn="64 128 256",
                 embedding_size=300,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=25,
                 coef_reg_cnn=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.kernel_sizes_cnn = kernel_sizes_cnn
        self.filters_cnn = filters_cnn
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_cnn = coef_reg_cnn
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled model of deep CNN
        Args:
            self: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        if type(self.filters_cnn) is str:
            self.filters_cnn = list(map(int, self.filters_cnn.split()))

        if type(self.kernel_sizes_cnn) is str:
            self.kernel_sizes_cnn = [int(x) for x in self.kernel_sizes_cnn.split(' ')]

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = inp

        for i in range(len(self.kernel_sizes_cnn)):
            output = Conv1D(self.filters_cnn[i], kernel_size=self.kernel_sizes_cnn[i],
                            activation=None,
                            kernel_regularizer=l2(self.coef_reg_cnn),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model


class MAPCNN(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='MAPCNN',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 kernel_sizes_cnn="1 2 3",
                 filters_cnn=256,
                 embedding_size=300,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=25,
                 coef_reg_cnn=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.kernel_sizes_cnn = kernel_sizes_cnn
        self.filters_cnn = filters_cnn
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_cnn = coef_reg_cnn
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled model of shallow-and-wide CNN
        where average pooling after convolutions
        is replaced with concatenation of average and max poolings
        Args:
            self: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        # if type(self.filters_cnn) is str:
        #     self.filters_cnn = list(map(int, self.filters_cnn.split()))

        if type(self.kernel_sizes_cnn) is str:
            self.kernel_sizes_cnn = [int(x) for x in self.kernel_sizes_cnn.split(' ')]

        outputs = []
        for i in range(len(self.kernel_sizes_cnn)):
            output_i = Conv1D(self.filters_cnn, kernel_size=self.kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(self.coef_reg_cnn),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i_0 = GlobalMaxPooling1D()(output_i)
            output_i_1 = GlobalAveragePooling1D()(output_i)
            output_i = Concatenate()([output_i_0, output_i_1])
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class BiLSTM(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='BiLSTM',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 units_lstm=64,
                 lear_metrics="fmeasure",
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 rec_dropout_rate=0.5,
                 confident_threshold=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.units_lstm = units_lstm
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled BiLSTM
        Args:
            self: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        # print(type(self.rec_dropout_rate))

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = Bidirectional(LSTM(self.units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(inp)  # i don't understand

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class BiBiLSTM(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='BiBiLSTM',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 units_lstm_1=64,
                 units_lstm_2=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.units_lstm_1 = units_lstm_1
        self.units_lstm_2 = units_lstm_2
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled two-layers BiLSTM
        Args:
            self: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = Bidirectional(LSTM(self.units_lstm_1, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(inp)

        output = Dropout(rate=self.dropout_rate)(output)

        output = Bidirectional(LSTM(self.units_lstm_2, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class BiGRU(BiLSTM):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='BiGRU',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 units_lstm_1=64,
                 units_lstm_2=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.units_lstm_1 = units_lstm_1
        self.units_lstm_2 = units_lstm_2
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Method builds uncompiled model BiGRU
        Args:
            self: disctionary of parameters for NN
        Returns:
            Uncompiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = Bidirectional(GRU(self.units_lstm, activation='tanh',
                                   return_sequences=True,
                                   kernel_regularizer=l2(self.coef_reg_lstm),
                                   dropout=self.dropout_rate,
                                   recurrent_dropout=self.rec_dropout_rate))(inp)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)
        return model


class SelfAttMultBiLSTM(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='SelfAttMultBiLSTM',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 units_lstm=64,
                 self_att_hid=64,
                 self_att_out=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.self_att_hid = self_att_hid
        self.self_att_out = self_att_out
        self.units_lstm = units_lstm
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Method builds uncompiled model of BiLSTM with self multiplicative attention
        Args:
            self.: disctionary of parameters for NN
        Returns:
            Uncompiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = Bidirectional(LSTM(self.units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(inp)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = multiplicative_self_attention(output, n_hidden=self.self_att_hid,
                                               n_output_features=self.self_att_out)
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class SelfAttAddBiLSTM(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='SelfAttAddBiLSTM',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 units_lstm=64,
                 self_att_hid=64,
                 self_att_out=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.self_att_hid = self_att_hid
        self.self_att_out = self_att_out
        self.units_lstm = units_lstm
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Method builds uncompiled model of BiLSTM with self additive attention
        Args:
            params: disctionary of parameters for NN
        Returns:
            Uncompiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))
        output = Bidirectional(LSTM(self.units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(inp)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = additive_self_attention(output, n_hidden=self.self_att_hid,
                                         n_output_features=self.self_att_out)
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class CNN_BiLSTM(WCNN):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='CNNBiLSTM',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 kernel_sizes_cnn=[1, 2, 3],
                 filters_cnn=256,
                 coef_reg_cnn=1e-4,
                 units_lstm=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.kernel_sizes_cnn = kernel_sizes_cnn
        self.filters_cnn = filters_cnn
        self.coef_reg_cnn = coef_reg_cnn
        self.units_lstm = units_lstm
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled BiLSTM-CNN
        Args:
            params: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        outputs = []
        for i in range(len(self.kernel_sizes_cnn)):
            output_i = Conv1D(self.filters_cnn, kernel_size=self.kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(self.coef_reg_cnn),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = MaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)
        output = Dropout(rate=self.dropout_rate)(output)

        output = Bidirectional(LSTM(self.units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)

        return model


class BiLSTM_CNN(CNN_BiLSTM):
    def __init__(self, fit_name="train", predict_names=["test", "valid"], new_names=["pred_test", "pred_valid"],
                 op_type='model', op_name='BiLSTMCNN',
                 checkpoint_path="./data/russian/vkusvill/checkpoints/CNN/",
                 embedding_size=300,
                 kernel_sizes_cnn=[1, 2, 3],
                 filters_cnn=256,
                 coef_reg_cnn=1e-4,
                 units_lstm=64,
                 rec_dropout_rate=0.5,
                 lear_metrics="fmeasure",
                 confident_threshold=0.5,
                 model_from_saved=False,
                 optimizer="Adam",
                 lear_rate=0.1,
                 lear_rate_decay=0.1,
                 loss="binary_crossentropy",
                 last_activation="softmax",
                 text_size=15,
                 coef_reg_lstm=1e-4,
                 coef_reg_den=1e-4,
                 dropout_rate=0.5,
                 epochs=20,
                 dense_size=100,
                 model_name="cnn_model",
                 batch_size=64,
                 val_every_n_epochs=5,
                 verbose=True,
                 val_patience=5,
                 classes=None,
                 metrics_names='fmeasure'):

        super().__init__(fit_name, predict_names, new_names, op_type, op_name)

        # graph variables
        self.kernel_sizes_cnn = kernel_sizes_cnn
        self.filters_cnn = filters_cnn
        self.coef_reg_cnn = coef_reg_cnn
        self.units_lstm = units_lstm
        self.coef_reg_lstm = coef_reg_lstm
        self.rec_dropout_rate = rec_dropout_rate
        self.embedding_size = embedding_size
        self.last_activation = last_activation
        self.text_size = text_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.dense_size = dense_size
        self.coef_reg_den = coef_reg_den
        self.classes = classes
        self.n_classes = None

        # learning parameters
        self.lear_metrics = lear_metrics
        self.confident_threshold = confident_threshold
        self.optimizer = optimizer
        self.lear_rate = lear_rate
        self.lear_rate_decay = lear_rate_decay
        self.loss = loss
        self.epochs = epochs

        # other
        self.checkpoint_path = checkpoint_path
        self.model_from_saved = model_from_saved
        self.model_name = model_name
        self.val_every_n_epochs = val_every_n_epochs
        self.verbose = verbose
        self.val_patience = val_patience
        self.metrics_names = metrics_names
        self.metrics_values = None

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)
            self.model_init = True
        else:
            if self.classes is not None:
                self.n_classes = np.array(self.classes.split(" ")).shape[0]
                self.model = self.cnn_model()
                self.model = self.init_model_from_scratch(add_metrics_file=metrics_file)
                self.model_init = True

    def cnn_model(self):
        """
        Build un-compiled BiLSTM-CNN
        Args:
            self.: dictionary of parameters for NN
        Returns:
            Un-compiled model
        """

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = Bidirectional(LSTM(self.units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(self.coef_reg_lstm),
                                    dropout=self.dropout_rate,
                                    recurrent_dropout=self.rec_dropout_rate))(inp)

        output = Reshape(target_shape=(self.text_size, 2 * self.units_lstm))(output)
        outputs = []
        for i in range(len(self.kernel_sizes_cnn)):
            output_i = Conv1D(self.filters_cnn,
                              kernel_size=self.kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(self.coef_reg_cnn),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = Concatenate(axis=1)(outputs)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
