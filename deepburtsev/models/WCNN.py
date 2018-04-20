import keras.metrics
import keras.optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from deepburtsev.core import metrics as metrics_file
from deepburtsev.core.keras_model import KerasModel

# from deepburtsev.core.utils import log_metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))


class WCNN(KerasModel):
    def __init__(self, fit_name=["train_vec"], predict_names=["predicted_test"], new_names=["test_vec"],
                 op_type='keras_model', op_name='WCNN',
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
                 val_every_n_epochs=30,
                 verbose=True,
                 val_patience=5,
                 n_classes=None):

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
        self.n_classes = n_classes

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
        self.metrics_names = None
        self.metrics_values = None

        # model graph
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

        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_size, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation(self.last_activation)(output)
        self.model = Model(inputs=inp, outputs=act_output)

        # load weights if need
        if self.model_from_saved:
            self.model = self.restore(self.checkpoint_path)

        # check metrics and optimizers
        optimizer_func = getattr(keras.optimizers, self.optimizer, None)

        if callable(optimizer_func):
            self.optimizer = optimizer_func(lr=self.lear_rate, decay=self.lear_rate_decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(self.optimizer))

        loss_func = getattr(keras.losses, self.loss, None)
        if callable(loss_func):
            self.loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(self.loss))

        metrics_names = self.lear_metrics.split(' ')
        self.metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                self.metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(metrics_file, metrics_names[i], None)
                if callable(metrics_func):
                    self.metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(metrics_names[i]))

    def init_model(self, dataset):

        self.n_classes = len(dataset.get_classes())

        if not self.model_init:
            # compilation
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics_funcs,
                               loss_weights=None,
                               sample_weight_mode=None,
                               # weighted_metrics=weighted_metrics,
                               # target_tensors=target_tensors
                               )

            self.metrics_names = self.model.metrics_names
            self.metrics_values = len(self.metrics_names) * [0.]

            self.model_init = True
        else:
            raise AttributeError('Model was already initialized. Add reset method in your model'
                                 'or create new pipeline')
        return self
