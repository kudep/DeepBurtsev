from .WCNN import WCNN
from keras.layers import Dense, Input, Activation, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2


class DCNN(WCNN):
    def cnn_model(self):
        """
        Method builds uncompiled intent_model of deep CNN

        Returns:
            Uncompiled intent_model
        """
        if type(self.kernel_sizes_cnn) is str:
            self.kernel_sizes_cnn = [int(x) for x in self.kernel_sizes_cnn.split(' ')]

        if type(self.filters_cnn) is str:
            self.filters_cnn = [int(x) for x in self.filters_cnn.split(' ')]

        inp = Input(shape=(self.text_size, self.embedding_size))

        output = inp

        for i in range(len(self.kernel_sizes_cnn)):
            output = Conv1D(self.filters_cnn, kernel_size=self.kernel_sizes_cnn[i],
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
        act_output = Activation(self.last_activation)(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
