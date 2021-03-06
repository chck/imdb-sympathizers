# -*- coding: utf-8 -*-
from tensorflow.python.keras.layers import (
    Dense, Input, InputLayer, Dropout,
    Embedding, LSTM, Bidirectional, Conv1D, MaxPool1D, GlobalMaxPool1D, Concatenate,
    Permute, Reshape, Multiply, Flatten, Layer
)
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils import plot_model

from trainer.datasets import load_imdb

model_names = [
    'lstm_simple',
    'lstm_simple_dropout',
    'bilstm_dropout',
    'cnn_maxpool',
    'cnn_maxpool_multifilter',
    'cnn_bilstm_dropout',
    'cnn_bilstm_attention_dropout',
]


class BaseModel:
    def __init__(self, input_dim: int, embed_dim=128, maxlen=1000, units=128,
                 conv_filters=128, conv_kernel_size=5, conv_pool_size=5,
                 classes=1):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.units = units
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_pool_size = conv_pool_size
        self.classes = classes

    def build(self, name: str) -> Model:
        if name == 'lstm_simple':
            return self._lstm_simple(name)
        elif name == 'lstm_simple_dropout':
            return self._lstm_simple_dropout(name)
        elif name == 'bilstm_dropout':
            return self._bilstm_dropout(name)
        elif name == 'cnn_maxpool':
            return self._cnn_maxpool(name)
        elif name == 'cnn_maxpool_multifilter':
            return self._cnn_maxpool_multifilter(name)
        elif name == 'cnn_bilstm_dropout':
            return self._cnn_bilstm_dropout(name)
        elif name == 'cnn_bilstm_attention_dropout':
            return self._cnn_bilstm_attention_dropout(name)
        else:
            RuntimeError('Unexpected model name. Specify name in {}'.format(','.join(model_names)))

    def _lstm_simple(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.maxlen,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            LSTM(units=self.units, name='lstm'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _lstm_simple_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.maxlen,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Dropout(0.2, name='input_dropout'),
            LSTM(units=self.units, name='lstm'),
            Dropout(0.5, name='hidden_dropout'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _bilstm_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.maxlen,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Dropout(0.2, name='input_dropout'),
            Bidirectional(LSTM(units=self.units, name='bilstm')),
            Dropout(0.5, name='hidden_dropout'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _cnn_maxpool(self, name: str) -> Model:
        """https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
        """
        return Sequential([
            InputLayer(input_shape=(self.maxlen,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, activation='relu'),
            MaxPool1D(pool_size=self.conv_pool_size),
            Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, activation='relu'),
            MaxPool1D(pool_size=self.conv_pool_size),
            Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, activation='relu'),
            GlobalMaxPool1D(),
            Flatten(),
            Dense(self.units, activation='relu'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _cnn_maxpool_multifilter(self, name: str) -> Model:
        """https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
        """
        convs = []
        filter_sizes = [3, 4, 5]

        _inputs = Input((self.maxlen,), name='input')
        l_embed = Embedding(input_dim=self.input_dim,
                            output_dim=self.embed_dim,
                            input_length=self.maxlen,
                            name='embedding')(_inputs)

        for fsz in filter_sizes:
            l_conv = Conv1D(filters=self.conv_filters, kernel_size=fsz, activation='relu')(l_embed)
            l_pool = MaxPool1D(self.conv_pool_size)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)
        l_cov1 = Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, activation='relu')(l_merge)
        l_pool1 = MaxPool1D(pool_size=self.conv_pool_size)(l_cov1)
        l_cov2 = Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, activation='relu')(l_pool1)
        l_pool2 = GlobalMaxPool1D()(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(self.units, activation='relu')(l_flat)
        _preds = Dense(self.classes, activation='sigmoid', name='fc1')(l_dense)

        return Model(inputs=_inputs, outputs=_preds, name=name)

    def _cnn_bilstm_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.maxlen,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Dropout(0.2, name='input_dropout'),
            Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size, padding='same', activation='relu'),
            MaxPool1D(pool_size=self.conv_pool_size),
            Bidirectional(LSTM(units=self.units, name='bilstm')),
            Dropout(0.5, name='hidden_dropout'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _cnn_bilstm_attention_dropout(self, name: str) -> Model:
        """https://qiita.com/fufufukakaka/items/4f9d42a4300392691bf3
        """
        _inputs = Input(shape=(self.maxlen,), name='input')
        l_embed = Embedding(input_dim=self.input_dim,
                            output_dim=self.embed_dim,
                            input_length=self.maxlen,
                            name='embedding')(_inputs)
        l_drop1 = Dropout(0.2, name='input_dropout')(l_embed)
        l_cov1 = Conv1D(filters=self.conv_filters,
                        kernel_size=self.conv_kernel_size,
                        padding='same',
                        activation='relu')(l_drop1)
        l_pool1 = MaxPool1D(pool_size=self.conv_pool_size)(l_cov1)
        l_bilstm1 = Bidirectional(LSTM(units=self.units,
                                       dropout=0.2,
                                       recurrent_dropout=0.2,
                                       return_sequences=True,
                                       name='bilstm_dropout'))(l_pool1)
        l_flat = Flatten()(self.__attention_3d_block(l_bilstm1, l_pool1.shape[1].value))
        l_drop2 = Dropout(0.5, name='hidden_dropout')(l_flat)
        _preds = Dense(self.classes, activation='sigmoid', name='fc1')(l_drop2)
        return Model(inputs=_inputs, outputs=_preds, name=name)

    def __attention_3d_block(self, _lstm_output, _time_steps) -> Layer:
        """https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
        """
        att = Permute((2, 1))(_lstm_output)
        att = Reshape((_lstm_output.shape[2].value, _time_steps))(att)
        att = Dense(_time_steps, activation='softmax')(att)
        att_probs = Permute((2, 1), name='attention_vec')(att)
        return Multiply(name='attention_mul')([_lstm_output, att_probs])


if __name__ == '__main__':
    (X_train, y_train), (X_val, y_val) = load_imdb()
    basemodel = BaseModel(input_dim=X_train.shape[1])
    for name in model_names:
        model = basemodel.build(name)
        plot_model(model, show_shapes=True, to_file='../static/{}.png'.format(name))
