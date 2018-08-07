# -*- coding: utf-8 -*-
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import (
    Dense, Input, InputLayer, Dropout,
    Embedding, LSTM, Bidirectional, Conv1D, MaxPool1D,
    Permute, Reshape, Multiply, Flatten, Layer
)
from tensorflow.python.keras.models import Model, Sequential

from trainer.datasets import load_imdb

model_names = [
    'lstm_simple',
    'lstm_simple_dropout',
    'bilstm_dropout',
    'cnn_bilstm_dropout',
    'cnn_bilstm_attention_dropout',
]


class BaseModel:
    def __init__(self, input_dim: int, embed_dim=128, maxlen=100, units=64,
                 conv_filters=64, conv_kernel_size=5, conv_pool_size=4,
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
        elif name == 'cnn_bilstm_dropout':
            return self._cnn_bilstm_dropout(name)
        elif name == 'cnn_bilstm_attention_dropout':
            return self._cnn_bilstm_attention_dropout(name)
        else:
            RuntimeError('Unexpected model name. Specify name in {}'.format(','.join(model_names)))

    def _lstm_simple(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.input_dim,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            LSTM(units=self.units, name='lstm'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _lstm_simple_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.input_dim,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Dropout(0.2, name='input_dropout'),
            LSTM(units=self.units, name='lstm'),
            Dropout(0.5, name='hidden_dropout'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _bilstm_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.input_dim,), name='input'),
            Embedding(input_dim=self.input_dim, output_dim=self.embed_dim, input_length=self.maxlen, name='embedding'),
            Dropout(0.2, name='input_dropout'),
            Bidirectional(LSTM(units=self.units, name='bilstm')),
            Dropout(0.5, name='hidden_dropout'),
            Dense(self.classes, activation='sigmoid', name='fc1'),
        ], name=name)

    def _cnn_bilstm_dropout(self, name: str) -> Model:
        return Sequential([
            InputLayer(input_shape=(self.input_dim,), name='input'),
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
        _input = Input(shape=(self.input_dim,), name='input')
        _output = Embedding(input_dim=self.input_dim,
                            output_dim=self.embed_dim,
                            input_length=self.maxlen,
                            name='embedding')(_input)
        _output = Dropout(0.2, name='input_dropout')(_output)
        _output = Conv1D(filters=self.conv_filters,
                         kernel_size=self.conv_kernel_size,
                         padding='same',
                         activation='relu')(_output)
        _pool_out = MaxPool1D(pool_size=self.conv_pool_size)(_output)
        _output = Bidirectional(LSTM(units=self.units,
                                     dropout=0.2,
                                     recurrent_dropout=0.2,
                                     return_sequences=True,
                                     name='bilstm_dropout'))(_pool_out)
        _output = Flatten()(self.__attention_3d_block(_output, _pool_out.shape[1].value))
        _output = Dropout(0.5, name='hidden_dropout')(_output)
        _output = Dense(self.classes, activation='sigmoid', name='fc1')(_output)
        return Model(inputs=_input, outputs=_output, name=name)

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
