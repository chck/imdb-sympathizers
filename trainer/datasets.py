# -*- coding: utf-8 -*-
from tensorflow.python.keras.datasets import imdb as _imdb
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix


def load_imdb(max_features=20000, maxlen=100):
    (X_train, y_train), (X_val, y_val) = _imdb.load_data(num_words=max_features)
    X_train = csr_matrix(pad_sequences(X_train, maxlen=maxlen))
    X_val = csr_matrix(pad_sequences(X_val, maxlen=maxlen))
    return (X_train, y_train), (X_val, y_val)
