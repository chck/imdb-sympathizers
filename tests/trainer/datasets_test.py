# -*- coding: utf-8 -*-
import numpy as np
import pytest

from trainer.datasets import load_imdb


def test_load_imdb():
    max_features = 10000
    maxlen = 500
    (X_train, y_train), (X_val, y_val) = load_imdb(max_features=max_features, maxlen=maxlen)
    assert max_features > np.unique(np.concatenate((X_train, X_val))).size
    assert maxlen == len(X_train[0])


if __name__ == '__main__':
    pytest.main([__file__])
