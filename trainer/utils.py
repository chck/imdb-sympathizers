# -*- coding: utf-8 -*-
import csv
import os
from collections import Iterable, OrderedDict
from logging import getLogger, StreamHandler, DEBUG

import numpy as np
import six
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import CSVLogger, Callback
from tensorflow.python.lib.io import file_io

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


def get_available_gpus():
    return [x.name for x in K.get_session().list_devices() if 'GPU' in x.name]


def set_session(debug_mode=False):
    sess = K.get_session()
    if debug_mode:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
    K.set_session(sess)
    return sess


def copy_file_to_gcs(remotekey, filename, rm_local=False):
    with file_io.FileIO(filename, mode='r') as input_f:
        with file_io.FileIO(os.path.join(remotekey, filename), mode='w+') as output_f:
            output_f.write(input_f.read())
    if rm_local:
        os.remove(filename)


class ModelCheckpointToGcs(Callback):
    def __init__(self,
                 remotekey,
                 filename,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(ModelCheckpointToGcs, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.remotekey = remotekey
        self.filename = filename
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            logger.warning('ModelCheckpoint mode %s is unknown, '
                           'fallback to auto mode.' % mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filename = self.filename.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warning('Can save best model only with %s available, '
                                   'skipping.' % (self.monitor))
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filename))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filename, overwrite=True)
                            copy_file_to_gcs(self.remotekey, filename, rm_local=True)
                        else:
                            self.model.save(filename, overwrite=True)
                            copy_file_to_gcs(self.remotekey, filename, rm_local=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' % (epoch + 1,
                                                                      self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filename))
                if self.save_weights_only:
                    self.model.save_weights(filename, overwrite=True)
                    copy_file_to_gcs(self.remotekey, filename, rm_local=True)
                else:
                    self.model.save(filename, overwrite=True)
                    copy_file_to_gcs(self.remotekey, filename, rm_local=True)


class CSVLoggerToGcs(CSVLogger):
    def __init__(self, remotekey, filename, separator=',', append=False):
        super(CSVLoggerToGcs, self).__init__(filename, separator, append)
        self.remotekey = remotekey

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=['epoch'] + self.keys,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        copy_file_to_gcs(remotekey=self.remotekey, filename=self.filename)

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        os.remove(self.filename)
