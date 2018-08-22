# -*- coding: utf-8 -*-
from os import path

from absl import app, flags
from tensorflow.python.keras.utils import multi_gpu_model

from trainer.datasets import load_imdb
from trainer.models import BaseModel
from trainer.utils import ModelCheckpointToGcs, CSVLoggerToGcs, set_session, get_available_gpus

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', None, 'Specify model name to train.')
flags.DEFINE_string('gcs_path', None,
                    'String name of gcs path to put training results. Do not include file name. e.g. gs://xxx/yyy/ ')
flags.DEFINE_string('optimizer', 'adam', 'String name of optimizer.')
flags.DEFINE_integer('max_features', 20000, 'Max number of words to include.')
flags.DEFINE_integer('maxlen', 1000, 'Sequences longer than this will be filtered out.')
flags.DEFINE_integer('batch_size', 128, 'Number of samples per gradient update.')
flags.DEFINE_integer('epochs', 10, 'An epoch is an iteration over the entire `x` and `y` data provided.')


def main(argv):
    with set_session().as_default():
        (X_train, y_train), (X_val, y_val) = load_imdb(max_features=FLAGS.max_features, maxlen=FLAGS.maxlen)
        basemodel = BaseModel(input_dim=FLAGS.max_features, maxlen=FLAGS.maxlen)
        model = multi_gpu_model(basemodel.build(name=FLAGS.model_name), gpus=len(get_available_gpus()))
        model.compile(loss='binary_crossentropy', optimizer=FLAGS.optimizer, metrics=['accuracy'])
        model.fit(X_train.todense(), y_train,
                  validation_data=(X_val.todense(), y_val),
                  batch_size=FLAGS.batch_size,
                  epochs=FLAGS.epochs,
                  callbacks=[
                      ModelCheckpointToGcs(remotekey=FLAGS.gcs_path,
                                           filename='model_checkpoint.{epoch:02d}_{loss:.6f}_{val_loss:.6f}.h5',
                                           save_best_only=True),
                      CSVLoggerToGcs(remotekey=FLAGS.gcs_path,
                                     filename=path.join('history_{}_{}.tsv').format(FLAGS.model_name, FLAGS.optimizer),
                                     separator='\t'),
                  ])


if __name__ == '__main__':
    flags.mark_flags_as_required(['model_name', 'gcs_path'])
    app.run(main)
