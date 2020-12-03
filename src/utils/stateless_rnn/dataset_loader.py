import numpy as np
import tensorflow as tf
from tensorflow.train import Example, BytesList, FloatList, Feature, Features
import os

class DatasetLoaderStatelessRNN:
    def __init__(self, root, nSteps, name, nCon):
        self.root = root
        self.name = name
        self.nSteps = nSteps
        self.nCon = nCon
        self.filePaths = [os.path.join(root, path) for path in os.listdir(
            root) if path.split("_")[0] == self.name]

    def loadDataset(self, n_read_threads=5, n_prep_threads=5, cache=None,
                    shuffle_buffer_size=None, batch_size=32):
        dataset = tf.data.TFRecordDataset(
            self.filePaths, num_parallel_reads=n_read_threads)

        if cache:
            dataset = dataset.cache(cache)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.map(
            self._prepProcess, num_parallel_calls=n_prep_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def _prepProcess(self, serializedExample):
        mapping = {
            "con": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "cat": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "label": tf.io.FixedLenFeature([], tf.float32, default_value=-1)
        }

        example = tf.io.parse_single_example(serializedExample, mapping)
        conFeat = tf.io.parse_tensor(example["con"], out_type=tf.float64)
        # have to cast for table lookup
        catFeat = tf.cast(tf.io.parse_tensor(
            example["cat"], out_type=tf.int16), tf.int64)
        conFeat = tf.reshape(conFeat, shape=[self.nSteps, self.nCon])
        catFeat = tf.reshape(catFeat, shape=[self.nSteps, 5])
        return ((conFeat, catFeat), tf.cast(example["label"], tf.int8))

