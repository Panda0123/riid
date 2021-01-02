import numpy as np
import tensorflow as tf
from tensorflow.train import Example, BytesList, FloatList, Feature, Features
import os

class DatasetLoaderStatelessRNN:
    """Data loader for StatlessRNN created from DataCreatorStatelessRNN.

    args:
            root: the root directory of the tfrecords
            nSteps: # of time steps
            name: the type of dataset, is it train or valid
            nCon: number of continuous features
            nCat: number of categorical features
    """
    def __init__(self, root, nSteps, name, nCon, nCat):
        self.root = root
        self.name = name
        self.nSteps = nSteps
        self.nCon = nCon
        self.nCat = nCat
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
        catFeat = tf.reshape(catFeat, shape=[self.nSteps, self.nCat])
        return ((conFeat, catFeat), tf.cast(example["label"], tf.int8))

class DatasetLoaderStatelessRNNModified:
    """Data loader for StatlessRNN created from DataCreatorStatelessRNN.
    args:
            root: the root directory of the tfrecords
            nSteps: # of time steps
            name: the type of dataset, is it train or valid
            nCon: number of continuous features
            nCat: number of categorical features
    """
    def __init__(self, root, nSteps, name, nCon, nCat):
        self.root = root
        self.name = name
        self.nSteps = nSteps
        self.nCon = nCon
        self.nCat = nCat
        self.filePaths = [os.path.join(root, path) for path in os.listdir(
            root) if path.split("_")[0] == self.name]

    def loadDataset(self, batch_size=32, cache=None, shuffle_buffer_size=None):
        dataset = tf.data.Dataset.list_files(self.filePaths)
        dataset = dataset.interleave( lambda fPath:
            tf.data.TFRecordDataset(fPath),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if cache:
            dataset = dataset.cache(cache)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.map(
            self._prepProcess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        # have to cast for table lookup (saved it as int16 so I can save disk)
        catFeat = tf.cast(tf.io.parse_tensor(example["cat"], out_type=tf.int16),
                    tf.int64)
        conFeat = tf.reshape(conFeat, shape=[self.nSteps, self.nCon])
        catFeat = tf.reshape(catFeat, shape=[self.nSteps, self.nCat])

        # TODO: test remove the cast if still working (yes - remove it)
        return ((conFeat, catFeat), example["label"])

class DatasetLoaderStatelessRNNBatch:
    """Data loader for StatlessRNN created from DataCreatorStatelessRNN.

    args:
            root: the root directory of the tfrecords
            nSteps: # of time steps
            name: the type of dataset, is it train or valid
            nCon: number of continuous features
            nCat: number of categorical features
    """
    def __init__(self, root, nSteps, name, nCon, nCat):
        self.root = root
        self.name = name
        self.nSteps = nSteps
        self.nCon = nCon
        self.nCat = nCat
        self.filePaths = [os.path.join(root, path) for path in os.listdir(
            root) if path.split("_")[0] == self.name]

    def loadDataset(self, cache=None, shuffle_buffer_size=None, batch_size=32):
        dataset = tf.data.Dataset.list_files(self.filePaths)
        dataset = dataset.interleave( lambda fPath:
            tf.data.TFRecordDataset(fPath),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            self.parseBatch,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if cache:
            dataset = dataset.cache()

        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def parseBatch(self, serializedExample):
        mapping = {
            "con": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "cat": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "label": tf.io.FixedLenFeature([], tf.float32, default_value=-1)
        }

        parsedBatch = tf.io.parse_example(serializedExample, mapping)
        catFeat = self.parseCatMapF(parsedBatch["cat"])
        conFeat = self.parseConMapF(parsedBatch["con"])


        # TODO: might need to cast parsedBatch - no need
        return ((conFeat, catFeat), parsedBatch["label"])

    # TODO: probably remove reshape - can't
    @tf.function
    def parseCatMapF(self, elem):
        return tf.map_fn(
                lambda x: tf.reshape(
                            tf.cast(tf.io.parse_tensor(x, out_type=tf.int16),
                                    tf.int64),
                            shape=[self.nSteps, self.nCat]),
                elem,
                parallel_iterations=10,
                fn_output_signature=tf.int64)

    # TODO: probably remove reshape - can't
    @tf.function
    def parseConMapF(self, elem):
        return tf.map_fn(
                lambda x: tf.reshape(
                            tf.io.parse_tensor(x, out_type=tf.float64),
                                shape=[self.nSteps, self.nCon]),
                elem,
                parallel_iterations=10,
                fn_output_signature=tf.float64)
