"""
DatasetCreator2 and DatasetCreator3 are similar except 3 is paritioned
DatasetCreator4 uses numpy (instead of DF) to save previous
interactios of each user.
"""

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.train import Example, BytesList, FloatList, Feature, Features
import pandas as pd
import tensorflow as tf
import numpy as np
import os

class DatasetCreatorV2:
    """ DatasetCreator for creating TFREcord for stateless RNN.
    Can specify the postfix of the generated files.

    Inspiration: So I can partition large dataset and create them by partition.

    args:
        nCon - specify the number of continuous features.
        isPartitioned - identify if the dataset that will be created
                        is partitioned. (partitioned by instances, not time)
        start - specify the starting of postfix of the generated file.
        end - specify the ending of postfix of the generated file.
    """

    def __init__(self, nCon, isPartitioned=False,
                 n_steps=10, shift=1, batch_size=32):
        self.n_steps = n_steps
        self.shift = shift
        self.batch_size = batch_size
        self.nCon = nCon
        self.isPartitioned = isPartitioned
        self.instances = {}

    def createTrain(self, df, rootFilePath, nShards, start, end):
        """
            Training dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            if self.isPartitioned:
                tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                                   for shard in range(start, end)]
            else:
                tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                                   for shard in range(nShards)]

            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_)
                # TO DO: save only the needed feature remove last 2 before adding
                self.instances[id_] = user.tail(self.n_steps - 1)
                dataset = self._createDataset(user)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(lambda x, y: tf.py_function(self._createExample, [x[0], x[1], y], tf.string),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createValid(self, df, rootFilePath, nShards, start, end):
        """
            Valid dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            if self.isPartitioned:
                tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                                   for shard in range(start, end)]
            else:
                tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                                   for shard in range(nShards)]

            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_)
                isNew = id_ not in self.instances.keys()
                dataset = self._createDataset(user, isNew=isNew)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(
                    lambda x, y: tf.py_function(
                        self._createExample, [x[0], x[1], y], tf.string),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createPredict(self, df):
        """
            Dataframe you want to make prediction of
        """
        feature0 = []
        feature1 = []
        for idx in df.index:
            userSer = df.loc[idx]
            if userSer.user_id in self.instances.keys():
                userDf = self.instances[userSer.user_id]
                userDf = userDf.append(userSer)
                if userDf.shape[0] < self.n_steps:
                    tempDf = pd.DataFrame(np.zeros(shape=(
                        self.n_steps - userDf.shape[0], df.shape[1])), columns=userSer.index.values)
                    userDf = tempDf.append(userDf)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())
            else:
                userDf = pd.DataFrame(
                    np.zeros(shape=(self.n_steps - 1, df.shape[1])), columns=userSer.index.values)
                userDf = userDf.append(userSer)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())

            self.instances[userSer.user_id] = userDf.tail(self.n_steps - 1)

        return (tf.constant(feature0, dtype=tf.float64), tf.cast(tf.constant(feature1), dtype=tf.int64))

    def _createDataset(self, df, isNew=True):
        # add a padding 0 for the first n_step timestep of new user
        if isNew:
            tempDf = pd.DataFrame(
                np.zeros(shape=(self.n_steps - 1, df.shape[-1])), columns=df.columns)
            df = tempDf.append(df)
        else:
            df = self.instances[df.user_id.values[0]].append(df)
            self.instances[df.user_id.values[0]] = df.tail(self.n_steps - 1)

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(df.values))
        dataset = dataset.window(
            self.n_steps, shift=self.shift, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.n_steps))
        return dataset

    def _createExample(self, feat0, feat1, label):
        insExample = Example(
            features=Features(
                feature={
                    "con": self._bytelist_feature_fromMatrix(feat0),
                    "cat": self._bytelist_feature_fromMatrix(feat1),
                    "label": self._float_feature(label)
                }))
        return insExample.SerializeToString()

    def _bytelist_feature_fromMatrix(self, temp):
        value = tf.io.serialize_tensor(temp)  # need to parse self when loading
        return Feature(bytes_list=BytesList(value=[value.numpy()]))

    def _float_feature(self, value):
        return Feature(float_list=FloatList(value=[value]))



# I HAVE TO OVERWRITE set_model TO SAVE THE MODEL USING ModelCheckpoint WITH save_weights_only=False
class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model


# TODO: removed since It is already implemented in DatasetCreatorV2
class DatasetCreatorV3:
    """ DatasetCreator for creating TFREcord for stateless RNN.
    Can specify the postfix of the generated files.

    Inspiration: So I can partition large dataset and create them by partition.

    args:
        isPartitioned - identify if the dataset that will be created
                        is partitioned.
        start - specify the starting of postfix of the generated file.
        end - specify the ending of postfix of the generated file.
    """

    def __init__(self, nCon, n_steps=10, shift=1, batch_size=32):
        self.n_steps = n_steps
        self.shift = shift
        self.batch_size = batch_size
        self.nCon = nCon
        self.instances = {}

    def createTrain(self, df, rootFilePath, nShards, start, end):
        """
            Training dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                               for shard in range(start, end)]
            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_)
                # TO DO: save only the needed feature remove last 2 before adding
                self.instances[id_] = user.tail(self.n_steps - 1)
                dataset = self._createDataset(user)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(lambda x, y: tf.py_function(self._createExample, [x[0], x[1], y], tf.string),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createValid(self, df, rootFilePath, nShards, start, end):
        """
            Valid dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/valid_{shard:05d}.tfrecord"))
                               for shard in range(start, end)]
            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_)
                isNew = id_ not in self.instances.keys()
                dataset = self._createDataset(user, isNew=isNew)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(
                    lambda x, y: tf.py_function(
                        self._createExample, [x[0], x[1], y], tf.string),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createPredict(self, df):
        """
            Dataframe you want to make prediction of
        """
        feature0 = []
        feature1 = []
        for idx in df.index:
            userSer = df.loc[idx]
            if userSer.user_id in self.instances.keys():
                userDf = self.instances[userSer.user_id]
                userDf = userDf.append(userSer)
                if userDf.shape[0] < self.n_steps:
                    tempDf = pd.DataFrame(np.zeros(shape=(
                        self.n_steps - userDf.shape[0], df.shape[1])), columns=userSer.index.values)
                    userDf = tempDf.append(userDf)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())
            else:
                userDf = pd.DataFrame(
                    np.zeros(shape=(self.n_steps - 1, df.shape[1])), columns=userSer.index.values)
                userDf = userDf.append(userSer)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())

            self.instances[userSer.user_id] = userDf.tail(self.n_steps - 1)

        return (tf.constant(feature0, dtype=tf.float64), tf.cast(tf.constant(feature1), dtype=tf.int64))

    def _createDataset(self, df, isNew=True):
        # add a padding 0 for the first n_step timestep of new user
        if isNew:
            tempDf = pd.DataFrame(
                np.zeros(shape=(self.n_steps - 1, df.shape[-1])), columns=df.columns)
            df = tempDf.append(df)
        else:
            df = self.instances[df.user_id.values[0]].append(df)
            self.instances[df.user_id.values[0]] = df.tail(self.n_steps - 1)

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(df.values))
        dataset = dataset.window(
            self.n_steps, shift=self.shift, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.n_steps))
        return dataset

    def _createExample(self, feat0, feat1, label):
        insExample = Example(
            features=Features(
                feature={
                    "con": self._bytelist_feature_fromMatrix(feat0),
                    "cat": self._bytelist_feature_fromMatrix(feat1),
                    "label": self._float_feature(label)
                }))
        return insExample.SerializeToString()

    def _bytelist_feature_fromMatrix(self, temp):
        value = tf.io.serialize_tensor(temp)  # need to parse self when loading
        return Feature(bytes_list=BytesList(value=[value.numpy()]))

    def _float_feature(self, value):
        return Feature(float_list=FloatList(value=[value]))


class DatasetCreatorV4:
    """ DatasetCreator for creating TFREcord for stateless RNN.
    Can specify the postfix of the generated files. Uses numpy
    keep records of previous interactions of the instances instead of
    the whole dataframe.

    Inspiration: So I can partition large dataset and create them by partition.
    """

    def __init__(self, nCon, n_steps=10, shift=1, batch_size=32):
        self.n_steps = n_steps
        self.shift = shift
        self.batch_size = batch_size
        self.nCon = nCon
        self.instances = {}

    def createTrain(self, df, rootFilePath, nShards, start, end):
        """
            Training dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/train_{shard:05d}.tfrecord"))
                               for shard in range(start, end)]
            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_)
                self.instances[id_] = user.tail(self.n_steps - 1).values
                dataset = self._createDataset(user.values, id_)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(lambda x, y: tf.py_function(self._createExample, [x[0], x[1], y], tf.string),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createValid(self, df, rootFilePath, nShards, start, end):
        """
            Valid dataframe
        """
        from contextlib import ExitStack
        grouped = df.groupby("user_id")
        keys = list(grouped.groups.keys())
        # iterate through the users and create a dataset for each, then iterate through dataset and save random each
        with ExitStack() as stack:
            tfRecordWriters = [stack.enter_context(tf.io.TFRecordWriter(f"{rootFilePath}/valid_{shard:05d}.tfrecord"))
                               for shard in range(nShards)]
            counter = 0
            for id_ in keys:
                user = grouped.get_group(id_).values
                isNew = id_ not in self.instances.keys()
                dataset = self._createDataset(user, id_=id_, isNew=isNew)
                dataset = dataset.map(lambda instance: ((instance[:, :self.nCon], tf.cast(instance[:, self.nCon:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
                                                        tf.cast(instance[-1, -1], tf.int8)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # label
                dataset = dataset.map(
                    lambda x, y: tf.py_function(
                        self._createExample, [x[0], x[1], y], tf.string),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                for instance in dataset.as_numpy_iterator():
                    shard = counter % nShards
                    tfRecordWriters[shard].write(instance)
                    counter += 1
            del user
            del dataset
        del tfRecordWriters
        return counter

    def createPredict(self, df):
        """
            Dataframe you want to make prediction of
        """
        feature0 = []
        feature1 = []
        for idx in df.index:
            userSer = df.loc[idx]
            if userSer.user_id in self.instances.keys():
                userDf = self.instances[userSer.user_id]
                userDf = userDf.append(userSer)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())
            else:
                userDf = pd.DataFrame(np.zeros(
                    shape=(self.n_steps - 1, userSer.shape[0])), columns=userSer.index.values)
                userDf = userDf.append(userSer)
                feature0.append(userDf.values[:, :self.nCon].tolist())
                feature1.append(userDf.values[:, self.nCon:-2].tolist())
            self.instances[userSer.user_id] = userDf.tail(self.n_steps - 1)
        return (tf.constant(feature0, dtype=tf.float64), tf.cast(tf.constant(feature1), dtype=tf.int64))

    def _createDataset(self, arr, id_, isNew=True):
        # add a padding 0 for the first n_step timestep of new user
        if isNew:
            tempArr = np.zeros(shape=(self.n_steps - 1, arr.shape[-1]))
            arr = np.r_[tempArr, arr]
        else:
            arr = np.r_[self.instances[arr[0, -1]], arr]
            self.instances[id_] = arr[-(self.n_steps - 1):, :]
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(arr))
        dataset = dataset.window(
            self.n_steps, shift=self.shift, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.n_steps))
        return dataset

    def _createExample(self, feat0, feat1, label):
        insExample = Example(
            features=Features(
                feature={
                    "con": self._bytelist_feature_fromMatrix(feat0),
                    "cat": self._bytelist_feature_fromMatrix(feat1),
                    "label": self._float_feature(label)
                }))
        return insExample.SerializeToString()

    def _bytelist_feature_fromMatrix(self, temp):
        value = tf.io.serialize_tensor(temp)  # need to parse self when loading
        return Feature(bytes_list=BytesList(value=[value.numpy()]))

    def _float_feature(self, value):
        return Feature(float_list=FloatList(value=[value]))

