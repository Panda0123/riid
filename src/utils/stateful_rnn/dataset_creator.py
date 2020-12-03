from tensorflow.train import Example, BytesList, FloatList, Feature, Features
import pandas as pd
import tensorflow as tf
import numpy as np
import os

class DatasetCreatorStateFulRNN:

    def __init__(self, nCon, n_steps=10):
        self.n_steps = n_steps
        self.shift = n_steps
        # self.batch_size = batch_size batchSize == nInstances
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

    def createValid(self, df, rootFilePath, nShards):
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
