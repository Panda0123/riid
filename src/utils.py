from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.train import Example, BytesList, FloatList, Feature, Features
import pandas as pd
import tensorflow as tf
import numpy as np
import os

def strat_split(df, target_name, random_state=42):
    spt = StratifiedShuffleSplit(n_splits=1, random_state=random_state)
    for train_ix, val_ix in spt.split(df, df[target_name]):
        df_train = df.iloc[train_ix]
        df_val = df.iloc[val_ix]
    return df_train, df_val

def time_split(df, n_records=5):
    df_val = pd.DataFrame()
    for i in range(n_records):
        last_record = df.drop_duplicates("user_id", keep="last")
        df = df[~df.index.isin(last_record.index)]
        df_val = df_val.append(last_record)
    return df, df_val


class DatasetCreatorV2:
    def __init__(self, n_steps=10, shift=1, batch_size=32):
        self.n_steps = n_steps
        self.shift = shift
        self.batch_size = batch_size
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
                self.instances[id_] = user.tail(self.n_steps - 1)
                dataset = self._createDataset(user)
                dataset = dataset.map(lambda instance: ((instance[:, :3], tf.cast(instance[:, 3:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
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
                user = grouped.get_group(id_)
                isNew = id_ not in self.instances.keys()
                dataset = self._createDataset(user, isNew=isNew)
                dataset = dataset.map(lambda instance: ((instance[:, :3], tf.cast(instance[:, 3:-2], tf.int16)),  # input -2 so user_id not includeget last time step's label
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
                feature0.append(userDf.values[:, :3].tolist())
                feature1.append(userDf.values[:, 3:-2].tolist())
            else:
                userDf = pd.DataFrame(np.zeros(
                    shape=(self.n_steps - 1, userDf.shape[0])), columns=userDf.index.values)
                userDf = userDf.append(userSer)
                feature0.append(userDf.values[:, :3].tolist())
                feature1.append(userDf.values[:, 3:-2].tolist())

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


class DatasetCreatorV3:

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


class DataLoaderStatelessRNN:
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
