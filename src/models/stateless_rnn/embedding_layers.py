"""
Contains Utilities for making a Statess RNN model.
Such as Embedding Layers for categorical features.
"""
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd


class SingleEmbeddingLayer(layers.Layer):
    """Embedding layer for a single categorical feature."""

    def __init__(self,
                 values,
                 nOov=1,
                 **kwargs):
        """ Embeding layer for a single categorical features.

            args:
                values - the categories/values
                nOov - number of out of vocabulary to allocate (default=1)
        """

        super(SingleEmbeddingLayer, self).__init__(**kwargs)
        self.nOov = nOov
        nUniqueVal = len(values)
        indices = tf.range(nUniqueVal, dtype=tf.int64)
        tableInit = tf.lookup.KeyValueTensorInitializer(values, indices)
        vocabSize = nUniqueVal + self.nOov
        self.table = tf.lookup.StaticVocabularyTable(tableInit, self.nOov)
        self.lambdaLayer = layers.Lambda(
                                    lambda inp: self.table.lookup(inp))
        self.embLayer = layers.Embedding(
                            input_dim=vocabSize,
                            output_dim=int(
                                min(tf.math.ceil(vocabSize / 2), 50)),
                            trainable=True)

    def call(self, X):
        X = self.lambdaLayer(X)
        X = self.embLayer(X)
        return X

    def get_config(self):
        config = super(SingleEmbeddingLayer, self).get_config()
        config.update({"nOov": self.nOov})
        return config


class CatEmbeddingLayers(layers.Layer):
    """Embedding Layer for Catagorical Features.
       If data contains multiple categorical features this class if for it.
       Just call its .adapt() method on the categorical features.

       After the embedding layers it is followed by batch
       normalization and dense layer.

       Make sure to call .adapt() first passing in the categorical features.
       It accept a dataframe from pandas.
    """

    def __init__(self,
                 nOov=1,
                 layersNUnits=[150, 100],
                 **kwargs):
        """Embedding Layer for Catagorical Features.
            args:
                nOov - number of out of vocabulary to allocate. (default=1)
                layersNUnits - a list that specifies the number of units for
                               each layer. (default=[150, 100])
        """
        super(CatEmbeddingLayers, self).__init__(**kwargs)
        self.nOov = nOov
        self.batchNorm0 = layers.BatchNormalization()

        # 157 embedding dimension
        self.hidden1 = layers.Dense(layersNUnits[0],
                                    activation="elu",
                                    kernel_initializer="he_normal")
        self.batchNorm1 = layers.BatchNormalization()
        self.hidden2 = layers.Dense(layersNUnits[1],
                                    activation="elu",
                                    kernel_initializer="he_normal")
        self.batchNorm2 = layers.BatchNormalization()

        self.concatLayer = layers.Concatenate(axis=-1)

    def call(self, X):
        outputEmb = []
        # transform each categories to their corresponding embedding
        for i in range(len(self.catValues)):
            output = self.catEmbLayers[i](X[:, :, i])
            outputEmb.append(output)
        output = self.concatLayer(outputEmb)
        output = self.batchNorm0(output)
        output = self.hidden1(output)
        output = self.batchNorm1(output)
        output = self.hidden2(output)
        output = self.batchNorm2(output)
        return output

    def adapt(self, df: pd.DataFrame):
        # For Categorical features
        # question_id contain 13,523 unique question for riid dataset
        catFeatures = ["question_id", "bundle_id",
                       "correct_answer", "part", "tags"]
        self.catValues = []
        for feat in catFeatures:
            self.catValues.append(df[feat].unique())
        self.catEmbLayers = []
        # PREPARING LOOK UP TABLE FOR EMBEDDING
        for i in range(len(catFeatures)):
            self.catEmbLayers.append(
                SingleEmbeddingLayer(self.catValues[i], nOov=self.nOov))

    def get_config(self):
        config = super(CatEmbeddingLayers, self).get_config()
        config.update({"nOov": self.nOov})
        return config
