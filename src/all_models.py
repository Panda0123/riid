from xgboost import XGBClassifier
import tensorflow as tf
import numpy as np

class SingleEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, values, nOov=1, **kwargs):
        super(SingleEmbeddingLayer, self).__init__(**kwargs)
        self.nOov = nOov
        nUniqueVal = len(values)
        indices = np.arange(nUniqueVal, dtype=np.int64)
        tableInit = tf.lookup.KeyValueTensorInitializer(values, indices)
        vocabSize = nUniqueVal + self.nOov
        self.table = tf.lookup.StaticVocabularyTable(tableInit, self.nOov)
        self.lambdaLayer = tf.keras.layers.Lambda(
            lambda inp: self.table.lookup(inp))
        self.embLayer = tf.keras.layers.Embedding(
            input_dim=vocabSize,
            output_dim=int(min(np.ceil(vocabSize / 2), 50)),
            trainable=True)

    def call(self, X):
        X = self.lambdaLayer(X)
        X = self.embLayer(X)
        return X

    def get_config(self):
        config = super(SingleEmbeddingLayer, self).get_config()
        config.update({"nOov": self.nOov})
        return config

class CatEmbeddingLayers(tf.keras.layers.Layer):

    def __init__(self, nOov=1, **kwargs):
        super(CatEmbeddingLayers, self).__init__(**kwargs)
        self.nOov = nOov
        self.batchNorm0 = tf.keras.layers.BatchNormalization()
        self.hidden1 = tf.keras.layers.Dense(150,
                                             activation="elu",
                                             kernel_initializer="he_normal")  # 157 embedding dimension
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(100,
                                             activation="elu",
                                             kernel_initializer="he_normal")
        self.batchNorm2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        outputEmb = []
        # transform each categories to their corresponding embedding
        for i in range(len(self.catValues)):
            output = self.catEmbLayers[i](X[:, :, i])
            outputEmb.append(output)
        output = tf.keras.layers.concatenate(outputEmb, axis=-1)
        output = self.batchNorm0(output)
        output = self.hidden1(output)
        output = self.batchNorm1(output)
        output = self.hidden2(output)
        output = self.batchNorm2(output)
        return output


    def adapt(self, df):
        # For Categorical features
        # question_id contain 13,523 unique question
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


class StatelessRNN(tf.keras.Model):
    def __init__(self, nUnits=104, nLayers=2,
                 activation="tanh", embLayer=None, **kwargs):
        super().__init__(**kwargs)
        self.embLayer = embLayer
        self.activation = activation
        self.nUnits = nUnits
        self.nLayers = nLayers
        # Hidden layers
        self.rnnLayers = []
        for i in range(nLayers - 1):
            self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                      activation=tf.keras.activations.get(
                                                          activation),
                                                      return_sequences=True,
                                                      dropout=0.2,
                                                      recurrent_dropout=0.2))

        self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                  activation=tf.keras.activations.get(
                                                      activation),
                                                  dropout=0.2,
                                                  recurrent_dropout=0.2))

        self.outLayer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        embOutput = self.embLayer(inputs[1])
        output = tf.keras.layers.concatenate([inputs[0], embOutput], axis=-1)
        for layer in self.rnnLayers:
            output = layer(output)
        output = self.outLayer(output)
        return output

    def get_config(self):
        config = {
            "activation": self.activation,
            "nUnits": self.nUnits,
            "nLayers": self.nLayers,
        }
        return config


class StatelessRNNCuDNN(tf.keras.Model):

    def __init__(self, nUnits=104, nLayers=2,
                 activation="tanh", nSteps=10, embLayer=None, **kwargs):
        super().__init__(**kwargs)
        self.embLayer = embLayer
        self.activation = activation
        self.nUnits = nUnits
        self.nLayers = nLayers
        self.nSteps = nSteps

        # Hidden layers
        self.rnnLayers = []
        for i in range(nLayers - 1):
            self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                      dropout=0.2,
                                                      return_sequences=True,
                                                      return_state=True,
                                                      recurrent_dropout=0,
                                                      ))

        self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                  dropout=0.2,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_dropout=0))
        self.lambLayer = tf.keras.layers.Lambda(
            lambda input_: input_[:, self.nSteps - 1, :])
        self.outLayer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        embOutput = self.embLayer(inputs[1])
        output = tf.keras.layers.concatenate([inputs[0], embOutput], axis=-1)
        output = self.rnnLayers[0](output)

        for layer in self.rnnLayers[1:]:
            output = layer(output[0])

        output = self.lambLayer(output[0])
        output = self.outLayer(output)
        return output

    def get_config(self):
        config = {
            "activation": self.activation,
            "nUnits": self.nUnits,
            "nLayers": self.nLayers,
        }
        return config
