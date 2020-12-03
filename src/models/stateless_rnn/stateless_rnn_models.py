"""
Contains class for Stateless RNN models.
Such as StatelessRNN, StatelessRNNCuDNN (Running in CuDNN GPU)
"""

import tensorflow as tf

class StatelessRNN(tf.keras.Model):
    """
        args: nUnits, nLayers(RNN layers), activation,
              dropout, recurrent_dropout
    """

    def __init__(self, nUnits=104, nLayers=2,
                 activation="tanh", embLayer=None,
                 dropout=0.2, recurrent_dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embLayer = embLayer
        self.activation = activation
        self.nUnits = nUnits
        self.nLayers = nLayers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        # Hidden layers
        self.rnnLayers = []
        for _ in range(nLayers - 1):
            self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                      activation=tf.keras.activations.get(
                                                          activation),
                                                      return_sequences=True,
                                                      dropout=self.dropout,
                                                      recurrent_dropout=\
                                                      self.recurrent_dropout))

        self.rnnLayers.append(tf.keras.layers.GRU(nUnits,
                                                  activation=tf.keras.activations.get(
                                                      activation),
                                                  dropout=self.dropout,
                                                  recurrent_dropout=\
                                                  recurrent_dropout))

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
    """Works with GPU (CuDNN)"""

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
        for _ in range(nLayers - 1):
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
