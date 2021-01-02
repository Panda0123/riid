"""
Contains class for Stateless RNN models.
Such as StatelessRNN, StatelessRNNCuDNN (Running in CuDNN GPU)
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class StatelessRNN(tf.keras.Model):
    """Stateless Recurrent Neural Network using
        Gated Recurrent Unit as memory cell.
    """

    def __init__(self,
                 nUnits=104,
                 nLayers=2,
                 activation="tanh",
                 embLayer=None,
                 dropout=0.2,
                 recurrent_dropout=0.2,
                 **kwargs):
        """
            args:
                nUnits - number of units for each memory cell
                nLayers - number of rnn layers,
                activation - the activation (default = tanh)
                embaLayer - the embedding layer to be used for categorical feat
                dropout - the drop out rate for units
                recurrent_dropout - drop out rate for state
        """

        super(StatelessRNN, self).__init__(**kwargs)
        self.embLayer = embLayer
        self.activation = activation
        self.nUnits = nUnits
        self.nLayers = nLayers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        # Hidden layers
        self.rnnLayers = []
        for _ in range(nLayers - 1):
            self.rnnLayers.append(layers.GRU(nUnits,
                                             activation=keras.activations.get(
                                                 activation),
                                             return_sequences=True,
                                             dropout=self.dropout,
                                             recurrent_dropout=\
                                             self.recurrent_dropout))

        # last rnn layer does not return sequences
        self.rnnLayers.append(layers.GRU(nUnits,
                                         activation=keras.activations.get(
                                              activation),
                                         dropout=self.dropout,
                                         recurrent_dropout=\
                                         recurrent_dropout))

        self.outLayer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        embOutput = self.embLayer(inputs[1])  # index 1 are the categorical
        # concatenate output of embedding layer and the continuous features
        output = layers.concatenate([inputs[0], embOutput], axis=-1)
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

    def model(self):
        input0 = keras.Input(shape=(5, 4))
        input1 = keras.Input(shape=(5, 5))
        return keras.Model(
            inputs=[input0, input1],
            outpu=self.call((input0, input1))
        )


class StatelessRNNCuDNN(tf.keras.Model):
    """Works with GPU (CuDNN)"""

    def __init__(self,
                 nUnits=104,
                 nLayers=2,
                 activation="tanh",
                 nSteps=10,
                 embLayer=None,
                 **kwargs):
        """
            args:
                nUnits - number of units for each memory cell
                nLayers - number of rnn layers,
                activation - the activation (default = tanh)
                nSteps - number of time step (i.e., the windows length)
                embaLayer - the embedding layer to be used for categorical feat
        """
        super(StatelessRNNCuDNN, self).__init__(**kwargs)
        self.embLayer = embLayer
        self.activation = activation
        self.nUnits = nUnits
        self.nLayers = nLayers
        self.nSteps = nSteps

        # Hidden layers
        self.rnnLayers = []
        for _ in range(nLayers - 1):
            self.rnnLayers.append(layers.GRU(nUnits,
                                             dropout=0.2,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_dropout=0))

        self.rnnLayers.append(layers.GRU(nUnits,
                                         dropout=0.2,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_dropout=0))
        self.lambLayer = layers.Lambda(
                                lambda input_: input_[:, self.nSteps - 1, :])
        self.outLayer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        embOutput = self.embLayer(inputs[1])
        output = layers.concatenate([inputs[0], embOutput], axis=-1)
        output = self.rnnLayers[0](output)

        # index 0 cause we want only the first sequence
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
            "nSteps": self.nSteps
        }
        return config

    def model(self):
        input0 = keras.Input(shape=(5, 4))
        input1 = keras.Input(shape=(5, 5))
        return keras.Model(
            inputs=[input0, input1],
            outpu=self.call((input0, input1)))
