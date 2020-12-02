import pandas as pd
import tensorflow as tf
import numpy as np
import config
from models.stateless_rnn.stateless_rnn_models import StatelessRNNCuDNN
from models.stateless_rnn.stateless_rnn_utils import CatEmbeddingLayers
import preprocessing
import utils

data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

# LOAD DATA
trainDf = pd.read_csv(config.TRAIN_PATH_CSV,
                      nrows=10**5,
                      usecols=data_types_dict.keys(),
                      dtype=data_types_dict)
questionDf = pd.read_csv(config.QUESTION_PATH_CSV)

# get the question only
trainDf = trainDf[trainDf.answered_correctly != -1]

# SPLIT TRAIN,TEST,VALID
testDf = trainDf.groupby("user_id").tail(
    5)  # get the last 5 interaction of user
trainDf = trainDf[trainDf.index.isin(testDf.index)]
validDf = trainDf.groupby("user_id").tail(3)

# PREPROCESS DATA
ytrain = trainDf.answered_correctly
yvalid = validDf.answered_correctly
ytest = testDf.answered_correctly

prepTrans = preprocessing.PreprocessV3RNN()
prepTrans.fit(trainDf, questionDf)
trainDf = prepTrans.transform(trainDf)
testDf = prepTrans.transform(testDf)
validDf = prepTrans.transform(validDf)

trainDf["answered_correctly"] = ytrain
validDf["answered_correctly"] = yvalid
testDf["answered_correctly"] = 1
questionDf = prepTrans.questionDf

# CONFIGURATIONS
nCon = 4  # number of continuous features
batchSize = 32
epochs = 10
nSteps = 5

# BUILD THE DATASETCREATOR
datasetCreator = utils.DatasetCreatorV3(
    nCon=4, n_steps=nSteps, batch_size=batchSize)
datasetCreator.createTrain(
    trainDf, config.STATELESS_TRAIN_PATH_TFRECORD_ROOT, nShards=5, start=0, end=5)
datasetCreator.createValid(
    validDf, config.STATELESS_VALID_PATH_TFRECORD_ROOT, nShards=5, start=5, end=10)
testPred = datasetCreator.createPredict(testDf)

# BUILD THE DATALOADER
trainLoader = utils.DataLoaderStatelessRNN(
    config.STATELESS_TRAIN_PATH_TFRECORD_ROOT,
    nSteps=nSteps,
    name="train",
    nCon=nCon
)
validLoader = utils.DataLoaderStatelessRNN(
    config.STATELESS_VALID_PATH_TFRECORD_ROOT,
    nSteps=nSteps,
    name="valid",
    nCon=nCon
)
trainDataset = trainLoader.loadDataset(
    shuffle_buffer_size=1000, batch_size=batchSize)
validDataset = validLoader.loadDataset(
    shuffle_buffer_size=1000, batch_size=batchSize)

# MODEL
embLayer = CatEmbeddingLayers()
embLayer.adapt(questionDf)
model = StatelessRNNCuDNN(nSteps=nSteps, embLayer=embLayer)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=1e-2),
    metrics=[tf.keras.metrics.AUC(), "accuracy"],
    run_eagerly=False,
)

# callbacks
alwaysCb = tf.keras.callbacks.ModelCheckpoint(
    config.STATELESS_MODEL_ALWAYS_PATH, save_freq="epoch", save_weights_only=True)
bestCb = tf.keras.callbacks.ModelCheckpoint(
    config.STATELESS_MODEL_ALWAYS_PATH, save_freq="epoch", save_weights_only=True, mode="maximize", monitor="val_auc")

model.fit(trainDataset, epochs=epochs,
          validation_data=validDataset, callbacks=[alwaysCb, bestCb])


model.evaluate(testPred, ytest)
# try:
# model.evaluate((testPred, ytest))
# except as ex:
# print(ex)
# print(testPred[0].shape)
# print(testPred[1].shape)
# print(len(ytest))
