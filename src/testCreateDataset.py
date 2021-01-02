import pandas as pd
import config
from preprocessing.stateless_rnn import PreprocessV3RNN
from utils.stateless_rnn.dataset_creator import DatasetCreatorV3
import joblib as jb


pd.options.mode.chained_assignment = None  # default='warn'

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
# get the last 5 interaction of user
testDf = trainDf.groupby("user_id").tail(5)
trainDf = trainDf[trainDf.index.isin(testDf.index)]
validDf = trainDf.groupby("user_id").tail(3)

# PREPROCESS DATA
ytrain = trainDf.answered_correctly
yvalid = validDf.answered_correctly
ytest = testDf.answered_correctly

prepTrans = PreprocessV3RNN()
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
nCat = 5  # number of categorical features
batchSize = 32
epochs = 10
nSteps = 5

# BUILD THE DATASETCREATOR
datasetCreator = DatasetCreatorV3(
    nCon=4, n_steps=nSteps, batch_size=batchSize)
datasetCreator.createTrain(
    trainDf,
    config.STATELESS_TRAIN_PATH_TFRECORD_ROOT,
    nShards=5,
    start=0,
    end=5)

datasetCreator.createValid(
    validDf,
    config.STATELESS_VALID_PATH_TFRECORD_ROOT,
    nShards=5,
    start=5,
    end=10)

jb.dump(prepTrans, config.STATELESS_PREPPROCESSING_PATH)
