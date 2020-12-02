import pandas as pd
import numpy as np
from preprocessing import PreprocessV1, PreprocessV1ANN
import config
import utils
from tuning import optimizeXGB

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
                      nrows=10**3,
                      usecols=data_types_dict.keys(),
                      dtype=data_types_dict)
questionDf = pd.read_csv(config.QUESTION_PATH_CSV)
trainDf = trainDf[trainDf["answered_correctly"] != -1]

# SPLIT DATA
validDf = trainDf.groupby("user_id").tail(4)
trainDf = trainDf[~trainDf.index.isin(validDf.index)]

# PREPROCESS DATA
pipeline = PreprocessV1()
pipeline.fit(trainDf, questionDf)
trainDf = pipeline.transform(trainDf)
validDf = pipeline.transform(validDf)
print(trainDf)

# CREATE DATASET
# datasetCreator = utils.DatasetCreator()
# df_train = datasetCreator.transform(df_train)
# for temp in df_train:
#     print(temp[0].shape)
#     break

# print(df_train)
# OPTIMIZE XGB
# study = optimizeXGB(df_train, df_val, n_trials=5)
# print(study.best_params)
# print(study.best_value)
