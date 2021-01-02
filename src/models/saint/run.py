from torch.utils.data import DataLoader
import utils
import os
import sys
import pandas as pd

home = os.path.expanduser("~")
filePath = home + "/data/kaggle/riid/ValidSample.csv"
# data_types_dict = {
#     'timestamp': 'int64',
#     'user_id': 'int32',
#     'question_id': 'int64',
#     'part': 'int8',
#     'answered_correctly': 'int8',
#     'lagTimeMinDig': 'int16',
#     'elapsedTimeSec': 'float32'
# }
# validDf = pd.read_csv(filePath, dtype=data_types_dict)

maxStep = 100
# utils.createTFRecord(
#     home + "/data/kaggle/riid/valid.tfrecord", validDf, maxStep)
filePath = home + "/data/kaggle/riid/valid.tfrecord"
print(filePath)

# sys.exit()
dataset = utils.loadIterDataset(filePath, bufferSize=100)
dataLoader = DataLoader(dataset, batch_size=10)
sample = next(iter(dataLoader))
print(sample)
