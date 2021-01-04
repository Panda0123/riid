import os
import platform

home = os.path.expanduser("~")
if platform.system() == "Windows":
    PROJECT_PATH = home + "/sourceLnk/project/kaggle/riid"
else:
    PROJECT_PATH = home + "/source/project/kaggle/riid"

TRAIN_PATH_CSV = PROJECT_PATH + "inputs/train.csv"
TEST_PATH_CSV = PROJECT_PATH + "inputs/test.csv"
QUESTION_PATH_CSV = PROJECT_PATH + "inputs/questions.csv"


# === STATELESS ===
STATELESS_PREPPROCESSING_PATH =\
    PROJECT_PATH + "utils/DNN/prepTransStatelessRNN.joblib"

# STATELESS_TRAIN_PATH_TFRECORD_ROOT =\

STATELESS_VALID_PATH_TFRECORD_ROOT =\
    PROJECT_PATH + "datasets/stateless_rnn/valid/"

STATELESS_MODEL_ALWAYS_PATH =\
    PROJECT_PATH + "/models/DNN/stateless_rnn/always/"
STATELESS_MODEL_BEST_PATH =\
    PROJECT_PATH + "/models/DNN/stateless_rnn/best/"

# === SAINT ===
SAINT_MODEL_ALWAYS_PATH = PROJECT_PATH + "/models/DNN/saint/alwaysSaint.pth"
SAINT_MODEL_BEST_PATH = PROJECT_PATH + "/models/DNN/saint/bestSaint.pth"
SAINT_LOG_PATH = PROJECT_PATH + "/logs/saint/"
