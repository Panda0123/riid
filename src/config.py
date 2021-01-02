from pathlib import Path

PROJECT_PATH = str(Path.home()) + "/source/project/kaggle/riid/"

TRAIN_PATH_CSV = PROJECT_PATH + "inputs/train.csv"
TEST_PATH_CSV = PROJECT_PATH + "inputs/test.csv"
QUESTION_PATH_CSV = PROJECT_PATH + "inputs/questions.csv"


# ==== STATELESS ===
STATELESS_PREPPROCESSING_PATH =\
    PROJECT_PATH + "utils/DNN/prepTransStatelessRNN.joblib"

STATELESS_TRAIN_PATH_TFRECORD_ROOT =\
    PROJECT_PATH + "datasets/stateless_rnn/test/"
STATELESS_VALID_PATH_TFRECORD_ROOT =\
    PROJECT_PATH + "datasets/stateless_rnn/valid/"

STATELESS_MODEL_ALWAYS_PATH =\
    PROJECT_PATH + "/models/DNN/stateless_rnn/always/"
STATELESS_MODEL_BEST_PATH =\
    PROJECT_PATH + "/models/DNN/stateless_rnn/best/"

