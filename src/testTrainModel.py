import tensorflow as tf
import config
from models.stateless_rnn.stateless_rnn_models import StatelessRNNCuDNN
from models.stateless_rnn.embedding_layers import CatEmbeddingLayers
from utils.stateless_rnn.dataset_loader import \
                                        DatasetLoaderStatelessRNNBatch
import joblib as jb

# CONFIGURATIONS
nCon = 4  # number of continuous features
nCat = 5  # number of categorical features
batchSize = 32
epochs = 10
nSteps = 5


# IMPORT PREPROCESSING
prepTrans = jb.load(config.STATELESS_PREPPROCESSING_PATH)
questionDf = prepTrans.questionDf

# BUILD THE DATALOADER
trainLoader = DatasetLoaderStatelessRNNBatch(
    config.STATELESS_TRAIN_PATH_TFRECORD_ROOT,
    nSteps=nSteps,
    name="train",
    nCon=nCon,
    nCat=nCat
)

validLoader = DatasetLoaderStatelessRNNBatch(
    config.STATELESS_VALID_PATH_TFRECORD_ROOT,
    nSteps=nSteps,
    name="valid",
    nCon=nCon,
    nCat=nCat
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
            config.STATELESS_MODEL_ALWAYS_PATH,
            save_freq="epoch",
            save_weights_only=True)
bestCb = tf.keras.callbacks.ModelCheckpoint(
            config.STATELESS_MODEL_ALWAYS_PATH,
            save_freq="epoch",
            save_weights_only=True,
            mode="max",
            monitor="val_auc")

model.fit(trainDataset, epochs=epochs,
          validation_data=validDataset, callbacks=[alwaysCb, bestCb])
