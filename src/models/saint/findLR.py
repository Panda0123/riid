# https://sgugger.github.io/the-1cycle-policy.html
# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
import os
from functools import partial
import platform
import sys
import time

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle as pkl

from .transformer import Tutturu
from . import utils


bS = 64
bufferSize = 100

# initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD DATASET
trainFP, validFP = utils.getDataPath()
padIdx = -1
sosIdx = -2
nOov = 2  # for padIdx and sos token(for trg)
nGPU = torch.cuda.device_count()
nWorkers = 4 * nGPU if nGPU > 0 else 1

dataLoader = partial(utils.getDataLoader,
                     bufferSize=bufferSize,
                     sosIdx=sosIdx,
                     nOov=nOov,
                     bS=bS,
                     nWorkers=nWorkers)
trainLoader = dataLoader(filePath=trainFP)
# valid dataset has 4,536 batches when bS=64
validLoader = dataLoader(filePath=validFP)

# INITIALIZE MODEL
model = utils.getModel(padIdx, nOov, device).to(device)
lossFn = nn.CrossEntropyLoss(ignore_index=padIdx+1)  # padIdx at 0
lr = 1e-3
opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

lrsLog, losses = utils.findLR(trainLoader,
                              model,
                              lossFn,
                              opt,
                              nOov,
                              device,
                              initLr=1e-8,
                              maxLr=10.,
                              beta=0.98)
dct = {"lrsLog": np.array(lrsLog), "losses": np.array(losses)}
pkl.dump(dct, "saintLR.job")
