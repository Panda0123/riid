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
import numpy as np
import pickle as pkl
import joblib as jb
import matplotlib.pyplot as plt

import utils


def main():
    bS = 64
    bufferSize = 100

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # LOAD DATASET
    trainFP, validFP = utils.getDataPath()
    print("TrainFP:", trainFP)
    padIdx = -1
    sosIdx = -2
    nOov = 2  # for padIdx and sos token(for trg)
    nGPU = torch.cuda.device_count()
    nWorkers = 4 * nGPU if nGPU > 0 else 1
    print(nWorkers)

    dataLoader = partial(utils.getDataLoader,
                         bufferSize=bufferSize,
                         sosIdx=sosIdx,
                         nOov=nOov,
                         bS=bS,
                         nWorkers=nWorkers)
    trainLoader = dataLoader(filePath=trainFP)
    # valid dataset has 9,072 batches when bS=64
    # validLoader = dataLoader(filePath=validFP)

    # INITIALIZE MODEL
    lr = 1e-1
    model = utils.getModel(padIdx, nOov, device).to(device)
    lossFn = nn.CrossEntropyLoss(ignore_index=padIdx+1)  # padIdx at 0
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    lrsLog, losses = utils.findLearningRate(trainLoader,
                                  model,
                                  lossFn,
                                  opt,
                                  nOov,
                                  device,
                                  initLr=1e-8,
                                  maxLr=10.,
                                  beta=0.98)
    return lrsLog, losses

if __name__ == "__main__":
    lrsLog, losses = main()
    plt.plot(lrsLog[10:-5], losses[10:-5])
    plt.show()
    jb.dump(np.array(lrsLog), "saintlrlog.job")
    jb.dump(np.array(losses), "saintLosses.job")
