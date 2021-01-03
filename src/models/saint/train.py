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

from .transformer import Tutturu
from . import utils


def run(bS, nEpochs, bufferSize, logInterval,
        alwaysPath=None, bestPath=None, monitor=None):
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if monitor == "loss":
        direction = "minimize"
        bestMetric = float("inf")
    elif monitor == "auc":
        direction = "maximize"
        bestMetric = float("-inf")
    else:
        raise ValueError(f"monitor is {monitor} must be either loss or auc")
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
    # TODO: change scheduler to 1cycle 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt)
    # load model
    if alwaysPath is not None and os.path.exists(alwaysPath):
        loadedDct = utils.loadCkpt(alwaysPath, model, opt, scheduler)
        model = loadedDct["models"].to(device)
        opt = loadedDct["opt"]
        scheduler = loadedDct["scheduler"]
        strtEpoch = loadedDct["epoch"]
        print(f"[Loaded Existing Model] prev_loss:{loadedDct['loss']:.4f}",
              f"prev_auc:{loadedDct['auc']:.4f} prev_epoch:{strtEpoch}")
        loadedDct = utils.loadPrevMetrics(bestPath)
        bestMetric = loadedDct[monitor]
        print(f"[Loaded Best Model] best_loss:{loadedDct['loss']:.4f}",
              f"best_auc:{loadedDct['auc']:.4f}",
              f"best_epoch:{loadedDct['epoch']}")
    else:
        strtEpoch = 0
        # TODO: write function for warm up start if first train

    padIdx += 1
    nEpochs += strtEpoch
    for epoch in range(strtEpoch, nEpochs):
        model.train()
        runningLoss = 0.
        runningBatchTime = 0.
        epochLoss = 0.

        # src: (exerciseID, part)
        # trg: (elapsedTime, lagTime, correctness[trg])
        # exerciseID shape: [seqLen, N]
        for i, (src, trg) in enumerate(trainLoader):

            # PREPARING DATA
            batchTime = time.time()
            src = (src[0].to(device), src[1].to(device))
            trg = (trg[0].to(device), trg[1].to(device), trg[2].to(device))
            with torch.no_grad():
                # remove sos token
                # decrement labels so 0-padIdx, 1-negative, 2-positive
                lbls = trg[-1][1:].reshape(-1) - (nOov - 1)
            # remove last value of response features
            trg = tuple(feat[:-1] for feat in trg)

            # FORWARD PASS
            opt.zero_grad()
            outs = model(src, trg)

            # reshape to [seqLen * bS, trgVocabSize]
            outs = outs.reshape(-1, outs.shape[-1])
            loss = lossFn(outs, lbls)
            # BACKWARD PASS
            loss.backward()
            opt.step()

            # Logs
            lossValue = loss.item()
            runningLoss += lossValue
            epochLoss += lossValue
            runningBatchTime += time.time() - batchTime
            if i % logInterval == (logInterval - 1):
                print("[%d %d] %.2fms/step Loss:%.4f" %
                      (epoch+1, i+1, runningBatchTime*1000/logInterval,
                       runningLoss / logInterval))
                runningLoss = 0.
                runningBatchTime = 0.
        print("[%d] Total_Batch:%d Train_Loss:%.4f" %
              (epoch + 1, i + 1, epochLoss / (i + 1)),
              end=" ")
        model.eval()
        metricsDict = utils.evalEpoch(model, validLoader, lossFn,
                                      padIdx, nOov, device)
        print("Valid_Loss:%.4f Valid_AUC:%.4f" %
              (metricsDict["loss"], metricsDict["auc"]))
        scheduler.step(epochLoss / (i + 1))
        # SAVE MODEL
        if alwaysPath is not None:
            utils.saveCkpt(model, opt, metricsDict["loss"], metricsDict["auc"],
                           epoch, alwaysPath, scheduler)

            if utils.shouldSaveBest(metricsDict[monitor],
                                    bestMetric, direction):
                utils.saveCkpt(model, opt,
                               metricsDict["loss"], metricsDict["auc"],
                               epoch, bestPath, scheduler)
                bestMetric = metricsDict[monitor]


if __name__ == "__main__":
    bS = 64  # args.bs
    nEpochs = 5  # args.nepochs
    bufferSize = 100  # args.buffer
    logInterval = 100  # args.log
    run(bS, nEpochs, bufferSize, logInterval)
