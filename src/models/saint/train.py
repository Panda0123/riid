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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from .transformer import Tutturu
from . import utils


def run(bufferSize, logInterval, writerPath=None,
        alwaysPath=None, bestPath=None, monitor=None):

    if monitor == "loss":
        direction = "minimize"
        bestMetric = float("inf")
    elif monitor == "auc":
        direction = "maximize"
        bestMetric = float("-inf")
    else:
        raise ValueError(f"monitor is {monitor} must be either loss or auc")
    
    if writerPath is not None:
        writer = SummaryWriter(writerPath)

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset parameters
    padIdx = -1
    sosIdx = -2
    nOov = 2  # for padIdx and sos token(for trg)
    nGPU = torch.cuda.device_count()
    nWorkers = 4 * nGPU if nGPU > 0 else 1

    # trainig parameters
    lr = 1e-3  # still the best
    nEpochs = 30
    bS = 64
    if bS == 64:
        trainNBS = 36056  # valid dataset has 36,056 batches when bS=64
        validNBS = 9072  # valid dataset has 9,072 batches when bS=64
    totalTrainSteps = bS * trainNBS

    # LOAD DATASET
    trainFP, validFP = utils.getDataPath()
    dataLoader = partial(utils.getDataLoader,
                         bufferSize=bufferSize,
                         sosIdx=sosIdx,
                         nOov=nOov,
                         bS=bS,
                         nWorkers=nWorkers)
    print("TrainFP:", trainFP)
    print("ValidFP", validFP)
    sys.exit()
    trainLoader = dataLoader(filePath=trainFP)
    validLoader = dataLoader(filePath=validFP)


    # INITIALIZE MODEL
    model = utils.getModel(padIdx, nOov, device).to(device)
    lossFn = nn.CrossEntropyLoss(ignore_index=padIdx+1)  # padIdx at 0
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(opt,
            max_lr=lr,
            total_steps=totalTrainSteps,
            anneal_strategy="linear")

    # load model
    if alwaysPath is not None and os.path.exists(alwaysPath):
        loadedDct = utils.loadCkpt(alwaysPath, model, opt, scheduler)
        model = loadedDct["model"].to(device)
        opt = loadedDct["opt"]
        scheduler = loadedDct["scheduler"]
        strtEpoch = loadedDct["epoch"]
        print(f"[Loaded Always Model] prev_loss:{loadedDct['loss']:.4f}",
              f"prev_auc:{loadedDct['auc']:.4f} curr_epoch:{strtEpoch+1}")
        loadedDct = utils.loadPrevMetrics(bestPath)
        bestMetric = loadedDct[monitor]
        print(f"[Loaded Best Model] best_loss:{loadedDct['loss']:.4f}",
              f"best_auc:{loadedDct['auc']:.4f}",
              f"best_epoch:{loadedDct['epoch']+1}")
    else:
        strtEpoch = 0

    padIdx += 1
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
            scheduler.step()

            # Batch Logs
            lossValue = loss.item()
            runningLoss += lossValue
            epochLoss += lossValue
            runningBatchTime += time.time() - batchTime
            if i % logInterval == (logInterval - 1):
                print("[%d %d] %.2fms/step Loss:%.4f" %
                      (epoch+1, i+1, runningBatchTime*1000/logInterval,
                       runningLoss / logInterval))
                if writerPath is not None:
                    writer.add_scalar("running_train_loss",
                            runningLoss / logInterval,
                            epoch * trainNBS + i)
                runningLoss = 0.
                runningBatchTime = 0.
            

        # Epoch Logs
        print("[%d] Total_Batch:%d Train_Loss:%.4f" %
              (epoch + 1, i + 1, epochLoss / (i + 1)),
              end=" ")
        model.eval()
        metricsDict = utils.evalEpoch(model, validLoader, lossFn,
                                      padIdx, nOov, device)
        print("Valid_Loss:%.4f Valid_AUC:%.4f" %
              (metricsDict["loss"], metricsDict["auc"]))
        if writerPath is not None:
            writer.add_scalar("train_loss",
                    epochLoss / (i + 1),
                    global_step=epoch)
            writer.add_scalar("valid_loss",
                    metricsDict["loss"],
                    global_step=epoch)
            writer.add_scalar("valid_auc",
                    metricsDict["auc"],
                    global_step=epoch)
        # SAVE MODEL
        if alwaysPath is not None:
            utils.saveCkpt(model, opt, metricsDict["loss"], metricsDict["auc"],
                           epoch + 1, alwaysPath, scheduler)

            if utils.shouldSaveBest(metricsDict[monitor],
                                    bestMetric, direction):
                utils.saveCkpt(model, opt,
                               metricsDict["loss"], metricsDict["auc"],
                               epoch + 1, bestPath, scheduler)
                bestMetric = metricsDict[monitor]
        writer.close()


if __name__ == "__main__":
    bS = 64  # args.bs
    nEpochs = 5  # args.nepochs
    bufferSize = 100  # args.buffer
    logInterval = 100  # args.log
    run(bufferSize, logInterval, writerPath)
