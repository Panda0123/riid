import math
import sys
from functools import partial
import pickle as pkl
import os
import platform

import torch
import tfrecord
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import Tutturu


def shouldSaveBest(currMetric, bestMetric, direction="minimize"):
    """
    args:
        direction: if should save if < best or > best: default (minimize)
    """
    if direction == "minimize":
        if currMetric < bestMetric:
            return True
    elif direction == "maximize":
        if currMetric > bestMetric:
            return True
    else:
        raise ValueError(
            f"direction:{direction} must either be maximize or minimize")
    return False


def getModel(padIdx, nOov, device):
    # maxTarget [padIdx, negative, postive]
    return Tutturu(maxStep=100, embDim=512, padIdx=padIdx+nOov,
                   dropout=0., expansion=4,
                   nHeadsEncoder=8, nHeadsDecoder=8,
                   nEncoderLayers=3, nDecoderLayers=3,
                   maxPartUnique=9, maxExUnique=13524,
                   maxCorUnique=4, maxETUnique=303, maxLTUnique=1443,
                   maxTarget=3, device=device)


def getDataLoader(filePath, bufferSize,
                  sosIdx, nOov, bS, nWorkers):
    dataIter = loadIterDataset(filePath, bufferSize=bufferSize,
                                     sosIdx=sosIdx,
                                     nOov=nOov)
    return DataLoader(dataIter,
                      batch_size=bS,
                      num_workers=nWorkers,
                      collate_fn=collate_fn)


def getDataPath():
    home = os.path.expanduser("~")
    # identify if I'm using wsl2 or windows
    if platform.system() == "Windows":
        return (home + "/dataLnk/kaggle/riid/train.tfrecord",
                home + "/dataLnk/kaggle/riid/valid.tfrecord")
    else:
        return (home + "/data/kaggle/riid/valid.tfrecord",
                home + "/data/kaggle/riid/valid.tfrecord")


def splitDataframe(df, chunkSize=100):
    chunks = list()
    numberChunks = math.ceil(len(df) / chunkSize)
    for i in range(numberChunks):
        chunks.append(df[i*chunkSize:(i+1)*chunkSize])
    return chunks


def createTFRecord(fileName, df, maxStep):
    writer = tfrecord.TFRecordWriter(fileName)
    serialize = partial(pkl.dumps, protocol=0)
    ids = df.user_id.unique()
    for id_ in ids:
        sampleUser = df[df.user_id == id_]
        ltDf = []
        if len(sampleUser) < maxStep:
            shape = sampleUser.shape
            sampleUser = pd.concat(
                [sampleUser,
                 pd.DataFrame(np.full((maxStep - shape[0], shape[1]), -1),
                              columns=sampleUser.columns)], ignore_index=True)
            ltDf.append(sampleUser)
        elif len(sampleUser) > maxStep:
            ltDf = splitDataframe(sampleUser, maxStep)
            # check if last one is lacking
            if len(ltDf[-1]) < maxStep:
                shape = ltDf[-1].shape
                ltDf[-1] = pd.concat(
                    [ltDf[-1],
                     pd.DataFrame(np.full((maxStep - shape[0], shape[1]), -1),
                                  columns=sampleUser.columns)], ignore_index=True)
        else:
            ltDf.append(sampleUser)

        for user in ltDf:
            writer.write({
                "questID": (serialize(user.question_id.values), "byte"),
                "part": (serialize(user.part.values), "byte"),
                "elapsedTimeSec": (serialize(user.elapsedTimeSec.values), "byte"),
                "lagTimeMinDig": (serialize(user.lagTimeMinDig.values), "byte"),
                "answeredCorrectly": (serialize(user.answered_correctly.values), "byte")
            })
    writer.close()


def prepLoad(examples, sosIdx, nOov):
    questID = torch.tensor(
        pkl.loads(examples["questID"]),
        dtype=torch.long) + (nOov - 1)
    part = torch.tensor(
        pkl.loads(examples["part"]),
        dtype=torch.long) + (nOov - 1)
    elapsedTimeSec = torch.tensor(
        np.insert(pkl.loads(examples["elapsedTimeSec"]), 0, sosIdx),
        dtype=torch.long) + nOov
    lagTimeMinDig = torch.tensor(
        np.insert(pkl.loads(examples["lagTimeMinDig"]), 0, sosIdx),
        dtype=torch.long) + nOov
    answeredCorrectly = torch.tensor(
        np.insert(pkl.loads(examples["answeredCorrectly"]), 0, sosIdx),
        dtype=torch.long) + nOov
    return (questID, part), (elapsedTimeSec, lagTimeMinDig, answeredCorrectly)


def loadIterDataset(fileName, sosIdx, bufferSize=100, nOov=2):
    """
    args:
        fileName: filePath for the file
        sosIdx: index of sos token
        bufferSize: shuffle_queue_size
        nOov: number of out of vocabulary from the original vocabs

    """
    description = {
        "questID": "byte",
        "part": "byte",
        "elapsedTimeSec": "byte",
        "lagTimeMinDig": "byte",
        "answeredCorrectly": "byte"
    }
    prepProp = partial(prepLoad, sosIdx=sosIdx, nOov=nOov)

    dataset = tfrecord.torch.TFRecordDataset(fileName,
                                             index_path=None,
                                             description=description,
                                             shuffle_queue_size=bufferSize,
                                             transform=prepProp)
    return dataset


def collate_fn(data):
    """
    args:
        data: is a list whose shape is bS
            each sample is a tuple ((exerID, part),
            (elapsedTime, lagTime, correctness)
    """
    batchEID = []
    batchPart = []
    batchET = []
    batchLT = []
    batchCor = []

    for item in data:
        batchEID.append(item[0][0].reshape(-1, 1))
        batchPart.append(item[0][1].reshape(-1, 1))
        batchET.append(item[1][0].reshape(-1, 1))
        batchLT.append(item[1][1].reshape(-1, 1))
        batchCor.append(item[1][2].reshape(-1, 1))
    return ((torch.hstack(batchEID), torch.hstack(batchPart)),
            (torch.hstack(batchET), torch.hstack(batchLT),
             torch.hstack(batchCor)))


def overFit(sample, model, opt, lossFn, scheduler, padIdx, nOov, device):
    padIdx += 1
    lss = float("inf")
    while lss > 0.000005:
        src = (sample[0][0].to(device), sample[0][1].to(device))
        trg = (sample[1][0].to(device), sample[1][1].to(device),
               sample[1][2].to(device))
        with torch.no_grad():
            # remove sos token
            # decrement labels 0-padIdx, 1-negative, 2-positive
            lbls = trg[-1][1:].reshape(-1) - (nOov - 1)

        opt.zero_grad()
        # remove last response for trg
        trg = tuple(feat[:-1] for feat in trg)
        outs = model(src, trg)
        # reshape to [seqLen*bS, trgVocabSize]
        outs = outs.reshape(-1, outs.shape[-1])
        loss = lossFn(outs, lbls)
        loss.backward()
        opt.step()
        lss = loss.item()
        with torch.no_grad():
            # filterout paddings
            mask = (lbls != padIdx)
            # ignore padding
            auc = roc_auc_score(lbls[mask].cpu() - 1,
                                F.softmax(outs[mask, 1:], dim=-1)[:, -1].cpu())
            # torch.sigmoid(outs[mask, -1]))
        scheduler.step(lss)
        print(f"Loss:{lss:.2f} AUC:{auc:.2f}")
        # print(f"Loss:{lss:.2f}")


def evalEpoch(model, validLoader, lossFn, padIdx, nOov, device):
    yHat = []
    y = []
    with torch.no_grad():
        for i, (src, trg) in enumerate(validLoader):
            src = (src[0].to(device), src[1].to(device))
            trg = (trg[0].to(device), trg[1].to(device), trg[2].to(device))
            # remove sos token
            lbls = trg[-1][1:].reshape(-1) - (nOov - 1)
            # remove last value of response features
            trg = tuple(feat[:-1] for feat in trg)

            outs = model(src, trg)

            # reshape to [seqLen * bS, trgVocabSize]
            yHat.append(outs.reshape(-1, outs.shape[-1]))
            y.append(lbls)

        yHat = torch.cat(yHat)
        y = torch.cat(y)
        loss = lossFn(yHat, y)
        # filter out paddings
        mask = (y != padIdx)
        # ignore padding
        auc = roc_auc_score(y[mask].cpu() - 1,
                            F.softmax(yHat[mask, 1:], dim=-1)[:, -1].cpu())
    return {"loss": loss.item(), "auc": auc}


def saveCkpt(model, opt, loss, auc, epoch, fileName, scheduler=None):
    dctSave = {
        "modelStateDct": model.state_dict(),
        "optStateDct": opt.state_dict(),
        "loss": loss,
        "auc": auc,
        "epoch": epoch,
    }
    if scheduler is not None:
        dctSave["schedulerStateDct"] = scheduler.state_dict()

    torch.save(dctSave, fileName)


def loadPrevMetrics(fileName):
    loadedDct = torch.load(fileName)
    return {
        "loss": loadedDct["loss"],
        "auc": loadedDct["auc"],
        "epoch": loadedDct["epoch"]
    }


def loadCkpt(fileName, model, opt, scheduler=None):
    loadedDct = torch.load(fileName)
    initDct = {
        "model": model,
        "opt": opt,
        "loss": loadedDct["loss"],
        "epoch": loadedDct["epoch"],
        "auc": loadedDct["auc"]
    }
    initDct["model"].load_state_dict(loadedDct["modelStateDct"])
    initDct["opt"].load_state_dict(loadedDct["optStateDct"])

    if scheduler is not None:
        assert len(loadedDct) > 4, "No scheduler in file"
        initDct["scheduler"] = scheduler
        initDct["scheduler"].load_state_dict(loadedDct["schedulerStateDct"])
    return initDct


def findLearningRate(dataLoader,
           model,
           lossFn,
           opt,
           nOov,
           device,
           initLr=1e-8,
           maxLr=10.,
           beta=0.98):

    for i, data in enumerate(dataLoader):
        pass
    print(i)
    num = i
    mult = (maxLr / initLr) ** (1/num)
    lr = initLr
    opt.param_groups[0]['lr'] = lr
    avgLoss = 0.
    bestLoss = 0.
    batchNum = 0
    losses = []
    lrsLog = []
    for i, (src, trg) in enumerate(dataLoader):
        batchNum += 1
        src = (src[0].to(device), src[1].to(device))
        trg = (trg[0].to(device), trg[1].to(device), trg[2].to(device))
        with torch.no_grad():
            lbls = trg[-1][1:].reshape(-1) - (nOov - 1)
        trg = tuple(feat[:-1] for feat in trg)
        opt.zero_grad()
        outs = model(src, trg)

        outs = outs.reshape(-1, outs.shape[-1])
        loss = lossFn(outs, lbls)

        # Compute the smoothed loss
        # beta*avgLoss + (1-beta)*currLoss
        avgLoss = beta * avgLoss + (1 - beta) * loss.item()
        # avgLoss / (1 - beta^batchNum)
        smoothedLoss = avgLoss / (1 - beta**batchNum)
        # Stop if the loss is exploding
        if batchNum > 1 and smoothedLoss > 4 * bestLoss:
            return lrsLog, losses
        # Record the best loss
        if smoothedLoss < bestLoss or batchNum == 1:
            bestLoss = smoothedLoss
        # Store the values
        losses.append(smoothedLoss)
        lrsLog.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        opt.step()
        # Update the lr for the next step
        lr *= mult
        opt.param_groups[0]['lr'] = lr
    return lrsLog, losses
