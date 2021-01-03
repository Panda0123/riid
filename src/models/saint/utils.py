import torch
import tfrecord
import pickle as pkl
from functools import partial
import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import sys


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
        src= (sample[0][0].to(device), sample[0][1].to(device))
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
    return loss.item(), auc


def saveCkpt(model, opt, loss, epoch, fileName, scheduler=None):
    dctSave = {
        "modelStateDct": model.state_dict(),
        "optStateDct": opt.state_dict(),
        "loss": loss,
        "epoch": epoch,
    }
    if scheduler is not None:
        dctSave["schedulerStateDct"] = scheduler.state_dict()

    torch.save(dctSave, fileName)

def loadCkpt(fileName, model, opt, scheduler=None):
    loadedDct = torch.load(fileName)
    initDct = {
        "model": model,
        "opt": opt,
        "loss": loadedDct["loss"],
        "epoch": loadedDct["epoch"]
    }
    initDct["model"].load_state_dict(loadedDct["modelStateDct"])
    initDct["opt"].load_state_dict(loadedDct["optStateDct"])

    if scheduler is not None:
        assert len(loadedDct) > 4, "No scheduler in file"
        initDct["scheduler"] = scheduler
        initDct["scheduler"].load_state_dict(loadedDct["schedulerStateDct"])
    return initDct
