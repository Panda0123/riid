import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
import sys
import os
import time

from transformer import Tutturu
import utils

def getModel(padIdx, nOov, device):
    # maxTarget [padIdx, negative, postive]
    return Tutturu(maxStep=100, embDim=512, padIdx=padIdx+nOov,
                   dropout=0., expansion=4,
                   nHeadsEncoder=8, nHeadsDecoder=8,
                   nEncoderLayers=3, nDecoderLayers=3,
                   maxPartUnique=9, maxExUnique=13524,
                   maxCorUnique=4, maxETUnique=303, maxLTUnique=1443,
                   maxTarget=3, device=device)

def main():
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    home = os.path.expanduser("~")
    if device == "cpu":
        filePath = home + "/data/kaggle/riid/valid.tfrecord"
    else:
        filePath = home + "/dataLnk/kaggle/riid/valid.tfrecord"

    bufferSize = 100
    bS = 64
    nOov = 2  # for padIdx and sos token(for trg)
    padIdx = -1
    sosIdx = -2
    # valid dataset has 4,536 batches when bS=64
    dataIterator = utils.loadIterDataset(filePath,
                                         bufferSize=bufferSize,
                                         sosIdx=sosIdx,
                                         nOov=nOov)
    dataLoader = DataLoader(dataIterator,
                            batch_size=bS,
                            num_workers=2,
                            collate_fn=utils.collate_fn)

    # initialize model
    model = getModel(padIdx, nOov, device).to(device)
    lossFn = nn.CrossEntropyLoss(ignore_index=padIdx+1)  # padIdx at 0
    opt = optim.Adam(model.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt)
    nEpochs = 5

    sample = next(iter(dataLoader))
    print(sample[1][-1].shape)
    print(sample[0][-1].shape)
    # utils.overFit(sample, model, opt, lossFn, scheduler, padIdx, nOov, device)
    # lbls = trg[-1].reshape(-1)
    # outs shape: [seqLen, N, 1]
    # src, trg = next(iter(dataLoader))
    logInterval = 100
    padIdx += 1
    for epoch in range(nEpochs):
        model.train()
        runningLoss = 0.
        runningBatchTime = 0.
        epochLoss = 0.

        # src: (exerciseID, part)
        # trg: (elapsedTime, lagTime, correctness[trg])
        # exerciseID shape: [seqLen, N]
        for i, (src, trg) in enumerate(dataLoader):

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
            lossVal = loss.item()
            runningLoss += lossVal
            epochLoss += lossVal
            runningBatchTime += time.time() - batchTime
            if i % logInterval == (logInterval - 1):
                print("[%d %d] %.2fms/step Loss:%.4f" %
                      (epoch+1, i+1, runningBatchTime*1000/logInterval,
                       runningLoss / logInterval))
                runningLoss = 0.
                runningBatchTime = 0.
        scheduler.step(epochLoss / (i + 1))
        print("[%d] Total_Batch:%d Train_Loss:%.4f" %
              (epoch + 1, i + 1, epochLoss / (i + 1)),
              end=" ")
        model.eval()
        validLoss, validAUC = utils.evalEpoch(model, dataLoader, lossFn,
                                              padIdx, nOov, device)
        print("Valid_Loss:%.4f Valid_AUC:%.4f" % (validLoss, validAUC))

if __name__ == "__main__":
    main()
