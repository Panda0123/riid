import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys


class PositionEncoder(nn.Module):
    def __init__(self, maxStep, embDim, device):
        super(PositionEncoder, self).__init__()
        self.encoder = torch.empty(maxStep, embDim, device=device)
        for i in range(embDim // 2):
            self.encoder[:, i*2] = torch.tensor([
                # sin(p / 10,000^(2i/d))
                np.sin((p / (10000**(2*i/embDim)))) for p in range(maxStep)])
            self.encoder[:, i*2+1] = torch.tensor([
                np.cos((p/10000**(2*i/embDim))) for p in range(maxStep)
            ])
        self.register_buffer("eklash", self.encoder)

    def forward(self, x):
        # x shape: seqLen, bS, embDim
        return self.encoder[:x.shape[0], :x.shape[-1]]


class LayerNormBlock(nn.Module):
    def __init__(self, nLayers, embDim):
        super(LayerNormBlock, self).__init__()
        self.nLayers = nLayers
        self.layers = nn.ModuleList(
            [nn.LayerNorm(embDim) for _ in range(nLayers)])

    def forward(self, *inputs):
        assert len(inputs) == self.nLayers, "length of inputs != # layers"
        outputs = []
        for x, layer in zip(inputs, self.layers):
            outputs.append(layer(x))
        return outputs


class TransformerBlock(nn.Module):
    def __init__(self,
                 embDim,
                 nHeads,
                 dropout,
                 expansion,
                 isEncoder,
                 device):
        super(TransformerBlock, self).__init__()
        self.isEncoder = isEncoder
        if self.isEncoder:
            self.norm1 = nn.LayerNorm(embDim)
        else:
            self.norm1 = LayerNormBlock(2, embDim)

        self.multihead = nn.MultiheadAttention(
            embDim, nHeads, dropout, bias=False).to(device)
        self.dropoutLayer = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embDim)
        self.ff = nn.Sequential(
            nn.Linear(embDim, expansion*embDim),
            nn.ReLU(),
            nn.Linear(expansion*embDim, embDim))

    def forward(self, values, keys, queries, mask=None):
        if self.isEncoder:
            values = self.norm1(values)
            outs, _ = self.multihead(
                values, values, values,
                need_weights=False, attn_mask=mask)
        else:
            values, queriesNormed = self.norm1(values, queries)
            outs, _ = self.multihead(
                values, values, queriesNormed,
                need_weights=False, attn_mask=mask)
        outs = self.dropoutLayer(outs + queries)
        ffOuts = self.ff(self.norm2(outs))
        return self.dropoutLayer(ffOuts + outs)


class Encoder(nn.Module):
    def __init__(self,
                 maxStep,
                 embDim,
                 padIdx,
                 nHeads,
                 dropout,
                 expansion,
                 nLayers,
                 maxExUnique,
                 maxPartUnique,
                 device):
        super(Encoder, self).__init__()
        self.exerEmbedding = nn.Embedding(maxExUnique, embDim,
                                          padding_idx=padIdx)
        self.partEmbedding = nn.Embedding(maxPartUnique, embDim,
                                          padding_idx=padIdx)
        self.posEncoding = PositionEncoder(maxStep, embDim, device)
        self.dropoutLayer = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
                TransformerBlock(embDim=embDim, nHeads=nHeads, dropout=dropout,
                                 expansion=expansion, isEncoder=True,
                                 device=device) for _ in range(nLayers)])

    def forward(self, exercise, part, mask):
        outs = self.exerEmbedding(exercise) + self.partEmbedding(part)
        # print(outs.shape)
        # print(self.posEncoding(outs).shape)
        outs = self.dropoutLayer(
            outs.transpose(0, 1) + self.posEncoding(outs)).transpose(0, 1)
        for layer in self.layers:
            outs = layer(outs, outs, outs, mask)
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, maxStep, embDim, nHeads, dropout, expansion, device):
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embDim)
        self.multiHead = nn.MultiheadAttention(
            embDim, nHeads, dropout, bias=False)
        self.dropoutLayer = nn.Dropout(dropout)
        self.transBlock = TransformerBlock(embDim, nHeads,
                                           dropout, expansion,
                                           isEncoder=False, device=device)

    def forward(self, x, values, keys, mask):
        queries = self.norm1(x)
        queries, _ = self.multiHead(queries, queries, queries,
                                    need_weights=False, attn_mask=mask)
        queries = self.dropoutLayer(x + queries)
        outs = self.transBlock(values, keys, queries, mask)
        return outs


class Decoder(nn.Module):
    def __init__(self, maxStep, embDim, padIdx,
                 nHeads, dropout, expansion, nLayers,
                 maxCorUnique, maxETUnique, maxLTUnique,
                 device):
        super(Decoder, self).__init__()
        self.corEmb = nn.Embedding(maxCorUnique, embDim, padding_idx=padIdx)
        self.etEmb = nn.Embedding(maxETUnique, embDim, padding_idx=padIdx)
        self.ltEmb = nn.Embedding(maxLTUnique, embDim, padding_idx=padIdx)
        self.posEnc = PositionEncoder(maxStep, embDim, device)
        self.dropoutLayer = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(maxStep, embDim, nHeads, dropout, expansion, device)
            for _ in range(nLayers)])

    def forward(self, elapsedTime, lagTime, cor, values, keys, mask):
        outs = self.etEmb(elapsedTime) + self.ltEmb(lagTime) + self.corEmb(cor)
        outs = self.dropoutLayer(
            outs.transpose(0, 1) + self.posEnc(outs)).transpose(0, 1)
        for layer in self.layers:
            outs = layer(values, keys, outs, mask)
        return outs


class Tutturu(nn.Module):
    def __init__(self, maxStep, embDim,  padIdx,
                 dropout, expansion,
                 nHeadsEncoder, nHeadsDecoder,
                 nEncoderLayers, nDecoderLayers,
                 maxPartUnique, maxExUnique,
                 maxCorUnique, maxETUnique, maxLTUnique,
                 maxTarget, device):
        super(Tutturu, self).__init__()
        self.device = device
        self.encoder = Encoder(maxStep, embDim, padIdx, nHeadsEncoder, dropout,
                               expansion, nEncoderLayers, maxExUnique,
                               maxPartUnique, device)
        self.decoder = Decoder(maxStep, embDim, padIdx, nHeadsDecoder, dropout,
                               expansion, nDecoderLayers, maxCorUnique,
                               maxETUnique, maxLTUnique, device)
        self.fc = nn.Linear(embDim, maxTarget)

    def mkMask(self, mat):
        seqLen, N = mat.shape
        mask = torch.tril(torch.ones(seqLen, seqLen))
        return mask.to(self.device)

    def forward(self, src, trg):
        # src: (exerciseID, part)
        # trg: (elapsedTime, lagTime, correctness[trg])
        # exserciseID: [seqLen, N] same with others
        mask = self.mkMask(src[0])
        encOuts = self.encoder(*src, mask)
        outs = self.decoder(*trg, encOuts, encOuts, mask)
        return self.fc(outs)
