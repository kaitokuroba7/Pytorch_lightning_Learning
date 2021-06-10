#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: supervised.py
@time: 2021/6/10 15:49
"""
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl


class LightningClassifier(pl.LightningModule):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        best_loss = 0
        # probability distribution over labels
        logits = self.backbone(x)
        x = torch.log_softmax(x, dim=1)

        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    pass
