#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: data_model.py
@time: 2021/6/10 16:32
"""
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):

    def setup(self, stage: str = None):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = mnist_test

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64, num_workers=6)


if __name__ == "__main__":
    pass
