import pytorch_lightning as pl

from model.linear import Backbone
from tasks.supervised import LightningClassifier
from utils.data.data_model import MNISTDataModule


if __name__ == '__main__':
    model = Backbone()
    classifier = LightningClassifier(model)
    data = MNISTDataModule()
    trainer = pl.Trainer(gpus=1)

    trainer.fit(classifier, data)
