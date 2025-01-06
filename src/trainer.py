import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class L_model(L.LightningModule):
    def __init__(self, model, loss_fn, test_fn, optimizer, config, grapher):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.test_fn = test_fn
        self.optimizer = optimizer
        self.grapher = grapher

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=config['SCHEDULER']['FACTOR'],
                                           patience=config['SCHEDULER']['PATIENCE'])

    def configure_optimizers(self):
        return self.optimizer

    def network_step(self, batch):
        store_emb, sku_emb, feature_vector, label = batch
        logits = self.model(store_emb, sku_emb, feature_vector)

        loss = self.loss_fn(logits, label)
        error = self.test_fn(logits, label)

        return logits, loss, error

    def training_step(self, batch):
        logits, loss, error = self.network_step(batch)

        return loss

    def validation_step(self, batch):
        logits, loss, error = self.network_step(batch)

        return loss

