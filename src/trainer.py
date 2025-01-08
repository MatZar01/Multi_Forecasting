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
                                           patience=config['SCHEDULER']['PATIENCE'],
                                           threshold=config['SCHEDULER']['THRESHOLD'])

        self.error_train = []
        self.error_test = []
        self.loss_train = []
        self.loss_test = []

        self.best_error_train = np.inf
        self.best_error_test = np.inf

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

        self.error_train.append(error.detach().cpu().numpy())
        self.loss_train.append(loss.detach().cpu().numpy())

        return loss

    def validation_step(self, batch):
        logits, loss, error = self.network_step(batch)

        self.error_test.append(error.detach().cpu().numpy())
        self.loss_test.append(loss.detach().cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        train_error = np.nanmean(self.error_train).item()
        train_loss = np.nanmean(self.loss_train).item()
        if train_error < self.best_error_train:
            self.best_error_train = train_error

        self.grapher.error_train.append(train_error)
        self.grapher.loss_train.append(train_loss)

        self.error_train = []
        self.loss_train = []

        self.grapher.save_data()

    def on_validation_epoch_end(self):
        test_error = np.nanmean(self.error_test).item()
        test_loss = np.nanmean(self.loss_test).item()
        if test_error < self.best_error_test:
            self.best_error_test = test_error

        self.grapher.error_test.append(test_error)
        self.grapher.loss_test.append(test_loss)
        self.grapher.lr.append(self.optimizer.param_groups[0]["lr"])

        self.scheduler.step(metrics=test_error)

        self.error_test = []
        self.loss_test = []

    def on_train_end(self):
        self.grapher.save_data()
        print(f'[INFO] TRAINING END\nTrain error: {self.best_error_train}\nVal error: {self.best_error_test}')
