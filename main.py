import numpy as np
import torch
import lightning as L
import importlib
from torch.utils.data import DataLoader

from src import get_args
from src import Forecasting_Dataset, get_matches
from src import L_model
from src import Grapher


if __name__ == '__main__':
    # get training config
    config = get_args()
    # load grapher and logger
    grapher = Grapher(config=config)
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir='./', version=grapher.time_name)

    # get models dir for import
    models_lib = importlib.import_module('src.models')
    # get loss dir for import
    loss_lib = importlib.import_module('src.loss_functions')

    # get all matches
    matches_all = get_matches(path=config['DATA_PATH'])
    # get dataset
    train_data = Forecasting_Dataset(path=config['DATA_PATH'], year=config['YEARS']['TRAIN'], lag=config['LAG'],
                                     columns=config['COLUMNS'], matches=None, onehot_paths=config['ONEHOT_EMBEDDERS'])
    test_data = Forecasting_Dataset(path=config['DATA_PATH'], year=config['YEARS']['TEST'], lag=config['LAG'],
                                    columns=config['COLUMNS'], matches=None, onehot_paths=config['ONEHOT_EMBEDDERS'])

    # get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
                                 persistent_workers=True)

    # load model
    model_class = getattr(models_lib, config['MODEL'])
    model = model_class(sample_input=train_data.batch_sample, store_size=train_data.emb_2_size,
                        sku_size=train_data.emb_3_size, embedding_dim=config['EMBEDDING_SIZE'])

    # load loss and test functions
    loss_class = getattr(loss_lib, config['LOSS_FN'])
    test_class = getattr(loss_lib, config['TEST_FN'])
    loss_fn = loss_class()
    test_fn = test_class()

    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'],
                                  weight_decay=config['WEIGHT_DECAY'], amsgrad=False)

    # setup lightning trainer
    light_model = L_model(model=model, loss_fn=loss_fn, test_fn=test_fn,
                          optimizer=optimizer, config=config, grapher=grapher)

    light_trainer = L.Trainer(accelerator=config['DEVICE'], max_epochs=config['EPOCHS'],
                              limit_train_batches=500, limit_val_batches=400,
                              check_val_every_n_epoch=1, log_every_n_steps=5,
                              enable_progress_bar=True, enable_checkpointing=False,
                              logger=logger)

    # train model
    light_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    print('[INFO] DONE!')
