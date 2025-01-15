import numpy as np
import torch
import lightning as L
import importlib
from torch.utils.data import DataLoader

from src import get_args
from src import Forecasting_Dataset, get_matches, get_dataloader
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

    # get datasets
    train_dataloader, test_dataloader, data_info = get_dataloader(config=config, year=config['YEARS']['TRAIN'],
                                                                  matches=None)

    # load model
    model_class = getattr(models_lib, config['MODEL'])
    model = model_class(sample_input=data_info['sample_input'], store_size=data_info['store_size'],
                        sku_size=data_info['sku_size'], embedding_dim=config['EMBEDDING_SIZE']).to(config['DEVICE'])

    # load loss and test functions
    loss_class = getattr(loss_lib, config['LOSS_FN'])
    test_class = getattr(loss_lib, config['TEST_FN'])
    loss_fn = loss_class()
    test_fn = test_class()

    ##### PRE-TRAINING PHASE #####
    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR_PRE'],
                                  weight_decay=config['WEIGHT_DECAY'], amsgrad=False)

    # setup lightning trainer for pretraining
    light_model = L_model(model=model, loss_fn=loss_fn, test_fn=test_fn,
                          optimizer=optimizer, config=config, grapher=grapher, meta_phase=False)

    light_trainer_pre = L.Trainer(accelerator=config['DEVICE'], max_epochs=config['EPOCHS_PRE'],
                                  limit_train_batches=500, limit_val_batches=400,
                                  check_val_every_n_epoch=1, log_every_n_steps=5,
                                  enable_progress_bar=True, enable_checkpointing=False,
                                  logger=logger, num_sanity_val_steps=0)
    # train model
    light_trainer_pre.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)


    ##### META-TRAINING PHASE #####
    # get all matches
    matches_all = get_matches(path=config['DATA_PATH'])
    # get datasets
    meta_train_dataloader, meta_test_dataloader, meta_data_info = get_dataloader(config=config,
                                                                                 year=config['YEARS']['META'],
                                                                                 matches=matches_all[2])
    # set optimizer
    optimizer_meta = torch.optim.AdamW(model.parameters(), lr=config['LR_META'],
                                       weight_decay=config['WEIGHT_DECAY'], amsgrad=False)

    # setup lightning trainer for meta phase
    light_model_meta = L_model(model=model, loss_fn=loss_fn, test_fn=test_fn,
                               optimizer=optimizer_meta, config=config, grapher=grapher, meta_phase=True)

    light_trainer_meta = L.Trainer(accelerator=config['DEVICE'], max_epochs=config['EPOCHS_META'],
                                   limit_train_batches=500, limit_val_batches=400,
                                   check_val_every_n_epoch=1, log_every_n_steps=5,
                                   enable_progress_bar=True, enable_checkpointing=False,
                                   logger=logger, num_sanity_val_steps=0)

    # train model
    light_trainer_meta.fit(model=light_model_meta, train_dataloaders=meta_train_dataloader, val_dataloaders=meta_test_dataloader)

    print('[INFO] DONE!')
