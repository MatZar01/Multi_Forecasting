import numpy as np
import torch
import lightning as L
import importlib
from torch.utils.data import DataLoader

from src import get_args
from src import Forecasting_Dataset, get_matches
from src import RMSELoss


if __name__ == '__main__':
    # get training config
    config = get_args()

    # get all matches
    matches_all = get_matches(path=config['DATA_PATH'])
    # get dataset
    train_data = Forecasting_Dataset(path=config['DATA_PATH'], train=True, lag=config['LAG'],
                                     columns=config['COLUMNS'], matches=None, onehot_paths=config['ONEHOT_EMBEDDERS'])
    test_data = Forecasting_Dataset(path=config['DATA_PATH'], train=False, lag=config['LAG'],
                                    columns=config['COLUMNS'], matches=None, onehot_paths=config['ONEHOT_EMBEDDERS'])

    # get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
                                 persistent_workers=True)


