import torch
import lightning as L
import importlib
from torch.utils.data import DataLoader

from src import get_args
from src import Forecasting_Dataset, get_matches


if __name__ == '__main__':
    # get training config
    config = get_args()

    # get dataset
    train_data = Forecasting_Dataset(path=config['DATA_PATH'], train=True, lag=config['LAG'],
                                     columns=config['COLUMNS'], matches=None, onehot_paths=config['ONEHOT_EMBEDDERS'])

    print(0)
