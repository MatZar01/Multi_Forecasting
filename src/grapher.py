import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter
import yaml
import seaborn as sns
import pickle
import os


class Grapher:
    def __init__(self, config: dict):
        self.config = config
        self.time_name = None
        self.path, self.checkpoint_path = self.__set_path(base_pt=config['LOG_DIR'])

        self.error_train = []
        self.error_test = []
        self.loss_train = []
        self.loss_test = []
        self.lr = []

    def __set_path(self, base_pt: str):
        date = datetime.now()
        DIR = f'{date.year}-{date.month}-{date.day}+{date.hour}+{date.minute}_{self.config["MODEL"]}'
        DIR_checkpoint = f'{DIR}/checkpoints'
        self.time_name = DIR
        if not os.path.exists(f'{base_pt}/{DIR}'):
            os.makedirs(f'{base_pt}/{DIR}')
            os.makedirs(f'{base_pt}/{DIR_checkpoint}')
        return f'{base_pt}/{DIR}', f'{base_pt}/{DIR_checkpoint}'

    def save_yaml(self):
        out_dict = {'epoch_num': len(self.error_train),
                    'lr': self.lr,
                    'train': {'loss': self.loss_train, 'error': self.error_train},
                    'test': {'loss': self.loss_test, 'error': self.error_test},
                    'best': {'train_loss': np.nanmin(self.loss_train).item(),
                             'train_error': np.nanmin(self.error_train).item(),
                             'test_loss': np.nanmin(self.loss_test).item(),
                             'test_error': np.nanmin(self.error_test).item()}}

        self.config.update(out_dict)
        yaml.dump(self.config, open(f'{self.path}/results.yml', 'w'))
