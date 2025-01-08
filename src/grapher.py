import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter
import yaml
import seaborn as sns
import pickle
import os
from copy import deepcopy


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

        config_copy = deepcopy(self.config)
        config_copy.update(out_dict)
        yaml.dump(config_copy, open(f'{self.path}/results.yml', 'w'))

    def save_graphs(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        fig.suptitle(f'{self.config["MODEL"]} Results')
        ax1.plot(list(range(len(self.error_train))), self.error_train, label='Train error')
        ax1.plot([np.argmin(self.error_train)], [np.min(self.error_train)], 'ro')
        ax1.plot(list(range(len(self.error_test))), self.error_test, label='Test error')
        ax1.plot([np.argmin(self.error_test)], [np.min(self.error_test)], 'ro')
        ax1.set_title(f'Error: {self.config["TEST_FN"]}')
        ax1.set(xlabel='Epoch', ylabel='Error')
        ax1.grid()

        ax1.text(len(self.error_test)/3, np.max(self.error_test)*0.9,
                 f'Error min:\nTrain: {np.min(self.error_train):.4f}\nTest: {np.min(self.error_test):.4f}', color='black',
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=1'))

        ax1_lr = ax1.twinx()
        ax1_lr.plot(list(range(len(self.lr))), self.lr, label='LR', color='c', linestyle='dashed')
        ax1_lr.grid(color='c', linestyle='dashed')
        ax1_lr.tick_params(colors='c')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_lr.get_legend_handles_labels()
        lines_a = lines1 + lines2
        labels_a = labels1 + labels2
        plt.legend(lines_a, labels_a, loc='upper right')

        ax2.plot(list(range(len(self.loss_train))), self.loss_train, label='Train loss')
        ax2.plot([np.argmin(self.loss_train)], [np.min(self.loss_train)], 'ro')
        ax2.plot(list(range(len(self.loss_test))), self.loss_test, label='Test loss')
        ax2.plot([np.argmin(self.loss_test)], [np.min(self.loss_test)], 'ro')
        ax2.set_title(f'Loss: {self.config["LOSS_FN"]}')
        ax2.set(xlabel='Epoch', ylabel='Loss')
        ax2.grid()

        ax2.text(len(self.loss_test) / 3, np.max(self.loss_test) * 0.9,
                 f'Error min:\nTrain: {np.min(self.loss_train):.4f}\nTest: {np.min(self.loss_test):.4f}',
                 color='black',
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=1'))

        ax2_lr = ax2.twinx()
        ax2_lr.plot(list(range(len(self.lr))), self.lr, label='LR', color='c', linestyle='dashed')
        ax2_lr.grid(color='c', linestyle='dashed')
        ax2_lr.tick_params(colors='c')

        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_lr.get_legend_handles_labels()
        lines_b = lines3 + lines4
        labels_b = labels3 + labels4
        plt.legend(lines_b, labels_b, loc='upper right')

        plt.savefig(f'{self.path}/result_graph.png')
        plt.clf()
        plt.close()

    def save_data(self):
        self.save_yaml()
        self.save_graphs()
