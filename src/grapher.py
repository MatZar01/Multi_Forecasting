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

        self.error_train = {}
        self.error_test = {}
        self.loss_train = {}
        self.loss_test = {}
        self.lr = {}

        self.overall_results = {}
        self.log = {}

    def __set_path(self, base_pt: str):
        date = datetime.now()
        DIR = f'{date.year}-{date.month}-{date.day}+{date.hour}+{date.minute}_{self.config["MODEL"]}'
        DIR_checkpoint = f'{DIR}/checkpoints'
        self.time_name = DIR
        if not os.path.exists(f'{base_pt}/{DIR}'):
            os.makedirs(f'{base_pt}/{DIR}')
            os.makedirs(f'{base_pt}/{DIR_checkpoint}')
        return f'{base_pt}/{DIR}', f'{base_pt}/{DIR_checkpoint}'

    def add_task_folder(self, task):
        if not os.path.exists(f'{self.path}/{task}'):
            os.makedirs(f'{self.path}/{task}')

    def select_data(self, task):
        if task == -1:
            return self.error_train[task], self.error_test[task], self.loss_train[task], \
                   self.loss_test[task], self.lr[task]
        else:
            error_train = self.error_train[-1] + self.error_train[task]
            error_test = self.error_test[-1] + self.loss_test[task]
            loss_train = self.loss_train[-1] + self.loss_train[task]
            loss_test = self.loss_test[-1] + self.loss_test[task]
            lr = self.lr[-1] + self.lr[task]
            return error_train, error_test, loss_train, loss_test, lr

    def update_temp_name(self, task, temp_task, temp_mode):
        if temp_task is not None:
            return f'{temp_task}_{temp_mode}'
        else:
            return task

    def save_yaml(self, task, temp_task=None, temp_mode=None):
        error_train, error_test, loss_train, loss_test, lr = self.select_data(task)

        # update task number if it was temporal
        task = self.update_temp_name(task, temp_task, temp_mode)

        out_dict = {'epoch_num': len(error_train),
                    'lr': self.lr,
                    'train': {'loss': loss_train, 'error': error_train},
                    'test': {'loss': loss_test, 'error': error_test},
                    'best': {'train_loss': np.nanmin(loss_train).item(),
                             'train_error': np.nanmin(error_train).item(),
                             'test_loss': np.nanmin(loss_test).item(),
                             'test_error': np.nanmin(error_test).item()}}

        config_copy = deepcopy(self.config)
        config_copy.update(out_dict)
        yaml.dump(config_copy, open(f'{self.path}/{task}/results.yml', 'w'))
        self.overall_results[task] = out_dict

    def save_graphs(self, task, temp_task=None, temp_mode=None):
        error_train, error_test, loss_train, loss_test, lr = self.select_data(task)

        # update task number if it was temporal
        task = self.update_temp_name(task, temp_task, temp_mode)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        fig.suptitle(f'{self.config["MODEL"]} Results')
        ax1.plot(list(range(len(error_train))), error_train, label='Train error')
        ax1.plot([np.argmin(error_train)], [np.min(error_train)], 'ro')
        ax1.plot(list(range(len(error_test))), error_test, label='Test error')
        ax1.plot([np.argmin(error_test)], [np.min(error_test)], 'ro')
        ax1.set_title(f'Error: {self.config["TEST_FN"]}')
        ax1.set(xlabel='Epoch', ylabel='Error')
        ax1.axvline(self.config['EPOCHS_PRE'])
        ax1.grid()

        ax1.text(len(error_test)/3, np.max(error_test)*0.9,
                 f'Error min:\nTrain: {np.min(error_train):.4f}\nTest: {np.min(error_test):.4f}', color='black',
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=1'))

        ax1_lr = ax1.twinx()
        ax1_lr.plot(list(range(len(lr))), lr, label='LR', color='c', linestyle='dashed')
        ax1_lr.grid(color='c', linestyle='dashed')
        ax1_lr.tick_params(colors='c')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_lr.get_legend_handles_labels()
        lines_a = lines1 + lines2
        labels_a = labels1 + labels2
        plt.legend(lines_a, labels_a, loc='upper right')

        ax2.plot(list(range(len(loss_train))), loss_train, label='Train loss')
        ax2.plot([np.argmin(loss_train)], [np.min(loss_train)], 'ro')
        ax2.plot(list(range(len(loss_test))), loss_test, label='Test loss')
        ax2.plot([np.argmin(loss_test)], [np.min(loss_test)], 'ro')
        ax2.set_title(f'Loss: {self.config["LOSS_FN"]}')
        ax2.set(xlabel='Epoch', ylabel='Loss')
        ax2.axvline(self.config['EPOCHS_PRE'])
        ax2.grid()

        ax2.text(len(loss_test) / 3, np.max(loss_test) * 0.9,
                 f'Error min:\nTrain: {np.min(loss_train):.4f}\nTest: {np.min(loss_test):.4f}',
                 color='black',
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=1'))

        ax2_lr = ax2.twinx()
        ax2_lr.plot(list(range(len(lr))), lr, label='LR', color='c', linestyle='dashed')
        ax2_lr.grid(color='c', linestyle='dashed')
        ax2_lr.tick_params(colors='c')

        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_lr.get_legend_handles_labels()
        lines_b = lines3 + lines4
        labels_b = labels3 + labels4
        plt.legend(lines_b, labels_b, loc='upper right')

        plt.savefig(f'{self.path}/{task}/result_graph.png')
        plt.clf()
        plt.close()

    def save_mean_std_plot(self, means, stds):
        plt.plot(list(range(len(means))), means)
        plt.title('Mean scores')
        plt.grid()
        plt.xlabel('Pair count')
        plt.ylabel('Mean RMSE')
        plt.savefig(f'{self.path}/means.png')
        plt.clf()

        plt.plot(list(range(len(stds))), stds)
        plt.title('Mean stds')
        plt.grid()
        plt.xlabel('Pair count')
        plt.ylabel('Mean STD')
        plt.savefig(f'{self.path}/stds.png')
        plt.clf()

    def save_data(self, task, temp_task=None, temp_mode=None):
        self.save_yaml(task=task, temp_task=temp_task, temp_mode=temp_mode)
        self.save_graphs(task=task, temp_task=temp_task, temp_mode=temp_mode)

    def save_metadata(self, multitask_manager):
        task_to_pair = multitask_manager.task_to_pair
        results = {}
        n_pairs = []
        best_train = []
        best_test = []
        for key in self.overall_results.keys():
            if key != -1:
                results[key] = {'n_pairs': len(task_to_pair[key]),
                                'best_train': self.overall_results[key]['best']['train_error'],
                                'best_test': self.overall_results[key]['best']['test_error']}

                n_pairs.append(len(task_to_pair[key]))
                best_train.append(self.overall_results[key]['best']['train_error'])
                best_test.append(self.overall_results[key]['best']['test_error'])

        results['overall'] = {'train': (np.sum(np.array(n_pairs) * np.array(best_train)) / np.sum(n_pairs)).item(),
                              'test': (np.sum(np.array(n_pairs) * np.array(best_test)) / np.sum(n_pairs)).item()}

        task_to_pair.update(results)
        yaml.dump(task_to_pair, open(f'{self.path}/metadata.yml', 'w'))

    def save_log(self):
        yaml.dump(self.log, open(f'{self.path}/log.yml', 'w'))
