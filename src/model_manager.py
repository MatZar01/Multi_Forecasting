import numpy as np
import torch
from copy import deepcopy


class Model_Manager:
    def __init__(self, save_path):
        self.save_path = save_path
        self.last_best_error = np.inf

    def save_model(self, model, current_error, task):
        # add prefix to model
        if task == -1:
            name = 'pre'
        else:
            name = f'T_{task}'

        model_cp = deepcopy(model)
        model_cp = model_cp.to('cpu')
        torch.save(model_cp, f'{self.save_path}/{name}_last.pt')

        if current_error < self.last_best_error:
            self.last_best_error = current_error
            torch.save(model_cp, f'{self.save_path}/{name}_best.pt')
