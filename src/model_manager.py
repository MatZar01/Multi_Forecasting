import numpy as np
import torch
from copy import deepcopy


class Model_Manager:
    def __init__(self, save_path):
        self.save_path = save_path
        self.last_best_error = np.inf

    def save_model(self, model, current_error):
        model_cp = deepcopy(model)
        model_cp = model_cp.to('cpu')
        torch.save(model_cp, f'{self.save_path}/last.pt')

        if current_error < self.last_best_error:
            self.last_best_error = current_error
            torch.save(model_cp, f'{self.save_path}/best.pt')
