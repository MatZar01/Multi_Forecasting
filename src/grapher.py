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
        self.path = self.__set_path(base_pt=config['LOG_DIR'])
        print(0)

    def __set_path(self, base_pt: str):
        date = datetime.now()
        DIR = f'{date.year}-{date.month}-{date.day}+{date.hour}+{date.minute}_{self.config["MODEL"]}'
        self.time_name = DIR
        if not os.path.exists(f'{base_pt}/{DIR}'):
            os.makedirs(f'{base_pt}/{DIR}')
        return f'{base_pt}/{DIR}/'

