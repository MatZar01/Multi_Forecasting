#%%
import importlib
from src.dataset import get_matches, get_dataloader
import warnings
warnings.filterwarnings('ignore')


cfg_name = 'default'
cfg_module = importlib.import_module(f'cfgs.{cfg_name}')
config = cfg_module.config

matches_all = get_matches(path=config['DATA_PATH'])

m1 = matches_all[0]
m2 = matches_all[77]

dl_train_1, dl_test_1, data_info, train_1, test_1 = get_dataloader(config=config, year=config['YEARS']['TRAIN'], matches=[m1])
dl_train_2, dl_test_2, data_info, train_2, test_2 = get_dataloader(config=config, year=config['YEARS']['TRAIN'], matches=[m2])

#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import root_mean_squared_error

d_1 = np.mean(np.concatenate([x[2].astype(float) for x in test_1.x_y_lagged], axis=1), axis=1)
d_2 = np.mean(np.concatenate([x[2].astype(float) for x in test_2.x_y_lagged], axis=1), axis=1)

sim = cosine_similarity(d_1.reshape(1, -1), d_2.reshape(1, -1))
dist = euclidean_distances(d_1.reshape(1, -1), d_2.reshape(1, -1))
rmse = root_mean_squared_error(d_1.reshape(1, -1), d_2.reshape(1, -1))
