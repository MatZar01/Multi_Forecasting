#%%
import pandas as pd
import numpy as np


data_path = '/Users/pro/Desktop/Multi_Forecasting/DS/train.csv'

data = pd.read_csv(data_path)
#%%
total_prices = np.array(data['total_price'].tolist())
base_prices = np.array(data['base_price'].tolist())
#%%
total_min = np.min(total_prices)
total_max = np.max(total_prices)

base_min = np.min(base_prices)
base_max = np.max(base_prices)
#%%
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

total_norm = normalize_data(total_prices)
base_norm = normalize_data(base_prices)
#%%
data_norm = data.copy()
data_norm['total_price'] = total_norm
data_norm['base_price'] = base_norm
