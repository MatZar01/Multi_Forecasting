#%%
import yaml
import numpy as np
import os
from src import Style

m_path = '/Users/mateusz/Desktop/Multi_Forecasting/log_dir/rand'
experiment_list = [f.path for f in os.scandir(m_path) if f.is_dir()]

logs = []
metas = []

for e in experiment_list:
    meta_data = yaml.safe_load(open(f'{e}/metadata.yml', 'r'))
    log_data = yaml.safe_load(open(f'{e}/log.yml', 'r'))

    metas.append(meta_data)
    logs.append(log_data)
#%%
head_counts = []
head_selects = []
test_res = []

for i in logs[0]:
    head_counts.append([logs[0][i]['head_count'], logs[1][i]['head_count'], logs[2][i]['head_count']])
    head_selects.append([logs[0][i]['selected_head'], logs[1][i]['selected_head'], logs[2][i]['selected_head']])
    test_res.append([logs[0][i]['test'], logs[1][i]['test'], logs[2][i]['test']])
