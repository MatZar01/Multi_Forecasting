#%%
import torch

PATH = '/home/mateusz/Desktop/Multi_Forecasting/log_dir/2025-1-14+20+26_MLP_base/checkpoints/last.pt'

model = torch.load(PATH)
