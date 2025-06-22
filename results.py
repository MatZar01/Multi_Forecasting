import yaml
import numpy as np
import os
from src import Style

ds_path = '/Users/mateusz/Downloads/DS1'
experiment_list = [f.path for f in os.scandir(ds_path) if f.is_dir()]

train_res = []
test_res = []

test_frac_mins = []
test_frac_max = []

for e in experiment_list:
    data_path = f'{e}/metadata.yml'
    data = yaml.safe_load(open(data_path, 'r'))

    overall_test = data['overall']['test']
    overall_train = data['overall']['train']
    train_res.append(overall_train)
    test_res.append(overall_test)

    temp = []
    for key in data.keys():
        if str(key).isdigit():
            temp.append(data[key]['best_test'])

    test_frac_mins.append(np.min(temp))
    test_frac_max.append(np.max(temp))

mean_test = np.mean(test_res)
std_test = np.std(test_res)

min_test = np.min(test_frac_mins)
std_min = np.std(test_frac_mins)

max_test = np.max(test_frac_max)
std_max = np.std(test_frac_max)

print(f'{Style.green("DS:")} {ds_path.split("/")[-1]}\n'
      f'{Style.orange("test RMSE:")} {Style.green(mean_test)}, {Style.orange("STD:")} {Style.blue(std_test)}\n'
      f'{Style.orange("test max:")} {Style.green(max_test)}, {Style.orange("STD:")} {Style.blue(std_max)}\n'
      f'{Style.orange("test min:")} {Style.green(min_test)}, {Style.orange("STD:")} {Style.blue(std_min)}')
#%%