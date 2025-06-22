import yaml
import numpy as np
import os

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

print(f'DS: {ds_path.split("/")[-1]}\n'
      f'test RMSE: {mean_test}, STD: {std_test}\n'
      f'test max: {max_test}, STD: {std_max}\n'
      f'test min: {min_test}, STD: {std_min}')
#%%