from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd
import numpy as np

ds_pt = '/Users/mateusz/Desktop/Multi_Forecasting/DS/train.csv'
data = pd.read_csv(ds_pt)

stores = np.array(data['store_id'].to_list())
skus = np.array(data['sku_id'].to_list())

store_encoder = OneHotEncoder()
sku_encoder = OneHotEncoder()

store_encoder.fit(stores.reshape(-1, 1))
sku_encoder.fit(skus.reshape(-1, 1))

pickle.dump(store_encoder, open('/Users/mateusz/Desktop/Multi_Forecasting/embedders3/onehot_C2.pkl', 'wb'))
pickle.dump(sku_encoder, open('/Users/mateusz/Desktop/Multi_Forecasting/embedders3/onehot_C3.pkl', 'wb'))
