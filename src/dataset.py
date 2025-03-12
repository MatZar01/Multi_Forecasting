from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader


class Forecasting_Dataset(Dataset):
    def __init__(self, path: str, year: int, lag: int, columns: list, matches: np.ndarray | None,
                 onehot_paths: dict):
        self.path = path
        self.year = year
        self.lag = lag
        self.columns = columns
        self.matches = matches

        self.onehot_2 = pickle.load(open(onehot_paths['C2'], 'rb'))
        self.onehot_3 = pickle.load(open(onehot_paths['C3'], 'rb'))
        self.emb_2_size = self.onehot_2.categories_[0].shape[0]
        self.emb_3_size = self.onehot_3.categories_[0].shape[0]

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()
        else:
            self.matches = np.array([])

        self.stores_ids, self.sku_ids, self.feature_vector, self.y = self.get_x_y()
        self.stores_ids_lagged, self.sku_ids_lagged, self.feature_vector_lagged, self.y_lag = self.get_lagged_data(self.stores_ids, self.sku_ids, self.feature_vector, self.y, self.lag)

        self.batch_sample = self.__getitem__(0)

    def load_data(self):
        data = pd.read_csv(self.path)
        # normalize prices data
        total_prices = np.array(data['total_price'].tolist())
        base_prices = np.array(data['base_price'].tolist())
        units_sold = np.array(data['units_sold'].tolist())
        total_norm = self.normalize_data(total_prices)
        base_norm = self.normalize_data(base_prices)
        units_norm = self.normalize_data(units_sold)
        data['total_price'] = total_norm
        data['base_price'] = base_norm
        data['units_norm'] = units_norm
        return data

    def get_lagged_data(self, stores_ids, sku_ids, feature_vector, y, lag):
        stores_ids_lagged = []
        sku_ids_lagged = []
        feature_vector_lagged = []
        y_lagged = []
        for i in range(feature_vector.shape[0] - (lag - 1) - 1):
            stores_ids_lagged.append(stores_ids[i:i+lag])
            sku_ids_lagged.append(sku_ids[i:i+lag])
            feature_vector_lagged.append(feature_vector[i:i+lag, :])
            y_lagged.append(y[i+lag])
        return (np.array(stores_ids_lagged), np.array(sku_ids_lagged),
                np.array(feature_vector_lagged), np.array(y_lagged))

    def get_x_y(self):
        stores_ids = self.data_all[:, 2]
        sku_ids = self.data_all[:, 3]
        y = self.data_all[:, -2]  # [-2] as new column of normed sales is added
        feature_vector = self.data_all[:, self.columns + [-1]]  # [-1] is for the addition of normed past sales in vector
        return stores_ids, sku_ids, feature_vector, y

    def get_match(self):
        train_matches = []
        for m in self.matches:
            store_match_train = self.data_all[np.where(self.data_all[:, 2] == m[0])[0]]
            train_single = store_match_train[np.where(store_match_train[:, 3] == m[1])[0]]
            train_matches.append(train_single)
        return np.concatenate(train_matches)

    def split_to_years(self, data):
        data = data.to_numpy()
        years = np.array([int(x.split('/')[-1]) for x in data[:, 1]])
        years = years - np.min(years)

        train_y1 = data[np.where(years == 0)]
        train_y2 = data[np.where(years == 1)]
        val = data[np.where(years == 2)]
        train_all = np.vstack([train_y1, train_y2])

        match self.year:
            case 1:
                return train_y1
            case 2:
                return train_y2
            case 3:
                return val
            case -1:
                return train_all

    def encode_cats(self, batch):
        stores_ids, sku_ids, feature_vectors, y = batch
        onehot_2 = self.onehot_2.transform(stores_ids.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_3 = self.onehot_3.transform(sku_ids.reshape(-1, 1).astype(np.object_)).toarray()

        emb_2 = torch.LongTensor(np.argmax(onehot_2, axis=1))
        emb_3 = torch.LongTensor(np.argmax(onehot_3, axis=1))

        return emb_2, emb_3, feature_vectors, y

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.feature_vector_lagged.shape[0]

    def __getitem__(self, idx):
        batch = (self.stores_ids_lagged[idx].astype(int), self.sku_ids_lagged[idx].astype(int),
                 self.feature_vector_lagged[idx].astype(float), self.y_lag[idx].astype(float))
        batch = self.encode_cats(batch)
        return batch[0], batch[1], torch.Tensor(batch[2]), torch.Tensor([batch[3]]) #, self.matches.astype(int)


def get_matches(path):
    data = pd.read_csv(path)
    data = data.to_numpy()

    pairs = data[:, [2, 3]].astype(int)
    uq_pairs = np.unique(pairs, axis=0)

    return uq_pairs.tolist()


def get_dataloader(config, year, matches):
    train_data = Forecasting_Dataset(path=config['DATA_PATH'], year=year, lag=config['LAG'],
                                     columns=config['COLUMNS'], matches=matches, onehot_paths=config['ONEHOT_EMBEDDERS'])
    test_data = Forecasting_Dataset(path=config['DATA_PATH'], year=config['YEARS']['TEST'], lag=config['LAG'],
                                    columns=config['COLUMNS'], matches=matches, onehot_paths=config['ONEHOT_EMBEDDERS'])

    # get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=15,
                                  persistent_workers=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=15,
                                 persistent_workers=True, drop_last=True)

    data_info = {'sample_input': train_data.batch_sample, 'store_size': train_data.emb_2_size,
                 'sku_size': train_data.emb_3_size}

    return train_dataloader, test_dataloader, data_info, train_data, test_data
