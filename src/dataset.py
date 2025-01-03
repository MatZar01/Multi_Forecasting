from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import torch


class Forecasting_Dataset(Dataset):
    def __init__(self, path: str, train: bool, lag: int, columns: list, matches: np.ndarray | None,
                 onehot_paths: dict):
        self.path = path
        self.train = train
        self.lag = lag
        self.columns = columns
        self.matches = matches

        self.onehot_2 = pickle.load(open(onehot_paths['C2'], 'rb'))
        self.onehot_3 = pickle.load(open(onehot_paths['C3'], 'rb'))

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.stores_ids, self.sku_ids, self.feature_vector, self.y = self.get_x_y()
        self.stores_ids_lagged, self.sku_ids_lagged, self.feature_vector_lagged, self.y_lag = self.get_lagged_data(self.stores_ids, self.sku_ids, self.feature_vector, self.y, self.lag)

    def load_data(self):
        data = pd.read_csv(self.path)
        # normalize prices data
        total_prices = np.array(data['total_price'].tolist())
        base_prices = np.array(data['base_price'].tolist())
        total_norm = self.normalize_data(total_prices)
        base_norm = self.normalize_data(base_prices)
        data['total_price'] = total_norm
        data['base_price'] = base_norm
        return data

    def get_lagged_data(self, stores_ids, sku_ids, feature_vector, y, lag):
        stores_ids_lagged = []
        sku_ids_lagged = []
        feature_vector_lagged = []
        y_lagged = []
        for i in range(feature_vector.shape[0] - (lag - 1) - 1):
            stores_ids_lagged.append(stores_ids[i])
            sku_ids_lagged.append(sku_ids[i])
            feature_vector_lagged.append(feature_vector[i:i+lag, :])
            y_lagged.append(y[i+lag])
        return (np.array(stores_ids_lagged), np.array(sku_ids_lagged),
                np.array(feature_vector_lagged).reshape(len(feature_vector_lagged), -1), np.array(y_lagged))

    def get_x_y(self):
        stores_ids = self.data_all[:, 2]
        sku_ids = self.data_all[:, 3]
        y = self.data_all[:, -1]
        feature_vector = self.data_all[:, self.columns + [-1]]  # [-1] is for the addition of past sales in vector
        return stores_ids, sku_ids, feature_vector, y

    def get_match(self):
        store_match_train = self.data_all[np.where(self.data_all[:, 2] == self.matches[0])[0]]
        train_single = store_match_train[np.where(store_match_train[:, 3] == self.matches[1])[0]]
        return train_single

    def split_to_years(self, data):
        data = data.to_numpy()
        years = np.array([int(x.split('/')[-1]) for x in data[:, 1]])
        years = years - np.min(years)

        train_y1 = data[np.where(years == 0)]
        train_y2 = data[np.where(years == 1)]
        val = data[np.where(years == 2)]
        train_all = np.vstack([train_y1, train_y2])
        if self.train:
            return train_all
        else:
            return val

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
        return batch[0], batch[1], torch.Tensor(batch[2]), torch.Tensor([batch[3]])


def get_matches(path):
    data = pd.read_csv(path)
    data = data.to_numpy()
    years = np.array([int(x.split('/')[-1]) for x in data[:, 1]])
    years = years - np.min(years)

    train_y1 = data[np.where(years == 0)]
    train_y2 = data[np.where(years == 1)]
    data_train = np.vstack([train_y1, train_y2])

    # get stores ids and sku ids
    stores = np.unique(data_train[:, 2])
    skus = np.unique(data_train[:, 3])
    # get cartesian product for store-sku
    c_prod = np.transpose([np.tile(stores, len(skus)), np.repeat(skus, len(stores))])
    # pick one pair for experiments
    return c_prod