config = {
    'DEVICE': 'cuda',
    'DATA_PATH': '/Users/pro/Desktop/Multi_Forecasting/DS/train.csv',
    'ONEHOT_EMBEDDERS': {'C2': '/Users/pro/Desktop/Multi_Forecasting/embedders/onehot_C2.pkl',
                         'C3': '/Users/pro/Desktop/Multi_Forecasting/embedders/onehot_C3.pkl'},

    'LAG': 2,
    'COLUMNS': [4, 5, 6, 7],  # [4, 5, 6, 7] for full feature vector

    'EPOCHS': 100,
    'LR': 1e-3,
    'BATCH_SIZE': 10,
}
