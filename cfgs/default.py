config = {
    'DEVICE': 'cuda',
    'DATA_PATH': '/home/mateusz/Desktop/Multi_Forecasting/DS/train.csv',
    'ONEHOT_EMBEDDERS': {'C2': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C2.pkl',
                         'C3': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C3.pkl'},

    'LAG': 10,
    'COLUMNS': [4, 5, 6, 7],

    'EPOCHS': 100,
    'LR': 1e-3,
    'BATCH_SIZE': 10,
}
