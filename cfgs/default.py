config = {
    'DEVICE': 'cuda',
    'DATA_PATH': '/home/mateusz/Desktop/Multi_Forecasting/DS/train.csv',
    'ONEHOT_EMBEDDERS': {'C2': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C2.pkl',
                         'C3': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C3.pkl'},
    'LOG_DIR': '/home/mateusz/Desktop/Multi_Forecasting/log_dir',

    'LAG': 15,
    'COLUMNS': [4, 5, 6, 7],  # [4, 5, 6, 7] for full feature vector
    'YEARS': {'TRAIN': -1, 'TEST': 3},

    'MODEL': 'MLP_base',
    'EMBEDDING_SIZE': 5,
    'EPOCHS': 30,
    'LR': 1e-3,
    'WEIGHT_DECAY': 0.004,
    'BATCH_SIZE': 10,

    'TEST_FN': 'RMSELoss',
    'LOSS_FN': 'RMSELoss',

    'SCHEDULER': {'FACTOR': 0.6, 'PATIENCE': 2, 'THRESHOLD': 1e-3}
}
