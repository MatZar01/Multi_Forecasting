config = {
    'DEVICE': 'cuda',
    'DATA_PATH': '/home/mateusz/Desktop/Multi_Forecasting/DS/train.csv',
    'ONEHOT_EMBEDDERS': {'C2': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C2.pkl',
                         'C3': '/home/mateusz/Desktop/Multi_Forecasting/embedders/onehot_C3.pkl'},
    'LOG_DIR': '/home/mateusz/Desktop/Multi_Forecasting/log_dir',

    'LAG': 15,
    'COLUMNS': [],  # [4, 5, 6, 7] for full feature vector
    'YEARS': {'TRAIN': 1, 'META': 2, 'TEST': 3},

    'MODEL': 'MLP_base',
    'EMBEDDING_SIZE': 5,
    'EPOCHS_PRE': 1,
    'EPOCHS_META': 30,
    'LR_PRE': 1e-3,
    'LR_META': 1e-3,
    'WEIGHT_DECAY': 0.04,
    'BATCH_SIZE': 10,

    'TEST_FN': 'RMSELoss',
    'LOSS_FN': 'RMSELoss',

    'SCHEDULER_PRE': {'FACTOR': 0.6, 'PATIENCE': 2, 'THRESHOLD': 1e-3},
    'SCHEDULER_META': {'FACTOR': 0.4, 'PATIENCE': 2, 'THRESHOLD': 1e-3}
}
