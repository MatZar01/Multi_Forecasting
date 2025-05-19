config = {
    'DEVICE': 'cpu',
    'DATA_PATH': 'DS/train.csv',
    'ONEHOT_EMBEDDERS': {'C2': 'embedders/onehot_C2.pkl',
                         'C3': 'embedders/onehot_C3.pkl'},
    'LOG_DIR': 'log_dir',

    'SEED': 24,
    'RANDOM_PAIR_NUM': 20,

    'LAG': 15,
    'COLUMNS': [],  # [4, 5, 6, 7] for full feature vector
    'YEARS': {'TRAIN': 1, 'META': 2, 'TEST': 3},

    'MODEL': 'MLP_base',
    'EMBEDDING_SIZE': 5,

    'EPOCHS_PRE': 100,
    'EPOCHS_META': 60,

    'LR_PRE': 1e-2,
    'LR_META': 1e-3,

    'WEIGHT_DECAY': 0.04,
    'BATCH_SIZE': 10,

    'TEST_FN': 'RMSELoss',
    'LOSS_FN': 'MSE',

    'SCHEDULER_PRE': {'FACTOR': 0.6, 'PATIENCE': 20, 'THRESHOLD': 1e-3},
    'SCHEDULER_META': {'FACTOR': 0.8, 'PATIENCE': 10, 'THRESHOLD': 1e-3}
}
