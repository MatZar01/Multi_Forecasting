import numpy as np
import torch
import lightning as L
import importlib

from src import get_args
from src import get_matches, get_dataloader
from src import L_model
from src import Grapher
from src import MultiTask_Manager


if __name__ == '__main__':
    # get training config
    config = get_args()
    # load grapher and logger
    grapher = Grapher(config=config)
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir='./', version=grapher.time_name)

    # get models dir for import
    models_lib = importlib.import_module('src.models')
    # get loss dir for import
    loss_lib = importlib.import_module('src.loss_functions')

    multitask_manager = MultiTask_Manager(config=config, grapher=grapher, logger=logger, model_lib=models_lib,
                                          loss_lib=loss_lib)

    random_pairs = multitask_manager.select_random_pairs(number=config['RANDOM_PAIR_NUM'], seed=config['SEED'])
    multitask_manager.add_random_pairs_to_tasks(random_pairs)

    '''multitask_manager.fit(task=-1)
    for i in range(len(multitask_manager.matches_all)):
        multitask_manager.fit(task=i)'''

    print('[INFO] DONE!')
