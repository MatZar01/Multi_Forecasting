import numpy as np
import torch
import lightning as L
import importlib
import warnings

from src import get_args
from src import get_matches, get_dataloader
from src import L_model
from src import Grapher
from src import MultiTask_Manager
from tqdm import tqdm


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
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

    # pre train
    multitask_manager.fit_simple(task=-1)

    # train selected random pairs
    for t in multitask_manager.task_to_pair.keys():
        multitask_manager.fit_simple(task=t)

    # group the rest of the pairs
    print('[INFO] GROUPING PAIRS')
    matches_left_num = len(multitask_manager.matches_left)
    for i in tqdm(range(matches_left_num)):
        multitask_manager.add_pair_to_task(multitask_manager.matches_left[0])

    # train groups
    for t in multitask_manager.task_to_pair.keys():
        multitask_manager.fit_simple(task=t)

    grapher.save_metadata(multitask_manager)
    print('[INFO] DONE!')
