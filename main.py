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

    # pre train
    multitask_manager.fit_simple(task=-1)

    # train in similarity-based loop
    while True:
        # get pair from matches_left
        pair = multitask_manager.select_new_pair()


        if len(multitask_manager.matches_left) == 0:
            break

    grapher.save_metadata(multitask_manager)

    """# pre train
    multitask_manager.fit_simple(task=-1)

    stds = []
    means = []

    multitask_manager.task_to_pair[1] = []
    for pair in multitask_manager.matches_left:
        multitask_manager.matches_used.append(pair)
        multitask_manager.add_pair_to_task(task=1, pair=pair)
        multitask_manager.fit_simple(task=1)
        out_scores = []
        for pair_test in multitask_manager.matches_used:
            test_score = multitask_manager.test_pair(pair=pair_test)[0]
            out_scores.append(test_score)

        stds.append(np.std(out_scores))
        means.append(np.mean(out_scores))
        grapher.save_mean_std_plot(means, stds)

    print('[INFO] DONE!')"""
