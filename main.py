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
from src import Style
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
    print(f'{Style.GREEN}[INFO]{Style.RESET} Pre-training...')
    multitask_manager.add_simple_task(task_number=-1, pair=None)
    multitask_manager.fit_simple(task=-1)

    # train in similarity-based loop
    while True:
        # get pair from matches_left
        pair = multitask_manager.select_new_pair()

        # add new task and train new pair if tasks are empty
        if not multitask_manager.task_to_pair:
            multitask_manager.add_simple_task(task_number=1, pair=pair)
            multitask_manager.fit_simple(task=1)
            continue

        # get most similar task
        task_sim = multitask_manager.check_similarity(pair=pair, mode=config['SIM'])

        # compare performance between updated, existing head vs new head from -1 task
        # both will use temporal task == 0
        # first finetune from the best sim task
        multitask_manager.add_temporal_task(init_task=task_sim, pair=pair, mode='tune')
        tune_error_train, tune_error_test = multitask_manager.fit_simple(task=0)

        # then train new pair from scratch
        multitask_manager.add_temporal_task(init_task=task_sim, pair=pair, mode='scratch')
        scratch_error_train, scratch_error_test = multitask_manager.fit_simple(task=0)


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
