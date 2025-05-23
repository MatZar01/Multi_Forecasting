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

    # set task counter
    task_count = 1

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
        temp_task_real_number = task_count + 1  # but we'll update temp number anyway
        # first finetune from the best sim task
        multitask_manager.add_temporal_task(init_task=task_sim, pair=pair, mode='tune')
        tune_error_train, tune_error_test = multitask_manager.fit_simple(task=0,
                                                                         temp_task_num=temp_task_real_number,
                                                                         mode='tune')

        # then train new pair from scratch
        multitask_manager.add_temporal_task(init_task=task_sim, pair=pair, mode='scratch')
        scratch_error_train, scratch_error_test = multitask_manager.fit_simple(task=0,
                                                                               temp_task_num=temp_task_real_number,
                                                                               mode='scratch')

        # now compare training error and decide if add to init_task or create new task
        if tune_error_test < scratch_error_test:
            multitask_manager.add_pair_to_task(task=task_sim, pair=pair)  # add to init_task
        else:
            task_count += 1  # update number of tasks
            multitask_manager.transfer_temporal_task(task_number=task_count)  # add new task from temporal
            multitask_manager.add_pair_to_task(task=task_count, pair=pair)  # and add pair to it

        if len(multitask_manager.matches_left) == 0:
            break

    grapher.save_metadata(multitask_manager)
    print(f'{Style.GREEN}[INFO]{Style.RESET} DONE!')
