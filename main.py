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
import logging


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
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
    print(f'{Style.green("[INFO]")} Pre-training...')
    multitask_manager.add_simple_task(task_number=-1, pair=None)
    multitask_manager.fit_simple(task=-1)

    # set task counter
    task_count = 1

    # train in similarity-based loop
    counter = 0  # initialize pair counter
    while True:
        counter += 1
        print(f'{Style.green("[INFO]")} Training {Style.orange(counter)} of '
              f'{Style.blue(len(multitask_manager.matches_all))} matches.')
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
            # add log info
            best_train, best_test = tune_error_train, tune_error_test
            selected_head = task_sim

            print(f'{Style.green("[INFO]")} updating task {Style.blue(task_sim)}')
            multitask_manager.add_pair_to_task(task=task_sim, pair=pair)  # add to init_task
            grapher.overall_results[task_sim] = grapher.overall_results[
                f'{temp_task_real_number}_tune']  # update overall results
        else:
            # add log info
            best_train, best_test = scratch_error_train, scratch_error_test
            task_count += 1  # update number of tasks
            selected_head = task_count
            print(f'{Style.green("[INFO]")} adding task {Style.blue(task_count)}')
            multitask_manager.transfer_temporal_task(task_number=task_count)  # add new task from temporal
            multitask_manager.add_pair_to_task(task=task_count, pair=pair)  # and add pair to it
            grapher.overall_results[task_count] = grapher.overall_results[
                f'{temp_task_real_number}_scratch']  # update overall results

        # remove temp dicts
        del grapher.overall_results[f'{temp_task_real_number}_tune']
        del grapher.overall_results[f'{temp_task_real_number}_scratch']

        # update grapher log
        grapher.log[counter-1] = {'head_count': task_count, 'selected_head': selected_head,
                                  'train': best_train, 'test': best_test}
        grapher.save_log()

        if len(multitask_manager.matches_left) == 0:
            break

    grapher.save_metadata(multitask_manager)
    print(f'{Style.green("[INFO]")} DONE!')
