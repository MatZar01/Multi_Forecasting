from .dataset import get_matches, get_dataloader
import torch
from src import L_model
import lightning as L
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from .print_style import Style


class MultiTask_Manager:
    def __init__(self, config, grapher, logger, model_lib, loss_lib):
        self.config = config
        self.matches_all = get_matches(path=config['DATA_PATH'])
        self.matches_left = get_matches(path=config['DATA_PATH'])
        self.matches_used = []
        self.current_task = None

        self.task_to_pair = {}
        self.pair_to_task = {}
        self.task_to_optimizer = {}

        self.grapher = grapher
        self.logger = logger

        loss_class = getattr(loss_lib, config['LOSS_FN'])
        test_class = getattr(loss_lib, config['TEST_FN'])
        self.loss_fn = loss_class()
        self.test_fn = test_class()

        # initial dataloader for -1 task
        self.train_dataloader, self.test_dataloader, data_info, train_data, test_data = get_dataloader(config=config,
                                                                                year=config['YEARS']['TRAIN'],
                                                                                matches=None)

        model_class = getattr(model_lib, config['MODEL'])
        self.model = model_class(sample_input=data_info['sample_input'], store_size=data_info['store_size'],
                                 sku_size=data_info['sku_size'],
                                 embedding_dim=config['EMBEDDING_SIZE'],
                                 device=config['DEVICE']).to(config['DEVICE'])

    def add_simple_task(self, task_number, pair):
        self.model.add_head(task_number)

        if task_number == -1:
            self.task_to_optimizer[task_number] = torch.optim.AdamW(self.model.parameters(), lr=self.config['LR_PRE'],
                                                                    weight_decay=self.config['WEIGHT_DECAY'],
                                                                    amsgrad=False)
        else:
            self.task_to_optimizer[task_number] = torch.optim.AdamW(self.model.parameters(), lr=self.config['LR_META'],
                                                                    weight_decay=self.config['WEIGHT_DECAY'],
                                                                    amsgrad=False)
            if task_number not in self.task_to_pair.keys():
                self.task_to_pair[task_number] = []
                self.task_to_pair[task_number].append(pair)
                self.pair_to_task[f'{pair}'] = task_number

            # select new dataloaders for task
            self.select_dataloader(task_number)

    def add_temporal_task(self, init_task, pair, mode):
        """
        Adds temporal task 0 to check performance
        """
        # use -1 task or init_task head
        if mode == 'tune':
            self.model.add_head(task=0, copy_task=init_task)
        else:
            self.model.add_head(task=0)

        # set optimizer for temporal task
        self.task_to_optimizer[0] = torch.optim.AdamW(self.model.parameters(), lr=self.config['LR_META'],
                                                      weight_decay=self.config['WEIGHT_DECAY'],
                                                      amsgrad=False)

        # select dataloader for temporal task
        self.select_temp_dataloader(task_number=init_task, pair=pair, mode=mode)

    def check_similarity(self, pair, mode):
        """
        Returns the number of most similar task
        SIM is computed as RMSE or EUC error, so less is better ;-)
        """

        print(f'{Style.GREEN}[INFO]{Style.RESET} Checking similarity across {Style.ORANGE}{len(self.pair_to_task)}{Style.RESET} tasks')
        # get pair data
        _, _, _, pair_data, _ = get_dataloader(config=self.config, year=self.config['YEARS']['TRAIN'], matches=[pair])
        pair_data = np.mean(np.concatenate([x[2].astype(float) for x in pair_data.x_y_lagged], axis=1), axis=1)

        sims = []
        tasks = []
        # compute sim for every task
        for key in tqdm(self.task_to_pair):
            tasks.append(key)
            pairs = self.task_to_pair[key]
            _, _, _, task_data, _ = get_dataloader(config=self.config, year=self.config['YEARS']['TRAIN'],
                                                   matches=pairs)
            task_data = np.mean(np.concatenate([x[2].astype(float) for x in task_data.x_y_lagged], axis=1), axis=1)

            # get sim
            if mode == 'RMSE':
                sim = root_mean_squared_error(pair_data.reshape(1, -1), task_data.reshape(1, -1))
            else:
                sim = euclidean_distances(pair_data.reshape(1, -1), task_data.reshape(1, -1))
            sims.append(sim)

            return tasks[np.argmin(sims)]

    def select_new_pair(self):
        new_pair = self.matches_left.pop(0)
        self.matches_used.append(new_pair)
        return new_pair

    def add_random_pairs_to_tasks(self, pairs):
        task_number = 0
        for pair in pairs:
            self.task_to_pair[task_number] = []
            self.task_to_pair[task_number].append(pair)
            self.pair_to_task[f'{pair}'] = task_number
            self.matches_used.append(pair)
            self.matches_left.pop(self.matches_all.index(pair))
            task_number += 1

    def select_random_pairs(self, number, seed):
        np.random.seed(seed)
        random_ids = np.random.choice(list(range(len(self.matches_all))), size=number, replace=False)
        random_pairs = [self.matches_all[x] for x in random_ids]
        return random_pairs

    def select_dataloader(self, task_number):
        matches = self.task_to_pair[task_number]
        self.train_dataloader, self.test_dataloader, data_info, train_data, test_data = get_dataloader(config=self.config,
                                                                                year=self.config['YEARS']['META'],
                                                                                matches=matches)

    def select_temp_dataloader(self, task_number, pair, mode):
        if mode == 'tune':
            matches = self.task_to_pair[task_number]
            matches.append(pair)
            self.train_dataloader, self.test_dataloader, data_info, train_data, test_data = get_dataloader(
                config=self.config,
                year=self.config['YEARS']['META'],
                matches=matches)
        else:
            self.train_dataloader, self.test_dataloader, data_info, train_data, test_data = get_dataloader(
                config=self.config,
                year=self.config['YEARS']['META'],
                matches=[pair])

    def test_pair(self, pair):
        print('[INFO] testing model...')
        self.model.eval()
        self.model.to(self.config['DEVICE'])
        out_scores = []
        for t in self.task_to_pair.keys():
            self.train_dataloader, self.test_dataloader, data_info, train_data, test_data = get_dataloader(config=self.config,
                                                                                    year=self.config['YEARS']['TEST'],
                                                                                    matches=[pair])
            store_ids = []
            store_skus = []
            vectors = []
            ys = []
            test_range = np.min([len(test_data.x_y_lagged), 200])
            test_samples = np.random.choice(list(range(len(test_data.x_y_lagged))), size=test_range, replace=False)
            for i in test_samples:
                store_id, sku_id, vector, y = test_data.__getitem__(i)
                store_id = store_id.unsqueeze(0).to(self.config['DEVICE'])
                sku_id = sku_id.unsqueeze(0).to(self.config['DEVICE'])
                vector = vector.unsqueeze(0).to(self.config['DEVICE'])
                y = y.unsqueeze(0).to(self.config['DEVICE'])
                store_ids.append(store_id)
                store_skus.append(sku_id)
                vectors.append(vector)
                ys.append(y)
            ids = torch.concatenate(store_ids)
            skus = torch.concatenate(store_skus)
            vector_ts = torch.concatenate(vectors)
            y_hats = torch.concatenate(ys)
            result = self.model(ids, skus, vector_ts, t)
            out_scores.append(self.test_fn(result, y_hats).detach().cpu().numpy().item())

        return out_scores

    def add_test_pair_to_task(self, pair):
        out_scores = self.test_pair(pair)
        task_assignment = np.argmin(out_scores)
        self.task_to_pair[task_assignment].append(pair)
        self.pair_to_task[f'{pair}'] = task_assignment
        self.matches_used.append(pair)
        self.matches_left.remove(pair)

    def add_pair_to_task(self, task, pair):
        self.task_to_pair[task].append(pair)
        self.pair_to_task[f'{pair}'] = task

    def fit_simple(self, task):
        self.model.train()
        # freeze feature extractor for all tasks except -1
        if task != -1:
            self.model.freeze_model_layers()
            max_epochs = self.config['EPOCHS_META']
        else:
            max_epochs = self.config['EPOCHS_PRE']

        light_model = L_model(model=self.model, loss_fn=self.loss_fn, test_fn=self.test_fn,
                              optimizer=self.task_to_optimizer[task], config=self.config, grapher=self.grapher,
                              task=task)

        light_trainer = L.Trainer(accelerator=self.config['DEVICE'], max_epochs=max_epochs,
                                  limit_train_batches=500, limit_val_batches=400,
                                  check_val_every_n_epoch=1, log_every_n_steps=1,
                                  enable_progress_bar=True, enable_checkpointing=False,
                                  logger=self.logger, num_sanity_val_steps=0)
        # train model
        light_trainer.fit(model=light_model, train_dataloaders=self.train_dataloader,
                          val_dataloaders=self.test_dataloader)

        return light_model.best_error_train, light_model.best_error_test
