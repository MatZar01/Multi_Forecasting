from .dataset import get_matches, get_dataloader


class MultiTask_Manager:
    def __init__(self, config, grapher, logger, model_lib, loss_lib):
        self.config = config
        self.matches_all = get_matches(path=config['DATA_PATH'])

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
        self.train_dataloader, self.test_dataloader, data_info = get_dataloader(config=config,
                                                                                year=config['YEARS']['TRAIN'],
                                                                                matches=None)

        model_class = getattr(model_lib, config['MODEL'])
        self.model = model_class(sample_input=data_info['sample_input'], store_size=data_info['store_size'],
                                 sku_size=data_info['sku_size'],
                                 embedding_dim=config['EMBEDDING_SIZE']).to(config['DEVICE'])
