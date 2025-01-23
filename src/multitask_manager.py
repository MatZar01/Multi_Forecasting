from .dataset import get_matches


class MultiTask_Manager:
    def __init__(self, config):
        self.config = config
        self.matches_all = get_matches(path=config['DATA_PATH'])

        self.task_to_pair = {}
        self.pair_to_task = {}
        self.task_to_optimizer = {}
