from typing import Union
import random
from torch.utils.data import Dataset
import numpy as np


class TaskDataset(Dataset):
    def __init__(
        self,
        dataset_size: int,
        task_len: int = 6, # by default 6 frames per task
        category_size: int = 2,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
    ):
        self.feature = None
        self.task_index = None
        self.dataset_size = dataset_size
        self.category_size = category_size
        self.identity_size = identity_size
        self.position_size = position_size
        self.embedding_size = self.category_size + self.category_size * \
            self.identity_size + self.position_size  # CC IIII PPPP

        self.dataset = np.random.normal(loc=0, scale=std, size=(
            self.dataset_size, task_len, self.embedding_size))
        # 0 no match 1 match 2 no action
        self.actions = np.ones((self.dataset_size, task_len)) * 2
        self.task_len = task_len

    def reset(self):
        pass

    def _set_data(
        self,
        data: np.ndarray,
        category: int,
        identity: int,
        position: int,
        n_step_ago: int = 1,
    ):
        action = 0
        if random.random() < 0.5:
            action = 1
            if n_step_ago < self.task_len:
                if self.feature == 'category':
                    category, identity, position = self._set_random(
                        data, category=category)
                elif self.feature == 'identity':
                    category, identity, position = self._set_random(
                        data, category=category, identity=identity)
                elif self.feature == 'position':
                    category, identity, position = self._set_random(
                        data, position=position)
        else:
            if self.feature == 'category':
                new_category = self._get_random(category, self.category_size)
                category, identity, position = self._set_random(
                    data, category=new_category)
            elif self.feature == 'identity':
                new_identity = self._get_random(
                    (category - 1) * self.identity_size + identity, self.category_size * self.identity_size)
                new_category = (new_identity - 1) // self.identity_size + 1
                new_identity = (new_identity - 1) % self.identity_size + 1
                category, identity, position = self._set_random(
                    data, category=new_category, identity=new_identity)
            elif self.feature == 'position':
                new_position = self._get_random(position, self.position_size)
                category, identity, position = self._set_random(
                    data, position=new_position)
        return action, category, identity, position

    def _get_random(
        self,
        prev: int,
        limit: int,
    ):
        curr = random.randint(1, limit)
        while curr == prev:
            curr = random.randint(1, limit)
        return curr

    def _set_random(
        self,
        data: np.ndarray,
        category: Union[int, None] = None,
        identity: Union[int, None] = None,
        position: Union[int, None] = None,
    ):
        if category is None:
            category = random.randint(1, self.category_size)
        data[category - 1] += 1
        if identity is None:
            identity = random.randint(1, self.identity_size)
        data[self.category_size + (category - 1) *
             self.identity_size + identity - 1] += 1
        if position is None:
            position = random.randint(1, self.position_size)
        data[self.embedding_size - self.position_size + position - 1] += 1
        return category, identity, position

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx,):
        return self.dataset[idx], self.actions[idx], self.task_index


    