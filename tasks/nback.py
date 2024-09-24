from typing import Literal
from tasks.base import TaskDataset
from tasks.dms import get_one_hot


class NBackDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int,
        feature: Literal["category", "identity", "position"] = "category",
        pad_to: int = 0,
        category_size: int = 2,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
    ):
        task_len = max(6, pad_to)
        super(NBackDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
        )

        self.feature = feature
        self.task_index = get_one_hot({
            "category": 4,
            "identity": 5,
            "position": 6,
        }[feature])

        self.reset()

    def _reset(self, i):
        category, identity, position = self._set_random(self.dataset[i, 0])
        for j in range(1, 6):
            self.actions[i, j], category, identity, position = self._set_data(
                self.dataset[i, j], category, identity, position)

    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
