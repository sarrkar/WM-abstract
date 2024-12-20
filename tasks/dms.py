from typing import Literal
from tasks.base import TaskDataset
from tasks.utils import get_one_hot


class DMSDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int = 128,
        feature: Literal["category", "identity", "position"] = "category",
        pad_to: int = 0,
        category_size: int = 4,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        task_index_base_value: int = 40,
        total_tasks: int = 43
    ):
        task_len = max(2, pad_to)
        super(DMSDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
        )

        self.feature = feature
        self.task_index = get_one_hot({
            "position": task_index_base_value + 1,
            "identity": task_index_base_value + 2,
            "category": task_index_base_value + 3,
        }[feature], total = total_tasks)

        self.reset()

    def _reset(self, i):
        category, identity, position = self._set_random(self.dataset[i, 0])
        self.actions[i, 1], _, _, _ = self._set_data(
            self.dataset[i, 1], category, identity, position)

    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
