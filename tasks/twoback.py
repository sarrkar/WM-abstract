from typing import Literal
from tasks.base import TaskDataset
from tasks.dms import get_one_hot


class TwoBackDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int,
        feature: Literal["category", "identity", "position"] = "category",
        pad_to: int = 0,
        category_size: int = 2,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        add_noise: bool = False,
    ):
        task_len = max(6, pad_to)
        super(TwoBackDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
            add_noise=add_noise,
        )

        self.feature = feature
        self.task_index = get_one_hot({
            "category": 7,
            "identity": 8,
            "position": 9,
        }[feature])

        self.reset()

    def _reset(self, i):
      category, identity, position = self._set_random(self.dataset[i, 0])
      category_curr, identity_curr, position_curr = self._set_random(self.dataset[i, 1])
      for j in range(2, 6):
        self.actions[i, j], category_next, identity_next, position_next = self._set_data(self.dataset[i, j], category, identity, position)
        category, identity, position = category_curr, identity_curr, position_curr
        category_curr, identity_curr, position_curr = category_next, identity_next, position_next


    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
