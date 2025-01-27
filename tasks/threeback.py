from typing import Literal
from tasks.base import TaskDataset
from tasks.dms import get_one_hot


class ThreeBackDataset(TaskDataset):
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
        super(ThreeBackDataset, self).__init__(
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
            "category": 10,
            "identity": 11,
            "position": 12,
        }[feature])

        self.reset()

    def _reset(self, i):
      category_0, identity_0, position_0 = self._set_random(self.dataset[i, 0])
      category_1, identity_1, position_1 = self._set_random(self.dataset[i, 1])
      category_2, identity_2, position_2 = self._set_random(self.dataset[i, 2])
      self.actions[i, 3], category_3, identity_3, position_3 = self._set_data(self.dataset[i, 3], category_0, identity_0, position_0)
      self.actions[i, 4], category_4, identity_4, position_4 = self._set_data(self.dataset[i, 4], category_1, identity_1, position_1)
      self.actions[i, 5], category_5, identity_5, position_5 = self._set_data(self.dataset[i, 5], category_2, identity_2, position_2)


    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
