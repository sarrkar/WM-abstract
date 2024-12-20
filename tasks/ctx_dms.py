from typing import Literal
from tasks.base import TaskDataset
from tasks.utils import get_one_hot
from typing import List
from typing_extensions import Literal


class CtxDMSDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int = 128,
        features: List[Literal["category", "identity", "position"]] = ["category", "identity", "position"],
        pad_to: int = 0,
        category_size: int = 4,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        task_index_base_value: int = 36,
        total_tasks: int = 43
    ):
        task_len = max(6, pad_to)
        super(CtxDMSDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
            
        )

        self.features = features
        self.task_index = get_one_hot(
            task_index_base_value + 
            {
                ("position", "category", "identity"): 1,
                ("position", "identity", "category"): 2,
                ("identity", "position", "category"): 3,
                ("category", "identity", "position"): 4,
            }[tuple(features)]
        )

        self.reset()

    def _reset(self, i):
        category, identity, position = self._set_random(self.dataset[i, 0])

        self.feature = self.features[0]
        action, category, identity, position = self._set_data(
            self.dataset[i, 1], category, identity, position)
        self.actions[i, 1] = action

        if action == 1:
            self.feature = self.features[1]
            self.actions[i, 2], _, _, _ = self._set_data(
                self.dataset[i, 2], category, identity, position)
        elif action == 0:
            self.feature = self.features[2]
            self.actions[i, 2], _, _, _ = self._set_data(
                self.dataset[i, 2], category, identity, position)

    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
