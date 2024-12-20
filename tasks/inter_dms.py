from typing import Literal
from tasks.base import TaskDataset
from tasks.utils import get_one_hot


class InterDMSDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int = 128,
        feature_1: Literal["category", "identity", "position"] = "category",
        feature_2: Literal["category", "identity", "position"] = "category",
        pattern: Literal["AABB", "ABBA", "ABAB"] = "AABB",
        pad_to: int = 0,
        category_size: int = 4,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        task_index_base_value: int = 10,
        total_tasks: int = 43
    ):
        task_len = max(6, pad_to)
        super(InterDMSDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
        )

        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.pattern = pattern
        self.task_index = get_one_hot(
            task_index_base_value +
            {
                "AABB": 0,
                "ABAB": 1,
                "ABBA": 2,
                
            }[pattern] * 9 +
            {
                "category": 2,
                "identity": 1,
                "position": 0,
            }[feature_1] * 3 + 
            {
                "category": 2,
                "identity": 1,
                "position": 0,
            }[feature_2], total=total_tasks

        )

        self.reset()

    def _reset_AABB(self, i):
        self.feature = self.feature_1
        category, identity, position = self._set_random(self.dataset[i, 0])
        self.actions[i, 1], category, identity, position = self._set_data(
            self.dataset[i, 1], category, identity, position)

        self.feature = self.feature_2
        category, identity, position = self._set_random(self.dataset[i, 2])
        self.actions[i, 3], _, _, _ = self._set_data(
            self.dataset[i, 3], category, identity, position)

    def _reset_ABBA(self, i):
        self.feature = self.feature_1
        A_category, A_identity, A_position = self._set_random(
            self.dataset[i, 0])
        self.actions[i, 3], _, _, _ = self._set_data(
            self.dataset[i, 3], A_category, A_identity, A_position)

        self.feature = self.feature_2        
        B_category, B_identity, B_position = self._set_random(
            self.dataset[i, 1])

        self.actions[i, 2], _, _, _ = self._set_data(
            self.dataset[i, 2], B_category, B_identity, B_position)
        

    def _reset_ABAB(self, i):
        self.feature = self.feature_1
        A_category, A_identity, A_position = self._set_random(
            self.dataset[i, 0])
        self.actions[i, 2], _, _, _ = self._set_data(
            self.dataset[i, 2], A_category, A_identity, A_position)
        
        self.feature = self.feature_2
        B_category, B_identity, B_position = self._set_random(
            self.dataset[i, 1])
        self.actions[i, 3], _, _, _ = self._set_data(
            self.dataset[i, 3], B_category, B_identity, B_position)

    def reset(self):
        reset_func = {
            "AABB": self._reset_AABB,
            "ABBA": self._reset_ABBA,
            "ABAB": self._reset_ABAB,
        }[self.pattern]
        for i in range(self.dataset_size):
            reset_func(i)
