from typing import Literal
from tasks.base import TaskDataset
from tasks.dms import get_one_hot


class NBackDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int = 128,
        feature: Literal["category", "identity", "position"] = "category",
        nback_n: int = 1, 
        pad_to: int = 0,
        category_size: int = 4,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        task_index_base_value: int = 0,
        total_tasks: int = 43
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
        self.nback_n = nback_n
        self.task_index = get_one_hot({
            "1-position": task_index_base_value + 1,
            "1-identity": task_index_base_value + 2,
            "1-category": task_index_base_value + 3,
            "2-position": task_index_base_value + 4,
            "2-identity": task_index_base_value + 5,
            "2-category": task_index_base_value + 6,
            "3-position": task_index_base_value + 7,
            "3-identity": task_index_base_value + 8,
            "3-category": task_index_base_value + 9,
        }[f"{self.nback_n}-{self.feature}"], total=total_tasks)

        self.reset()

    def _reset(self, i):
        categories = []
        identities = []
        positions = []
        
        for j in range(self.nback_n):
            # randomly set the first nback_n frames
            category, identity, position = self._set_random(self.dataset[i, j])
            categories.append(category)
            identities.append(identity)
            positions.append(position)
           
        for j in range(self.nback_n, self.task_len):
            # set the rest of the frames according to the nback
            self.actions[i, j], category, identity, position = self._set_data(
                self.dataset[i, j], categories[j-self.nback_n], identities[j-self.nback_n], positions[j-self.nback_n])
            categories.append(category)
            identities.append(identity)
            positions.append(position)

    def reset(self):
        for i in range(self.dataset_size):
            self._reset(i)
