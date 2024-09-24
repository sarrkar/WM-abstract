from typing import Literal
from tasks.base import TaskDataset
from tasks.utils import get_one_hot


class CtxDMSDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int,
        features: list[Literal["category", "identity", "position"]] = [
            "category", "identity", "position"],
        pad_to: int = 0,
        category_size: int = 2,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
    ):
        task_len = max(3, pad_to)
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
            {
                ("category", "identity", "position"): 7,
                ("position", "category", "identity"): 8,
                ("position", "identity", "category"): 9,
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
