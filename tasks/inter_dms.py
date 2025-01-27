from typing import Literal
from tasks.base import TaskDataset
from tasks.utils import get_one_hot


class InterDMSDataset(TaskDataset):
    def __init__(
        self,
        dataset_size: int,
        features: list[Literal["category", "identity", "position"]] = [
            "category", "identity", "position"],
        pattern: Literal["AABB", "ABBA", "ABAB"] = "AABB",
        pad_to: int = 0,
        category_size: int = 2,
        identity_size: int = 2,
        position_size: int = 4,
        std: float = 0,
        add_noise: bool = False,
    ):
        task_len = max(4, pad_to)
        super(InterDMSDataset, self).__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            std=std,
            add_noise=add_noise,
        )

        self.features = features
        self.pattern = pattern
        self.task_index = get_one_hot(
            16 +
            {
                "AABB": 0,
                "ABBA": 1,
                "ABAB": 2,
            }[pattern] * 9 +
            {
                ("category", "category",): 1,
                ("category", "identity",): 2,
                ("category", "position",): 3,
                ("identity", "category",): 4,
                ("identity", "identity",): 5,
                ("identity", "position",): 6,
                ("position", "category",): 7,
                ("position", "identity",): 8,
                ("position", "position",): 9,
            }[tuple(features)]
        )

        self.reset()

    def _reset_AABB(self, i):
        category, identity, position = self._set_random(self.dataset[i, 0])
        self.feature = self.features[0]
        self.actions[i, 1], category, identity, position = self._set_data(
            self.dataset[i, 1], category, identity, position)

        category, identity, position = self._set_random(self.dataset[i, 2])
        self.feature = self.features[1]
        self.actions[i, 3], _, _, _ = self._set_data(
            self.dataset[i, 3], category, identity, position)

    def _reset_ABBA(self, i):
        A_category, A_identity, A_position = self._set_random(
            self.dataset[i, 0])
        B_category, B_identity, B_position = self._set_random(
            self.dataset[i, 1])

        self.feature = self.features[1]
        self.actions[i, 2], _, _, _ = self._set_data(
            self.dataset[i, 2], B_category, B_identity, B_position)
        self.feature = self.features[0]
        self.actions[i, 3], _, _, _ = self._set_data(
            self.dataset[i, 3], A_category, A_identity, A_position)

    def _reset_ABAB(self, i):
        A_category, A_identity, A_position = self._set_random(
            self.dataset[i, 0])
        B_category, B_identity, B_position = self._set_random(
            self.dataset[i, 1])

        self.feature = self.features[0]
        self.actions[i, 2], _, _, _ = self._set_data(
            self.dataset[i, 2], A_category, A_identity, A_position)
        self.feature = self.features[1]
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
