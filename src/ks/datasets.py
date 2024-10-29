from itertools import pairwise
from random import shuffle

import torch
from torch.utils.data import Dataset

from src.utils import sliding_window


class KSDataset(Dataset):
    def __init__(
        self, trajectory: list[torch.Tensor], *, train: bool, percent_train: float = 0.8
    ):
        self.xy_pairs = list(pairwise(trajectory))
        self.train = train
        self.percent_train = percent_train
        shuffle(self.xy_pairs)
        self.num_train = int(len(self.xy_pairs) * percent_train)
        self.xy_pairs_train = self.xy_pairs[: self.num_train]
        self.xy_pairs_test = self.xy_pairs[self.num_train :]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.xy_pairs_train if self.train else self.xy_pairs_test)[index]

    def __len__(self) -> int:
        return self.num_train if self.train else len(self.xy_pairs) - self.num_train

    @classmethod
    def get_train_test(
        cls, trajectory: list[torch.Tensor], percent_train: float = 0.8
    ) -> tuple[Dataset, Dataset]:
        """Return a tuple of training and testing datasets split according to the given percentage."""
        return cls(trajectory, train=True, percent_train=percent_train), cls(
            trajectory, train=False, percent_train=percent_train
        )


class KSDatasetUnrolled(Dataset):
    def __init__(
        self,
        trajectory: list[torch.Tensor],
        *,
        train: bool,
        percent_train: float = 0.8,
        unrolling_horizon: int = 3,
    ):
        assert isinstance(
            unrolling_horizon, int
        ), "Unrolling horizon must be an integer"
        assert (
            unrolling_horizon >= 1
        ), "Unrolling horizon must be at least 1 (which is equivalent to KSDataset)"

        self.xy_tuples = list(sliding_window(trajectory, unrolling_horizon + 1))
        self.train = train
        self.percent_train = percent_train
        self.unrolling_horizon = unrolling_horizon
        shuffle(self.xy_tuples)
        self.num_train = int(len(self.xy_tuples) * percent_train)
        self.xy_tuples_train = self.xy_tuples[: self.num_train]
        self.xy_tuples_test = self.xy_tuples[self.num_train :]
        self.xy_tuples_to_use = (
            self.xy_tuples_train if self.train else self.xy_tuples_test
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_y = self.xy_tuples_to_use[index]
        return X_y[0], torch.concat(X_y[1:])

    def __len__(self) -> int:
        return self.num_train if self.train else len(self.xy_tuples) - self.num_train

    def get_horizon_size(self) -> int:
        return self.unrolling_horizon

    @classmethod
    def get_train_test(
        cls,
        trajectory: list[torch.Tensor],
        percent_train: float = 0.8,
        unrolling_horizon: int = 3,
    ) -> tuple[Dataset, Dataset]:
        return cls(
            trajectory,
            train=True,
            percent_train=percent_train,
            unrolling_horizon=unrolling_horizon,
        ), cls(
            trajectory,
            train=False,
            percent_train=percent_train,
            unrolling_horizon=unrolling_horizon,
        )


def get_train_test(
    trajectory: list[torch.Tensor],
    percent_train: float = 0.8,
    unrolling_horizon: int = 3,
) -> tuple[Dataset, Dataset]:
    if unrolling_horizon > 1:
        return KSDatasetUnrolled.get_train_test(
            trajectory, percent_train=percent_train, unrolling_horizon=unrolling_horizon
        )
    return KSDataset.get_train_test(trajectory, percent_train=percent_train)
