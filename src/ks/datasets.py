from itertools import pairwise
from random import shuffle

import torch
from torch.utils.data import Dataset

from src.utils import sliding_window


class KSDataset(Dataset):
    def __init__(
        self,
        trajectory: list[torch.Tensor],
    ):
        self.xy_pairs = list(pairwise(trajectory))
        shuffle(self.xy_pairs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xy_pairs[index]

    def __len__(self) -> int:
        return len(self.xy_pairs)
    
    def get_horizon_size(self) -> int:
        return 1


class KSDatasetUnrolled(Dataset):
    def __init__(
        self,
        trajectory: list[torch.Tensor],
        *,
        unrolling_horizon: int = 3,
    ):
        assert isinstance(
            unrolling_horizon, int
        ), "Unrolling horizon must be an integer"
        assert (
            unrolling_horizon >= 1
        ), "Unrolling horizon must be at least 1 (which is equivalent to KSDataset)"

        self.xy_tuples = list(sliding_window(trajectory, unrolling_horizon + 1))
        self.unrolling_horizon = unrolling_horizon
        shuffle(self.xy_tuples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_y = self.xy_tuples[index]
        return X_y[0], torch.concat(X_y[1:])

    def __len__(self) -> int:
        return len(self.xy_tuples)

    def get_horizon_size(self) -> int:
        return self.unrolling_horizon


def get_train_test(
    traj_train: list[torch.Tensor],
    traj_test: list[torch.Tensor],
    unrolling_horizon: int = 3,
) -> tuple[Dataset, Dataset]:
    if unrolling_horizon > 1:
        return (
            KSDatasetUnrolled(traj_train, unrolling_horizon=unrolling_horizon),
            KSDatasetUnrolled(traj_test, unrolling_horizon=unrolling_horizon),
        )
    return KSDataset(traj_train), KSDataset(traj_test)
