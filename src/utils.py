from collections import deque
from typing import TypeVar, Iterable, Generator
from itertools import islice

import torch

T = TypeVar("T")


def sliding_window(iterable: Iterable[T], n) -> Generator[tuple[T, ...], None, None]:
    """Collect stream data into overlapping fixed-length chunks or blocks.

    itertools.pairwise(iterable) is eqivalent to sliding_window(iterable, 2)

    >>> list(sliding_window([1, 2, 3, 4, 5], 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    # see https://docs.python.org/3/library/itertools.html#itertools-recipes
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def train_test_split(
    traj: list[torch.Tensor],
    percent_train: float = 0.8,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # TODO: problem with the training data being sampled from earlier times
    n = len(traj)
    n_train = int(n * percent_train)
    train = traj[:n_train]
    test = traj[n_train:]
    return train, test
