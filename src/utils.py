from collections import deque
from typing import TypeVar, Iterable, Generator
from itertools import islice


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
