"""
metrics:
- mean_squared_error of NUM_STEPS_FUTURE autoregressive prediction
- energy of the solution compared to the energy of the dataset
- spectral difference
"""
from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    @abstractmethod
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        pass
