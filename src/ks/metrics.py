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

    def __repr__(self) -> str:
        return self.__class__.__name__


class MeanSquaredError(Metric):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return torch.mean((y_true - y_pred) ** 2).item()


class LogSpectralDistance(Metric):
    _EPS = 1e-10

    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p

    @classmethod
    def compute_psd(cls, signal: torch.Tensor) -> torch.Tensor:
        n = signal.shape[-1]
        fft_result = torch.fft.fft(signal, n=n)
        psd = torch.abs(fft_result) ** 2 / n
        return psd[: n // 2 + 1]

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        psd1 = self.compute_psd(y_true)
        psd2 = self.compute_psd(y_pred)
        log_psd1 = torch.log10(psd1 + self._EPS)
        log_psd2 = torch.log10(psd2 + self._EPS)
        return torch.norm(log_psd1 - log_psd2, p=self.p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
