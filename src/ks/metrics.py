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

    def __init__(self, p: int = 2, mean_psd: torch.Tensor | None = None) -> None:
        super().__init__()
        self.p = p
        self.mean_psd = mean_psd

    @classmethod
    def compute_psd(cls, signal: torch.Tensor) -> torch.Tensor:
        n = signal.shape[-1]
        fft_result = torch.fft.fft(signal, n=n)
        psd = torch.abs(fft_result) ** 2 / n
        return psd[: n // 2 + 1]

    @classmethod
    def log_distance(
        cls, psd1: torch.Tensor, psd2: torch.Tensor, p: int
    ) -> torch.Tensor:
        assert psd2 is not None, "y_pred or mean_psd must be provided"
        log_psd1 = torch.log(psd1 + cls._EPS)
        log_psd2 = torch.log(psd2 + cls._EPS)
        return torch.norm(log_psd1 - log_psd2, p=p)

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor | None
    ) -> torch.Tensor:
        psd1 = self.compute_psd(y_true)
        psd2 = self.compute_psd(y_pred) if y_pred is not None else self.mean_psd
        return self.log_distance(psd1, psd2, self.p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def evaluate_trajectory(traj: torch.Tensor, mean_spectrum: torch.Tensor) -> float:
    # mean spectrum of the trajectory:
    spectra = []
    for u in traj:
        spectra.append(LogSpectralDistance.compute_psd(u))

    spectra = torch.stack(spectra)

    mean_spectrum_traj = spectra.mean(dim=0)

    return LogSpectralDistance.log_distance(
        mean_spectrum_traj, mean_spectrum, p=2
    ).item()
