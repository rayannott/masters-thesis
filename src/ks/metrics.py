"""
metrics:
- mean_squared_error of NUM_STEPS_FUTURE autoregressive prediction
- energy of the solution compared to the energy of the dataset
- spectral difference
"""

from abc import ABC, abstractmethod
from statistics import mean
from typing import Callable

import torch

from src.ks.kuramoto_sivashinsky import DifferentiableKS


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
        log_psd1 = torch.log(psd1 + cls._EPS).to(psd1.device)
        log_psd2 = torch.log(psd2 + cls._EPS).to(psd2.device)
        return torch.norm(log_psd1 - log_psd2, p=p)

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor | None
    ) -> torch.Tensor:
        psd1 = self.compute_psd(y_true)
        psd2 = self.compute_psd(y_pred) if y_pred is not None else self.mean_psd
        return self.log_distance(psd1, psd2, self.p)  # type: ignore

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


def evaluate_model_cum_mse(
    u_init: torch.Tensor,
    model: torch.nn.Module,
    solver_ks: DifferentiableKS,
    n_steps_future: int,
    burn_in_steps: int,
) -> tuple[float, dict]:
    """
    Evaluate the cumulative mean squared error of the model's autoregressive prediction.

    Returns the cumulative mean squared error and an
    info dictionary containing
    - the errors,
    - the predicted trajectory and
    - whether the evaluation was successful.
    """

    def model_f(u: torch.Tensor) -> torch.Tensor:
        return model(u.unsqueeze(0)).squeeze(0)

    def stop_early(u_pred: torch.Tensor) -> bool:
        # TODO: perhaps add a condition to stop if the value doesn't change?
        return bool(torch.isnan(u_pred).any().item()) or bool(
            torch.isinf(u_pred).any().item()
        )

    u_pred_traj = [u_init]
    for _ in range(burn_in_steps):
        u_pred_traj.append(solver_ks.etrk2(u_pred_traj[-1]))

    u_pred_traj = [u_pred_traj[-1]]
    u_pred_traj.append(model_f(u_pred_traj[-1]))

    ok = True
    errors: list[float] = []

    for _ in range(n_steps_future - 1):
        u1, u2 = u_pred_traj[-2:]
        u2_solv = solver_ks.etrk2(u1)
        errors.append(((u2_solv - u2) ** 2).mean().item())
        new_val = model_f(u2)
        u_pred_traj.append(new_val)

        if stop_early(new_val):
            ok = False
            break

    info = {"errors": errors, "traj_pred": u_pred_traj, "ok": ok}
    return mean(errors) / solver_ks.dt if ok else torch.inf, info


def evaluate_model_cum_mse_with_ds(
    u_init: torch.Tensor,
    model: torch.nn.Module,
    solver_ks: DifferentiableKS,
    n_steps_future: int,
    burn_in_steps: int,
    domain_encoding_func: Callable[[torch.Tensor, float], torch.Tensor],
    domain_size: float,
) -> tuple[float, dict]:
    """
    Evaluate the cumulative mean squared error of the model's autoregressive prediction.

    Returns the cumulative mean squared error and an
    info dictionary containing
    - the errors,
    - the predicted trajectory and
    - whether the evaluation was successful.
    """

    def model_f(u: torch.Tensor) -> torch.Tensor:
        return model.forward(domain_encoding_func(u, domain_size).T)[0, :]

    def err(u1: torch.Tensor, u2: torch.Tensor) -> float:
        return ((u1 - u2) ** 2).mean().item()

    u_pred_traj = [u_init]

    for _ in range(burn_in_steps):
        u_pred_traj.append(solver_ks.etrk2(u_pred_traj[-1]))

    u_pred_traj = [u_pred_traj[-1]]
    u_pred_traj.append(model_f(u_pred_traj[-1]))

    ok = True
    reason = ''
    errors: list[float] = []

    for _ in range(n_steps_future - 1):
        u1, u2 = u_pred_traj[-2:]
        if err(u1, u2) < 1e-4:
            reason = "Prediction is stuck; stopping early"
            ok = False
            break
        u2_solv = solver_ks.etrk2(u1)
        if (_new_err:=err(u2_solv, u2)) > 1e-1:
            reason = f"Error is too large: {_new_err}"
            ok = False
            break
        errors.append(_new_err)

        new_val = model_f(u2)
        u_pred_traj.append(new_val)

        if (
            bool(torch.isnan(new_val).any().item())
            or bool(torch.isinf(new_val).any().item())
            or bool(torch.any(new_val > 1e3).item())
        ):
            ok = False
            break

    info = {"errors": errors, "traj_pred": u_pred_traj, "ok": ok, "reason": reason}
    return mean(errors) / solver_ks.dt if ok else torch.inf, info
