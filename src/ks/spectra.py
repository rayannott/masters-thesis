import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import plotly.graph_objects as go


def get_welsh_psd(
    x_np: np.ndarray, u_traj_numpy: np.ndarray, resolution: int, num_points: int = 4
) -> go.Figure:
    fig = go.Figure()

    for i in range(num_points):
        time_series = u_traj_numpy[i, :]
        freqs, psd = welch(time_series)
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd,
                mode="lines",
                name=f"x={x_np[int(i*resolution/num_points)]:.3f}",
            )
        )

    fig.update_layout(
        autosize=False,
        width=600,
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
        template="plotly_dark",
    )
    fig.update_yaxes(type="log")
    return fig


def get_energy_spectrum(
    u_traj_numpy: np.ndarray,
    resolution: int,
    domain_size: float,
    timestep: float,
    num_timepoints: int = 10,
) -> go.Figure:
    fig = go.Figure()

    for i in range(num_timepoints):
        space_series = u_traj_numpy[:, i]
        freqs = fftfreq(resolution, domain_size / resolution)
        spectrum = np.abs(fft(space_series)) ** 2 # type: ignore
        fig.add_trace(
            go.Scatter(
                x=freqs[: resolution // 2],
                y=spectrum[: resolution // 2],
                mode="lines",
                name=f"t={i*timestep:.3f}",
            )
        )

    fig.update_layout(
        autosize=False,
        width=600,
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
        template="plotly_dark",
    )
    fig.update_yaxes(type="log")
    return fig
