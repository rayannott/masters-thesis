import torch


class DifferentiableKS:
    def __init__(
        self,
        resolution: int,
        domain_size: float,
        dt: float,
        device: torch.device,
        dealiasing=False,
    ):
        self.resolution = resolution
        self.domain_size = domain_size
        self.dt = dt
        self.dx = domain_size / resolution
        self.dealiasing = dealiasing
        self.device = device

        # Matrices for exponential timestepping
        self.wavenumbers = (
            torch.fft.fftfreq(resolution, self.dx, device=self.device) * 1j
        )
        self.L_mat = -(self.wavenumbers**2) - self.wavenumbers**4
        self.exp_lin = torch.exp(self.L_mat * dt)

        # Coefficients for RK2
        self.nonlinear_coef_1 = torch.where(
            self.L_mat == 0, dt, (self.exp_lin - 1) / self.L_mat
        )
        self.nonlinear_coef_2 = torch.where(
            self.L_mat == 0,
            dt / 2,
            (self.exp_lin - 1 - self.L_mat * dt) / (dt * self.L_mat**2),
        )

    # Step function for temporal evolution, using exponential timestepping for linear tearms and RK2 for nonlinear terms
    def etrk2(self, u):
        if self.dealiasing:
            u = self.dealias(u)
        nonlin_current = self.calc_nonlinear(u)
        u_interm = (
            self.exp_lin * torch.fft.fftn(u) + nonlin_current * self.nonlinear_coef_1
        )
        u_new = (
            u_interm
            + (self.calc_nonlinear(torch.fft.ifftn(u_interm)) - nonlin_current)
            * self.nonlinear_coef_2
        )
        return torch.fft.ifftn(u_new).real

    def calc_nonlinear(self, u):
        return -0.5 * self.wavenumbers * torch.fft.fftn(u**2)

    def dealias(self, u):
        # filter out the largest third of wavenumbers in the 1D spectrum
        u_hat = torch.fft.fftn(u)
        u_hat[..., self.resolution // 6 : 2 * self.resolution // 6] = 0
        return torch.fft.ifftn(u_hat).real

    def get_1d_grid(self) -> torch.Tensor:
        return (
            self.domain_size
            * torch.arange(0, self.resolution, device=self.device)
            / self.resolution
        )
