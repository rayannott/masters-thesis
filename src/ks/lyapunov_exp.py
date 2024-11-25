import torch
from torch.autograd.functional import jacobian


def calculate_lyapunov_exponent_jac(
    u_init: torch.Tensor | None, solver, resolution: int, device: torch.device, num_steps: int = 1000, transient_steps: int = 100
) -> float:
    EPS = 1e-5
    if hasattr(solver, "eval"):
        solver.eval()
    else:
        print("solver does not have eval method")
    
    if u_init is None:
        # a quarter cosine wave
        u_init = torch.cos(torch.linspace(0, 2 * 3.1415, resolution)).to(device)

    epsilon = EPS * torch.randn_like(u_init).to(device)
    u_0 = u_init.unsqueeze(0).to(device)

    log_norm_sum = torch.Tensor([0]).to(device)

    for i in range(num_steps + transient_steps):
        u_next = solver(u_0)

        J = jacobian(solver, u_0).reshape(resolution, resolution).to(device) # type: ignore

        epsilon_next = J @ epsilon

        norm_epsilon_next = torch.norm(epsilon_next)

        if i >= transient_steps:
            log_norm_sum += torch.log(norm_epsilon_next)

        epsilon = epsilon_next / norm_epsilon_next

        u_0 = u_next

    lyapunov_exponent = log_norm_sum / (num_steps - transient_steps)
    return lyapunov_exponent.item()
