"""Newtonian-order chirp seen through two rotating channels, plus exact whitening."""

import math

import torch

from .config import PERIODIC, T_SUN, Problem
from .noise import Noise


class Simulator:
    """Prior, waveform and whitened observations for one Problem on one device."""

    def __init__(self, problem: Problem | None = None, device="cpu"):
        self.problem = problem or Problem()
        self.device = torch.device(device)
        self.noise = Noise(self.problem, self.device)
        self.whiten = self.noise.whiten
        self.t = (
            torch.arange(self.problem.n, dtype=torch.float64, device=self.device)
            / self.problem.fs
        )
        lo, hi = self.problem.bounds()
        self.lo = torch.tensor(lo, dtype=torch.float64, device=self.device)
        self.hi = torch.tensor(hi, dtype=torch.float64, device=self.device)
        self.periodic = torch.zeros(5, dtype=torch.bool, device=self.device)
        self.periodic[list(PERIODIC)] = True

    def to_unit(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        return self.lo + u * (self.hi - self.lo)

    def sample_prior(self, n: int, generator: torch.Generator | None = None):
        u = torch.rand(
            n, 5, generator=generator, dtype=torch.float64, device=self.device
        )
        return self.from_unit(u)

    def phase_and_amplitude(self, theta: torch.Tensor, n: int | None = None):
        """GW phase and Newtonian amplitude on the first n grid points."""
        mc, tc, phic = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
        tau = tc - self.t[None, :n]
        chirp_time = mc * T_SUN
        phase = phic - 2.0 * (tau / (5.0 * chirp_time)) ** 0.625
        amp = self.problem.amplitude * (mc / 20.0) ** 1.25 * tau**-0.25
        return phase, amp

    def signal(self, theta: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Noise free two channel strain of shape (B, 2, n)."""
        phase, amp = self.phase_and_amplitude(theta, n)
        cosi, alpha = theta[:, 3:4], theta[:, 4:5]
        hp = amp * 0.5 * (1.0 + cosi**2) * torch.cos(phase)
        hx = amp * cosi * torch.sin(phase)
        psi2 = 2.0 * (alpha + self.problem.omega * self.t[None, :n])
        c, s = torch.cos(psi2), torch.sin(psi2)
        return torch.stack([c * hp + s * hx, -s * hp + c * hx], dim=1)

    def whitened_signal(self, theta: torch.Tensor, n: int | None = None):
        """Whitened signal of the first n samples (the whitener is causal)."""
        return self.signal(theta, n) @ self.whiten[:n, :n].T

    def observe(self, theta, n_obs, generator: torch.Generator | None = None):
        """Whitened data (B, 2, n): signal plus unit noise, zero after n_obs."""
        ws = self.whitened_signal(theta)
        z = torch.randn(
            ws.shape, generator=generator, dtype=ws.dtype, device=self.device
        )
        return (ws + z) * self.mask(n_obs)

    def mask(self, n_obs) -> torch.Tensor:
        n_obs = torch.as_tensor(n_obs, device=self.device).reshape(-1, 1)
        return (torch.arange(self.problem.n, device=self.device)[None, :] < n_obs)[
            :, None, :
        ]

    def snr(self, theta: torch.Tensor, n_obs) -> torch.Tensor:
        """Whitened network signal to noise ratio of the first n_obs samples."""
        ws = self.whitened_signal(theta) * self.mask(n_obs)
        return ws.pow(2).sum((1, 2)).sqrt()

    def frequency(self, theta: torch.Tensor) -> torch.Tensor:
        """Instantaneous GW frequency in Hz on the grid, shape (B, n)."""
        mc, tc = theta[:, 0:1], theta[:, 1:2]
        tau = tc - self.t[None, :]
        coef = (5.0 / 256.0) ** 0.375 / math.pi
        return coef * (mc * T_SUN) ** -0.625 * tau**-0.375
