"""Exact Gaussian likelihood and the reference posterior built on the sampler."""

import math

import torch

from .simulator import Simulator
from .smc import SMCResult, tempered_smc


class Likelihood:
    """Log likelihood ratio against noise only, for whitened data w of shape (2, n).

    The whitener is causal, so one pass gives the exact value for every prefix.
    """

    def __init__(self, sim: Simulator, w_obs: torch.Tensor):
        self.sim = sim
        self.w = w_obs.to(sim.device)

    def cumulative(self, theta: torch.Tensor, n: int) -> torch.Tensor:
        """Log likelihood ratio of the first i + 1 samples for i < n, shape (B, n)."""
        ws = self.sim.whitened_signal(theta, n)
        per_sample = (ws * self.w[:, :n]).sum(1) - 0.5 * ws.pow(2).sum(1)
        return torch.cumsum(per_sample, dim=1)

    def at(self, theta: torch.Tensor, *n_obs: int) -> tuple[torch.Tensor, ...]:
        cum = self.cumulative(theta, max(n_obs))
        return tuple(cum[:, n - 1] for n in n_obs)


def reference_posterior(
    sim: Simulator,
    w_obs: torch.Tensor,
    n_obs: int,
    n_particles: int = 1024,
    n_moves: int = 6,
    ess_frac: float = 0.5,
    generator: torch.Generator | None = None,
) -> SMCResult:
    """Posterior samples of the exact likelihood, from the prior by tempering."""
    lik = Likelihood(sim, w_obs)

    def logp(theta):
        (ell,) = lik.at(theta, n_obs)
        return torch.zeros_like(ell), ell

    particles = sim.sample_prior(n_particles, generator)
    return tempered_smc(
        logp,
        particles,
        sim.lo,
        sim.hi,
        sim.periodic,
        ess_frac=ess_frac,
        n_moves=n_moves,
        generator=generator,
    )


def angle_grid_posterior(sim, w_obs, n_obs, fixed, n_grid=160):
    """Exact posterior of (phic, alpha) on a grid with the other parameters fixed.

    Returns the cell probabilities (n_grid, n_grid), the cell edges of both
    angles and the log evidence relative to noise only (prior mean of the
    likelihood over the torus).
    """
    lik = Likelihood(sim, w_obs)
    pc = (torch.arange(n_grid, dtype=torch.float64, device=sim.device) + 0.5) / n_grid
    phic = sim.lo[2] + pc * (sim.hi[2] - sim.lo[2])
    alpha = sim.lo[4] + pc * (sim.hi[4] - sim.lo[4])
    grid = torch.stack(torch.meshgrid(phic, alpha, indexing="ij"), -1).reshape(-1, 2)
    theta = fixed.repeat(grid.shape[0], 1)
    theta[:, 2], theta[:, 4] = grid[:, 0], grid[:, 1]
    (ell,) = lik.at(theta, n_obs)
    log_z = torch.logsumexp(ell, 0) - math.log(ell.numel())
    prob = torch.softmax(ell, 0).reshape(n_grid, n_grid)
    return prob, log_z.item()


def angle_posterior(sim, w_obs, n_obs, fixed, n_particles=1024, generator=None):
    """The same two dimensional posterior from the tempered sampler."""
    lik = Likelihood(sim, w_obs)
    free = [2, 4]

    def logp(x):
        theta = fixed.repeat(x.shape[0], 1)
        theta[:, free] = x
        (ell,) = lik.at(theta, n_obs)
        return torch.zeros_like(ell), ell

    lo, hi = sim.lo[free], sim.hi[free]
    unit = torch.rand(
        n_particles, 2, generator=generator, dtype=lo.dtype, device=lo.device
    )
    x0 = lo + unit * (hi - lo)
    return tempered_smc(
        logp,
        x0,
        lo,
        hi,
        sim.periodic[free],
        generator=generator,
    )
