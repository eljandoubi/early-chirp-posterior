"""Sequential update: the posterior at one fraction is the proposal for the next."""

import torch

from .reference import Likelihood
from .simulator import Simulator
from .smc import SMCResult, tempered_smc


def sequential_step(
    sim: Simulator,
    w_obs: torch.Tensor,
    n_prev: int,
    n_next: int,
    particles: torch.Tensor,
    log_q: torch.Tensor | None = None,
    n_moves: int = 6,
    ess_frac: float = 0.5,
    generator: torch.Generator | None = None,
    symmetrize: bool = True,
) -> SMCResult:
    """Move particles from the posterior of the first n_prev samples to n_next.

    particles are posterior samples at n_prev (log_q None) or draws from a proposal
    with unit cube log density log_q (the flow). The exact likelihood of the new
    samples gives the first importance weights; tempering and moves do the rest.
    """
    lik = Likelihood(sim, w_obs)

    def logp(theta):
        return lik.at(theta, n_prev, n_next)

    return tempered_smc(
        logp,
        particles,
        sim.lo,
        sim.hi,
        sim.periodic,
        log_q=log_q,
        ess_frac=ess_frac,
        n_moves=n_moves,
        generator=generator,
        image=sim.pair_image if symmetrize else None,
    )


def sequential_path(
    sim: Simulator,
    w_obs: torch.Tensor,
    path: list[int],
    first: torch.Tensor,
    first_log_q: torch.Tensor | None = None,
    n_moves: int = 6,
    generator: torch.Generator | None = None,
    ess_frac: float = 0.5,
) -> list[dict]:
    """Run sequential_step along increasing sample counts; one record per step."""
    particles, log_q, records = first, first_log_q, []
    for n_prev, n_next in zip(path[:-1], path[1:], strict=True):
        res = sequential_step(
            sim,
            w_obs,
            n_prev,
            n_next,
            particles,
            log_q,
            n_moves,
            ess_frac,
            generator=generator,
        )
        records.append(dict(n_obs=n_next, result=res))
        particles, log_q = res.samples, None
    return records
