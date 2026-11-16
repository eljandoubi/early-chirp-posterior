"""Tempered sequential Monte Carlo with affine invariant ensemble moves."""

import math
import time
from dataclasses import dataclass

import torch


@dataclass
class SMCResult:
    samples: torch.Tensor
    log_evidence: float
    n_evals: int
    stages: int
    seconds: float
    ess_first: float


def ess_of(logw: torch.Tensor) -> torch.Tensor:
    w = torch.exp(logw - logw.max())
    return w.sum() ** 2 / w.pow(2).sum()


def next_step(inc: torch.Tensor, room: float, target: float) -> float:
    """Largest step in (0, room] whose incremental weights keep ESS >= target."""
    if ess_of(room * inc) >= target:
        return room
    lo, hi = 0.0, room
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess_of(mid * inc) >= target:
            lo = mid
        else:
            hi = mid
    return max(lo, 1e-6)


def systematic_resample(logw: torch.Tensor, generator=None) -> torch.Tensor:
    w = torch.exp(logw - logw.max())
    cdf = torch.cumsum(w / w.sum(), 0)
    cdf[-1] = 1.0
    n = logw.numel()
    start = torch.rand(1, generator=generator, dtype=cdf.dtype, device=cdf.device)
    points = (torch.arange(n, dtype=cdf.dtype, device=cdf.device) + start) / n
    return torch.searchsorted(cdf, points).clamp_max(n - 1)


def stretch_move(x, ls, le, beta, logp, lo, hi, periodic, generator, counter):
    """One sweep of the stretch move (a = 2) over two random half ensembles."""
    n, d = x.shape
    span = hi - lo
    order = torch.randperm(n, generator=generator, device=x.device)
    for active, other in (
        (order[: n // 2], order[n // 2 :]),
        (order[n // 2 :], order[: n // 2]),
    ):
        xa = x[active]
        k = active.numel()
        pick = torch.randint(
            0, other.numel(), (k,), generator=generator, device=x.device
        )
        partner = x[other[pick]]
        u = torch.rand(k, generator=generator, dtype=x.dtype, device=x.device)
        z = (u + 1.0) ** 2 / 2.0
        diff = xa - partner
        diff = torch.where(periodic, diff - span * torch.round(diff / span), diff)
        step = z[:, None] * diff
        wide = ((step.abs() >= 0.5 * span) & periodic).any(1)
        prop = partner + step
        prop = torch.where(periodic, lo + torch.remainder(prop - lo, span), prop)
        inside = (((prop >= lo) & (prop <= hi)) | periodic).all(1) & ~wide
        trial = torch.where(inside[:, None], prop, xa)
        ls_new, le_new = logp(trial)
        counter[0] += k
        new = (1.0 - beta) * ls_new + beta * le_new
        old = (1.0 - beta) * ls[active] + beta * le[active]
        log_acc = (d - 1) * torch.log(z) + new - old
        log_u = torch.log(
            torch.rand(k, generator=generator, dtype=x.dtype, device=x.device)
        )
        accept = inside & (log_u < log_acc) & ~torch.isnan(log_acc)
        x[active] = torch.where(accept[:, None], prop, xa)
        ls[active] = torch.where(accept, ls_new, ls[active])
        le[active] = torch.where(accept, le_new, le[active])


def tempered_smc(
    logp,
    particles,
    lo,
    hi,
    periodic,
    log_q=None,
    ess_frac=0.5,
    n_moves=6,
    generator=None,
    max_stages=400,
    image=None,
):
    """Move particles from pi_start to pi_end along (1 - b) log_start + b log_end.

    logp maps (B, d) points to (log_start, log_end), both unnormalised and
    relative to the prior box. particles come from a proposal: pi_start itself
    when log_q is None, otherwise a density with log values log_q. The returned
    log evidence is log of the end mass relative to the prior when log_q is
    given, and of the end mass relative to the start mass when it is None.
    image, if given, is an exact symmetry of both densities (an involution that
    keeps the prior box): after every stage each particle is mapped through it
    with probability one half, which is a valid move and balances the images.
    """
    t0 = time.perf_counter()
    x = particles.clone()
    n = x.shape[0]
    counter = [n]
    ls, le = logp(x)
    log_z = 0.0
    ess_first = 1.0
    if log_q is not None:
        logw = ls - log_q
        log_z = (torch.logsumexp(logw, 0) - math.log(n)).item()
        ess_first = (ess_of(logw) / n).item()
        idx = systematic_resample(logw, generator)
        x, ls, le = x[idx].clone(), ls[idx].clone(), le[idx].clone()
    beta, stages = 0.0, 0
    while beta < 1.0 and stages < max_stages:
        inc = le - ls
        delta = next_step(inc, 1.0 - beta, ess_frac * n)
        logw = delta * inc
        log_z += (torch.logsumexp(logw, 0) - math.log(n)).item()
        beta = 1.0 if beta + delta >= 1.0 - 1e-12 else beta + delta
        idx = systematic_resample(logw, generator)
        x, ls, le = x[idx].clone(), ls[idx].clone(), le[idx].clone()
        for _ in range(n_moves):
            stretch_move(x, ls, le, beta, logp, lo, hi, periodic, generator, counter)
        if image is not None:
            flip = torch.rand(n, generator=generator, dtype=x.dtype, device=x.device)
            x = torch.where((flip < 0.5)[:, None], image(x), x)
        stages += 1
    return SMCResult(
        samples=x,
        log_evidence=log_z,
        n_evals=counter[0],
        stages=stages,
        seconds=time.perf_counter() - t0,
        ess_first=ess_first,
    )
