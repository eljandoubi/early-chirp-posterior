"""Calibration, agreement and mode metrics for posterior samples in unit coordinates."""

import math

import torch
from torch import nn

from .smc import ess_of
from .util import circular_mean, circular_std, unit_distance, wrap_unit

FUNCTION_NAMES = (
    "chirp_mass",
    "coalescence_time",
    "cos_inclination",
    "cos_phase",
    "sin_phase",
    "cos_2sky",
    "sin_2sky",
)
LINEAR = (0, 1, 3)


def summaries(u: torch.Tensor) -> torch.Tensor:
    """Fixed scalar functions of the parameters (valid on the circle): (..., 7)."""
    two_pi = 2 * math.pi
    return torch.stack(
        [
            u[..., 0],
            u[..., 1],
            u[..., 3],
            torch.cos(two_pi * u[..., 2]),
            torch.sin(two_pi * u[..., 2]),
            torch.cos(two_pi * u[..., 4]),
            torch.sin(two_pi * u[..., 4]),
        ],
        -1,
    )


def sbc_ranks(truth: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Rank of each summary of the truth among the samples: (B, 7), values 0..L."""
    g_true = summaries(truth)[:, None, :]
    return (summaries(samples) < g_true).sum(1)


def kolmogorov_pvalue(d: float, n: int) -> float:
    """Asymptotic p value of the Kolmogorov distance d for n observations."""
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 0.2:
        return 1.0
    total = sum(
        2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101)
    )
    return float(min(1.0, max(0.0, total)))


def uniform_ks(ranks: torch.Tensor, n_samples: int) -> dict:
    """Kolmogorov distance of ranks (0..L) to the discrete uniform and its p value."""
    counts = torch.bincount(ranks.long(), minlength=n_samples + 1).double()
    ecdf = torch.cumsum(counts, 0) / counts.sum()
    ideal = torch.arange(1, n_samples + 2, dtype=torch.float64) / (n_samples + 1)
    d = (ecdf - ideal).abs().max().item()
    return dict(ks=d, p_value=kolmogorov_pvalue(d, int(counts.sum().item())))


def hpd_coverage(log_q_truth, log_q_samples, levels) -> dict:
    """Share of cases whose truth lies in the c highest density region."""
    above = (log_q_samples > log_q_truth[:, None]).double().mean(1)
    return {c: (above <= c).double().mean().item() for c in levels}


def central_coverage(truth, samples, levels) -> dict:
    """Central interval coverage per summary: {level: tensor of shape (k,)}."""
    out = {}
    for c in levels:
        q = torch.tensor([(1 - c) / 2, (1 + c) / 2], dtype=samples.dtype)
        lo, hi = torch.quantile(samples, q, dim=1)
        out[c] = ((truth >= lo) & (truth <= hi)).double().mean(0)
    return out


def tarp(truth, samples, reference, periodic, n_grid: int = 101):
    """Expected coverage curve of TARP (Lemos et al.): returns (levels, ecp).

    ecp(c) is the share of cases with fewer than a share c of the samples closer to
    a random reference point than the truth. Below the diagonal means overconfident.
    """
    d_true = unit_distance(truth, reference, periodic).norm(dim=-1)
    d_samp = unit_distance(samples, reference[:, None, :], periodic).norm(dim=-1)
    frac = (d_samp < d_true[:, None]).double().mean(1)
    levels = torch.linspace(0, 1, n_grid, dtype=torch.float64)
    ecp = (frac[None, :] < levels[:, None]).double().mean(1)
    return levels, ecp


def tarp_summary(levels, ecp) -> dict:
    gap = ecp - levels
    return dict(max_deviation=gap.abs().max().item(), mean_deviation=gap.mean().item())


def features(u: torch.Tensor, periodic: torch.Tensor) -> torch.Tensor:
    cols = []
    for i in range(u.shape[1]):
        if periodic[i]:
            cols += [torch.cos(2 * math.pi * u[:, i]), torch.sin(2 * math.pi * u[:, i])]
        else:
            cols.append(u[:, i])
    return torch.stack(cols, 1)


def c2st(a, b, periodic, generator=None, steps: int = 300, hidden: int = 64) -> float:
    """Classifier two sample test, held out accuracy; 0.5 is indistinguishable."""
    x = torch.cat([features(a, periodic), features(b, periodic)]).float()
    y = torch.cat([torch.zeros(a.shape[0]), torch.ones(b.shape[0])]).float()
    y = y.to(x.device)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    order = torch.randperm(x.shape[0], generator=generator).to(x.device)
    half = x.shape[0] // 2
    correct = 0
    for train, test in ((order[:half], order[half:]), (order[half:], order[:half])):
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            net = nn.Sequential(
                nn.Linear(x.shape[1], hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        net = net.to(x.device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(steps):
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(x[train])[:, 0], y[train]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (net(x[test])[:, 0] > 0).float()
        correct += (pred == y[test]).sum().item()
    return correct / x.shape[0]


def pair_image(u: torch.Tensor) -> torch.Tensor:
    """The exact symmetry: phase + pi and sky angle + pi/2, in unit coordinates."""
    out = u.clone()
    out[..., 2] = wrap_unit(out[..., 2] + 0.5)
    out[..., 4] = wrap_unit(out[..., 4] + 0.5)
    return out


def mode_labels(u: torch.Tensor) -> torch.Tensor:
    """0 for samples within a quarter period of the circular centre of 2 * sky, else 1.

    2 * u_sky (mod 1) is unchanged by the pair symmetry, so its circular mean
    locates one of the two mode centres and the other lies half a period away.
    """
    sky = u[:, 4]
    centre = circular_mean(wrap_unit(2 * sky), 0) / 2
    d = (sky - centre).abs()
    return (torch.minimum(d, 1 - d) >= 0.25).long()


def mode_summary(u: torch.Tensor) -> dict:
    """Share of the second mode and the sky angle std (radians) inside each mode."""
    labels = mode_labels(u)
    out = dict(weight=labels.double().mean().item())
    for m in (0, 1):
        sel = u[labels == m, 4]
        std = circular_std(sel, 0).item() / 2 if sel.numel() > 2 else float("nan")
        out[f"sky_std_{m}"] = std
    return out


def is_bimodal(
    summary: dict, lo: float = 0.4, hi: float = 0.6, max_std: float = 0.3
) -> bool:
    stds = (summary["sky_std_0"], summary["sky_std_1"])
    return lo <= summary["weight"] <= hi and all(s < max_std for s in stds)


def widths(u: torch.Tensor) -> dict:
    """Std over the prior std for the linear parameters."""
    scale = math.sqrt(12.0)
    return {
        name: u[:, i].std().item() * scale
        for name, i in (
            ("chirp_mass", 0),
            ("coalescence_time", 1),
            ("cos_inclination", 3),
        )
    }


def importance_efficiency(log_target: torch.Tensor, log_q: torch.Tensor) -> float:
    """ESS over N of weights target / q for samples drawn from q."""
    return (ess_of(log_target - log_q) / log_q.numel()).item()
