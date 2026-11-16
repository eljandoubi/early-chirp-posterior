import math

import torch

from chirpvi.smc import ess_of, next_step, systematic_resample, tempered_smc
from chirpvi.util import circular_mean, circular_std

F64 = torch.float64


def run(logp, d, periodic, n=2048, **kw):
    gen = torch.Generator().manual_seed(0)
    x0 = torch.rand(n, d, generator=gen, dtype=F64)
    lo, hi = torch.zeros(d, dtype=F64), torch.ones(d, dtype=F64)
    return tempered_smc(logp, x0, lo, hi, periodic, generator=gen, **kw)


def test_two_gaussian_modes_weights_means_and_evidence():
    m1 = torch.tensor([0.3, 0.3, 0.3], dtype=F64)
    m2 = torch.tensor([0.7, 0.6, 0.5], dtype=F64)
    s1 = torch.diag(torch.tensor([0.04, 0.05, 0.06], dtype=F64) ** 2)
    c, s = math.cos(0.7), math.sin(0.7)
    rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=F64)
    s2 = rot @ torch.diag(torch.tensor([0.02, 0.08, 0.05], dtype=F64) ** 2) @ rot.T
    p1, p2 = torch.linalg.inv(s1), torch.linalg.inv(s2)

    def logp(x):
        q1 = ((x - m1) @ p1 * (x - m1)).sum(1)
        q2 = ((x - m2) @ p2 * (x - m2)).sum(1)
        ell = torch.logsumexp(torch.stack([-0.5 * q1, -0.5 * q2]), 0) + math.log(0.5)
        return torch.zeros_like(ell), ell

    res = run(logp, 3, torch.zeros(3, dtype=torch.bool))
    c1 = torch.linalg.det(s1).sqrt() * (2 * math.pi) ** 1.5
    c2 = torch.linalg.det(s2).sqrt() * (2 * math.pi) ** 1.5
    assert abs(res.log_evidence - math.log(0.5 * (c1 + c2))) < 0.2
    first = res.samples[:, 0] < 0.5
    assert abs(first.double().mean().item() - (c1 / (c1 + c2)).item()) < 0.06
    mean1 = res.samples[first].mean(0)
    std1 = res.samples[first].std(0)
    assert ((mean1 - m1).abs() < 0.2 * torch.sqrt(torch.diag(s1))).all()
    assert (std1 / torch.sqrt(torch.diag(s1)) - 1).abs().max() < 0.2
    assert res.stages > 2 and res.n_evals > 2048


def test_periodic_bump_across_the_boundary():
    kappa = 50.0

    def logp(x):
        ell = kappa * (torch.cos(2 * math.pi * x[:, 0]) - 1.0)
        ell = ell - 0.5 * ((x[:, 1] - 0.5) / 0.1) ** 2
        return torch.zeros_like(ell), ell

    res = run(logp, 2, torch.tensor([True, False]))
    angle = res.samples[:, 0]
    dist = (circular_mean(angle) + 0.5) % 1.0 - 0.5
    assert abs(dist.item()) < 0.01
    r = torch.special.i1e(torch.tensor(kappa)) / torch.special.i0e(torch.tensor(kappa))
    expect = torch.sqrt(-2 * torch.log(r)).item()
    assert abs(circular_std(angle).item() / expect - 1) < 0.08
    truth = math.log(
        torch.special.i0e(torch.tensor(kappa)).item() * 0.1 * math.sqrt(2 * math.pi)
    )
    assert abs(res.log_evidence - truth) < 0.15


def test_proposal_weights_correct_a_wrong_proposal():
    mean, scale = 0.5, 0.1

    def logp(x):
        ell = -0.5 * (((x - mean) / scale) ** 2).sum(1)
        return ell, ell

    gen = torch.Generator().manual_seed(3)
    prop_mean, prop_scale = 0.56, 0.15
    x0 = prop_mean + prop_scale * torch.randn(4096, 2, generator=gen, dtype=F64)
    log_q = -0.5 * (((x0 - prop_mean) / prop_scale) ** 2).sum(1)
    log_q = log_q - 2 * math.log(prop_scale * math.sqrt(2 * math.pi))
    lo, hi = torch.zeros(2, dtype=F64), torch.ones(2, dtype=F64)
    res = tempered_smc(
        logp, x0, lo, hi, torch.zeros(2, dtype=torch.bool), log_q=log_q, generator=gen
    )
    assert abs(res.log_evidence - 2 * math.log(scale * math.sqrt(2 * math.pi))) < 0.1
    assert (res.samples.mean(0) - mean).abs().max() < 0.01
    assert 0.3 < res.ess_first < 0.99


def test_step_size_and_resampling_helpers():
    assert next_step(torch.zeros(100, dtype=F64), 0.7, 50.0) == 0.7
    inc = torch.linspace(0, 40, 200, dtype=F64)
    delta = next_step(inc, 1.0, 100.0)
    assert 0 < delta < 1
    assert abs(ess_of(delta * inc).item() - 100.0) < 1.0
    logw = torch.log(torch.tensor([0.5, 0.25, 0.25, 0.0], dtype=F64) + 1e-300)
    idx = systematic_resample(logw, torch.Generator().manual_seed(0))
    assert idx.numel() == 4 and (idx != 3).all()


def test_symmetry_move_balances_images_and_keeps_the_evidence():
    def logp(x):
        bump = lambda c: torch.exp(-0.5 * ((x[:, 0] - c) / 0.03) ** 2)  # noqa: E731
        ell = torch.log(0.5 * bump(0.25) + 0.5 * bump(0.75))
        ell = ell - 0.5 * ((x[:, 1] - 0.5) / 0.05) ** 2
        return torch.zeros_like(ell), ell

    def image(x):
        return torch.stack([1.0 - x[:, 0], x[:, 1]], 1)

    res = run(logp, 2, torch.zeros(2, dtype=torch.bool), image=image)
    left = (res.samples[:, 0] < 0.5).double().mean().item()
    assert abs(left - 0.5) < 0.03
    truth = math.log(0.03 * 0.05 * 2 * math.pi)
    assert abs(res.log_evidence - truth) < 0.15
