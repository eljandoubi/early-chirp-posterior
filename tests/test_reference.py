import math

import torch

from chirpvi.reference import (
    Likelihood,
    angle_grid_posterior,
    angle_posterior,
    reference_posterior,
)
from chirpvi.simulator import Simulator


def injection(sim, n_obs, seed=0, cosi=0.4):
    gen = torch.Generator().manual_seed(seed)
    theta = sim.sample_prior(1, gen)
    theta[0, 3] = cosi
    theta[0, 0], theta[0, 1] = 21.0, 2.34
    theta[0, 2], theta[0, 4] = 1.0, 0.9
    w = sim.observe(theta, torch.tensor([n_obs]), gen)[0]
    return theta, w


def test_likelihood_matches_the_direct_coloured_noise_formula():
    sim = Simulator()
    n = 100
    gen = torch.Generator().manual_seed(1)
    theta = sim.sample_prior(3, gen)
    w = sim.observe(theta[:1], torch.tensor([n]), gen)[0]
    cov = sim.noise.cov[:n, :n]
    data = (torch.linalg.cholesky(cov) @ w[:, :n].T).T
    sig = sim.signal(theta, n)
    got = Likelihood(sim, w).at(theta, n)[0]
    for i in range(3):
        sol = torch.linalg.solve(cov, sig[i].T).T
        direct = (sol * data).sum() - 0.5 * (sol * sig[i]).sum()
        assert abs(direct - got[i]) < 1e-6 * max(1.0, abs(direct))


def test_one_pass_gives_every_prefix():
    sim = Simulator()
    theta, w = injection(sim, 128)
    lik = Likelihood(sim, w)
    full = lik.cumulative(theta, 128)
    a, b = lik.at(theta, 64, 128)
    assert abs(a - full[0, 63]) < 1e-12 and abs(b - full[0, 127]) < 1e-12
    (c,) = Likelihood(sim, w).at(theta, 64)
    assert abs(c - a) < 1e-9


def test_pair_symmetry_of_the_likelihood():
    sim = Simulator()
    theta, w = injection(sim, 192)
    image = theta.clone()
    image[:, 2] = (image[:, 2] + math.pi) % (2 * math.pi)
    image[:, 4] = (image[:, 4] + math.pi / 2) % math.pi
    lik = Likelihood(sim, w)
    assert abs(lik.at(theta, 192)[0] - lik.at(image, 192)[0]) < 1e-9


def test_sampler_matches_the_exact_grid_posterior_of_the_angles():
    sim = Simulator()
    n_obs = 192
    theta, w = injection(sim, n_obs)
    prob, log_z = angle_grid_posterior(sim, w, n_obs, theta, n_grid=120)
    res = angle_posterior(
        sim,
        w,
        n_obs,
        theta,
        n_particles=1500,
        generator=torch.Generator().manual_seed(2),
    )
    x = res.samples
    for axis, span in ((0, 2 * math.pi), (1, math.pi)):
        pmf = prob.sum(1 - axis)
        edges = torch.arange(1, 121, dtype=torch.float64) / 120 * span
        cdf = torch.cumsum(pmf, 0)
        ecdf = (x[:, axis][:, None] <= edges[None, :]).double().mean(0)
        assert (cdf - ecdf).abs().max() < 0.06
    first = ((x[:, 1] - 0.9 + math.pi / 4) % math.pi) < math.pi / 2
    assert abs(first.double().mean().item() - 0.5) < 0.06
    assert abs(res.log_evidence - log_z) < 0.3


def test_five_dimensional_reference_covers_the_injection():
    sim = Simulator()
    n_obs = 256
    theta, w = injection(sim, n_obs, cosi=0.6)
    res = reference_posterior(
        sim,
        w,
        n_obs,
        n_particles=256,
        n_moves=4,
        generator=torch.Generator().manual_seed(4),
    )
    u = sim.to_unit(res.samples)
    assert (u >= 0).all() and (u <= 1).all()
    assert res.samples.shape == (256, 5)
    mc = res.samples[:, 0]
    assert abs(mc.mean() - theta[0, 0]) < 5 * mc.std()
    assert mc.std() < 0.5 * (sim.hi[0] - sim.lo[0]) / math.sqrt(12)
