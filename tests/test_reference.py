import math

import torch

from chirpvi.experiments import grid_check
from chirpvi.reference import Likelihood, reference_posterior
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
    out = grid_check(Simulator(), seed=0)
    assert out["settings"] == dict(
        n_obs=192, n_grid=120, particles=1500, moves=6, ess_fraction=0.5
    )
    assert out["ks_phase"] < 0.06 and out["ks_sky"] < 0.06
    assert abs(out["mode_weight"] - 0.5) < 0.06
    assert abs(out["sampler_log_evidence"] - out["grid_log_evidence"]) < 0.3
    assert out["symmetry_error"] < 1e-9


def test_five_dimensional_reference_finds_the_injection_and_repeats():
    sim = Simulator()
    n_obs = 256
    theta, w = injection(sim, n_obs, cosi=0.6)
    runs = [
        reference_posterior(
            sim, w, n_obs, 1024, 15, 0.7, torch.Generator().manual_seed(seed)
        )
        for seed in (4, 5)
    ]
    prior_std = (sim.hi[0] - sim.lo[0]) / math.sqrt(12)
    for res in runs:
        u = sim.to_unit(res.samples)
        assert (u >= 0).all() and (u <= 1).all() and res.samples.shape == (1024, 5)
        mc = res.samples[:, 0]
        assert abs(mc.mean() - theta[0, 0]) < 5 * mc.std()
        assert mc.std() < 0.2 * prior_std
    a, b = runs[0].samples[:, 0], runs[1].samples[:, 0]
    # about twice the largest gap over ten seed pairs: 0.26 std, ratio 0.88 to 1.12
    assert abs(a.mean() - b.mean()) < 0.5 * a.std()
    assert 0.75 < (b.std() / a.std()).item() < 1.33
