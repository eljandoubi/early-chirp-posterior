import torch

from chirpvi.reference import reference_posterior
from chirpvi.sequential import sequential_path, sequential_step
from chirpvi.simulator import Simulator


def stream(sim, seed=0):
    gen = torch.Generator().manual_seed(seed)
    theta = sim.sample_prior(1, gen)
    theta[0, 0], theta[0, 1], theta[0, 3] = 22.0, 2.35, 0.5
    w = sim.observe(theta, torch.tensor([256]), gen)[0]
    return theta, w


def test_a_sequential_step_matches_the_posterior_computed_from_scratch():
    sim = Simulator()
    theta, w = stream(sim)
    gen = torch.Generator().manual_seed(1)
    kw = dict(n_particles=1024, n_moves=15, ess_frac=0.7, generator=gen)
    early = reference_posterior(sim, w, 96, **kw)
    late = reference_posterior(sim, w, 160, **kw)
    step = sequential_step(
        sim, w, 96, 160, early.samples, n_moves=15, ess_frac=0.7, generator=gen
    )
    gain = late.log_evidence - early.log_evidence
    assert abs(step.log_evidence - gain) < 1.5
    for i in (0, 1):
        a, b = step.samples[:, i], late.samples[:, i]
        spread = max(b.std().item(), 1e-6)
        assert abs(a.mean() - b.mean()) < 1.0 * spread + 0.05 * (sim.hi[i] - sim.lo[i])
    assert step.samples.shape == (1024, 5)


def test_a_path_gives_one_record_per_step_and_uses_fewer_evaluations_late():
    sim = Simulator()
    theta, w = stream(sim, seed=2)
    gen = torch.Generator().manual_seed(3)
    first = reference_posterior(sim, w, 96, n_particles=256, n_moves=4, generator=gen)
    scratch = reference_posterior(
        sim, w, 256, n_particles=256, n_moves=4, generator=gen
    )
    records = sequential_path(
        sim, w, [96, 160, 256], first.samples, n_moves=4, generator=gen
    )
    assert [r["n_obs"] for r in records] == [160, 256]
    assert all(r["result"].samples.shape == (256, 5) for r in records)
    assert records[-1]["result"].n_evals < scratch.n_evals
