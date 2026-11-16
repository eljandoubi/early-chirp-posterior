import torch
from torch import nn

from chirpvi.flow import ConditionalFlow, active_sets

N_SAMPLES = 24


def make(dim=5, periodic=(2, 4), scale=0.4, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(layers=4, n_bins=5, hidden=24, context=8, embed_width=16, embed_blocks=1)
    cfg.update(kw)
    flow = ConditionalFlow(N_SAMPLES, dim=dim, periodic=periodic, **cfg).double()
    if scale:
        for layer in flow.layers:
            nn.init.normal_(layer.net[-1].weight, std=scale)
    return flow


def data(batch, seed=1):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, 2, N_SAMPLES, generator=gen, dtype=torch.float64)
    frac = torch.rand(batch, generator=gen, dtype=torch.float64)
    return w, frac


def test_flow_starts_as_the_identity_on_the_cube():
    flow = make(scale=0.0)
    w, frac = data(6)
    u = torch.rand(6, 5, dtype=torch.float64)
    assert flow.log_prob(u, w, frac).abs().max() < 1e-5
    context = flow.embedding(w, frac)
    x = u
    for layer in flow.layers:
        x, _ = layer(x, context)
    assert (x - u).abs().max() < 1e-5


def test_sampling_and_density_agree_and_the_map_inverts():
    flow = make()
    w, frac = data(3)
    u, log_q = flow.sample_many(w, frac, 50)
    back = flow.log_prob(
        u.reshape(-1, 5), w.repeat_interleave(50, 0), frac.repeat_interleave(50)
    )
    assert (back - log_q.reshape(-1)).abs().max() < 1e-7
    context = flow.embedding(w, frac).repeat_interleave(50, 0)
    x = u.reshape(-1, 5)
    for layer in flow.layers:
        x, _ = layer(x, context)
    for layer in reversed(flow.layers):
        x, _ = layer.inverse(x, context)
    assert (x - u.reshape(-1, 5)).abs().max() < 1e-7
    assert ((u >= 0) & (u <= 1)).all()


def test_density_integrates_to_one_with_a_circular_dimension():
    flow = make(dim=2, periodic=(1,), layers=4)
    w, frac = data(1)
    n = 300
    g = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    logq = flow.log_prob(grid, w.expand(n * n, -1, -1), frac.expand(n * n))
    assert abs(torch.exp(logq).mean().item() - 1.0) < 0.02


def test_density_is_continuous_across_the_angle_seam():
    flow = make()
    w, frac = data(40, seed=3)
    u = torch.rand(40, 5, dtype=torch.float64)
    lo, hi = u.clone(), u.clone()
    lo[:, 2], hi[:, 2] = 1e-7, 1 - 1e-7
    lo[:, 4], hi[:, 4] = 1e-7, 1 - 1e-7
    gap = flow.log_prob(lo, w, frac) - flow.log_prob(hi, w, frac)
    assert gap.abs().max() < 1e-3


def test_context_changes_the_density():
    flow = make()
    w, frac = data(2, seed=5)
    u = torch.rand(1, 5, dtype=torch.float64).repeat(2, 1)
    a, b = flow.log_prob(u, w, frac)
    assert abs(a - b) > 1e-3


def test_active_sets_cover_every_dimension():
    sets = active_sets(5, 8)
    assert sets[:5] == [(0, 1), (2, 3), (4, 0), (1, 2), (3, 4)]
    assert {i for s in sets for i in s} == set(range(5))
    assert active_sets(2, 2) == [(0,), (1,)]


def test_single_observation_sampler_returns_the_right_shapes():
    flow = make()
    w, frac = data(1)
    u, log_q = flow.sample(w[0], frac[0], 17)
    assert u.shape == (17, 5) and log_q.shape == (17,)
