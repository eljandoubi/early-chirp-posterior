import torch

from chirpvi.flow import ConditionalFlow
from chirpvi.simulator import Simulator
from chirpvi.training import (
    FlowPosterior,
    load_flow,
    mean_nll,
    save_flow,
    train_flow,
    training_batch,
)

SMALL = dict(layers=2, n_bins=4, hidden=32, context=16, embed_width=32, embed_blocks=1)


def small_flow(sim):
    torch.manual_seed(0)
    return ConditionalFlow(sim.problem.n, **SMALL)


def test_training_batch_shapes_and_range():
    sim = Simulator()
    u, w, frac = training_batch(sim, 16, 32, torch.Generator().manual_seed(0))
    assert u.shape == (16, 5) and w.shape == (16, 2, 256) and frac.shape == (16,)
    assert u.dtype == torch.float32 and ((u >= 0) & (u <= 1)).all()
    assert (frac >= 32 / 256).all() and (frac <= 1).all()
    n_obs = (frac * 256).round().long()
    for i in range(16):
        assert w[i, :, n_obs[i] :].abs().max() == 0


def test_training_lowers_the_validation_loss_and_keeps_the_best_weights():
    sim = Simulator()
    flow = small_flow(sim)
    out = train_flow(sim, flow, steps=60, batch=64, val_size=256, val_every=20, seed=1)
    nll = [h["val_nll"] for h in out["history"]]
    assert len(nll) == 3 and nll[-1] < nll[0] <= 0.05
    assert out["summary"]["best_val_nll"] == min(nll)
    val = training_batch(sim, 256, 32, torch.Generator().manual_seed(10_001))
    assert abs(mean_nll(flow, val) - out["summary"]["best_val_nll"]) < 1e-4


def test_minutes_budget_sets_the_number_of_steps():
    sim = Simulator()
    flow = small_flow(sim)
    out = train_flow(
        sim, flow, steps=10, batch=16, val_size=64, val_every=1000, minutes=0.02, seed=2
    )
    assert out["summary"]["steps"] >= 31
    assert out["summary"]["seconds"] < 0.02 * 60 * 3 + 30


def test_checkpoint_round_trip_and_posterior_wrapper(tmp_path):
    sim = Simulator()
    flow = small_flow(sim)
    train_flow(sim, flow, steps=5, batch=16, val_size=32, val_every=5)
    path = tmp_path / "flow.pt"
    save_flow(path, flow, {"note": "test"})
    again = load_flow(path)
    post_a, post_b = FlowPosterior(flow, sim), FlowPosterior(again, sim)
    theta = sim.sample_prior(2, torch.Generator().manual_seed(3))
    w = sim.observe(theta, torch.tensor([100, 256]), torch.Generator().manual_seed(4))
    a, _ = post_a.sample(w[0], 100, 20, torch.Generator().manual_seed(5))
    b, _ = post_b.sample(w[0], 100, 20, torch.Generator().manual_seed(5))
    assert torch.allclose(a, b, atol=1e-5)
    assert ((a >= sim.lo) & (a <= sim.hi)).all()
    logq = post_a.log_prob(theta, w, torch.tensor([100, 256]))
    assert logq.shape == (2,) and torch.isfinite(logq).all()
