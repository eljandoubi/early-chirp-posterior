import torch

from chirpvi.experiments import (
    calibration,
    compare_with_reference,
    sequential_experiment,
    timing,
)
from chirpvi.flow import ConditionalFlow
from chirpvi.simulator import Simulator
from chirpvi.training import FlowPosterior

SMALL = dict(layers=2, n_bins=4, hidden=16, context=8, embed_width=16, embed_blocks=1)


def tiny_posterior(sim):
    torch.manual_seed(0)
    flow = ConditionalFlow(sim.problem.n, **SMALL)
    return FlowPosterior(flow, sim)


def test_calibration_returns_every_statistic():
    sim = Simulator()
    out = calibration(
        sim,
        tiny_posterior(sim),
        (0.5,),
        n_test=12,
        n_samples=20,
        seed=1,
        chunk=6,
        loglik_cases=4,
    )
    row = out["0.5"]
    assert row["n_obs"] == 128 and set(row["hpd"]) == {"0.5", "0.68", "0.9", "0.95"}
    assert len(row["sbc"]) == 8 and "log_likelihood" in row["sbc"]
    assert all(0 <= v["ks"] <= 1 for v in row["sbc"].values())
    assert len(row["central"]["0.9"]) == 3
    assert {"max_deviation", "mean_deviation"} <= set(row["tarp"])


def test_reference_comparison_records_and_summary():
    sim = Simulator()
    out = compare_with_reference(
        sim,
        tiny_posterior(sim),
        (0.5,),
        n_cases=2,
        n_particles=64,
        seed=2,
        floor_cases=1,
        n_moves=2,
    )
    assert len(out["records"]) == 2
    rec = out["records"][0]
    assert 0 <= rec["c2st"] <= 1 and 0 <= rec["is_efficiency"] <= 1
    assert "c2st_floor" in rec and "c2st_floor" not in out["records"][1]
    summary = out["summary"]["0.5"]
    assert summary["cases"] == 2 and set(summary["width_ratio"]) == {
        "chirp_mass",
        "coalescence_time",
        "cos_inclination",
    }


def test_sequential_experiment_compares_three_ways():
    sim = Simulator()
    out = sequential_experiment(
        sim,
        tiny_posterior(sim),
        (96, 160),
        n_cases=1,
        n_particles=64,
        seed=3,
        n_moves=2,
    )
    assert set(out["summary"]) == {
        f"{v}@160" for v in ("scratch", "chain_sampler", "chain_flow")
    }
    assert out["summary"]["chain_sampler@160"]["evals"] > 0
    assert out["c2st_floor"] == out["c2st_floor"]


def test_timing_has_the_cost_fields():
    sim = Simulator()
    out = timing(sim, tiny_posterior(sim), 128, n_samples=30, n_particles=64, repeats=1)
    assert out["flow_seconds_one"] > 0 and out["reference_seconds"] > 0
    assert out["device"] == "cpu" and out["reference_evals"] > 0
