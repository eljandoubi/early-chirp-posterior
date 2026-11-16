import pytest
import torch

from chirpvi.experiments import (
    calibration,
    compare_with_reference,
    sequential_experiment,
    summarise_reference,
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


def record(ref_width, distinct):
    mode = dict(weight=0.5, sky_std_0=0.1, sky_std_1=0.1)
    return dict(
        fraction=1.0,
        snr=10.0,
        c2st=0.9,
        c2st_symmetry=0.5,
        is_efficiency=0.1,
        mode_ref=mode,
        mode_flow=mode,
        widths_ref=dict(
            chirp_mass=ref_width, coalescence_time=0.1, cos_inclination=0.1
        ),
        widths_flow=dict(chirp_mass=0.2, coalescence_time=0.2, cos_inclination=0.2),
        ref_seconds=1.0,
        ref_evals=10,
        ref_distinct=distinct,
    )


def test_summary_leaves_a_collapsed_reference_out_of_the_width_ratio():
    table = summarise_reference([record(0.1, 0.5), record(0.0, 0.02)], (1.0,))["1"]
    assert table["collapsed_reference"] == 1
    assert table["width_ratio"]["chirp_mass"] == 2.0
    assert table["median_reference_distinct"] == pytest.approx(0.26)


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
    assert all(sum(v["hist"]["counts"]) in (12, 4) for v in row["sbc"].values())
    assert len(row["central"]["0.9"]) == 3
    assert {"max_deviation", "mean_deviation"} <= set(row["tarp"])


def test_reference_comparison_records_and_summary():
    sim = Simulator()
    lines = []
    out = compare_with_reference(
        sim,
        tiny_posterior(sim),
        (0.5,),
        n_cases=2,
        n_particles=64,
        seed=2,
        floor_cases=1,
        n_moves=2,
        log=lines.append,
    )
    assert len(out["records"]) == 2
    assert [(x["fraction"], x["case"]) for x in lines] == [(0.5, 0), (0.5, 1)]
    rec = out["records"][0]
    assert 0 <= rec["c2st"] <= 1 and 0 <= rec["is_efficiency"] <= 1
    assert 0 < rec["ref_distinct"] <= 1
    assert "c2st_floor" in rec and "c2st_floor" not in out["records"][1]
    summary = out["summary"]["0.5"]
    assert summary["cases"] == 2 and set(summary["width_ratio"]) == {
        "chirp_mass",
        "coalescence_time",
        "cos_inclination",
    }


def test_the_floor_c2st_repeats_for_one_seed():
    sim = Simulator()
    kwargs = dict(
        fractions=(0.5,),
        n_cases=1,
        n_particles=64,
        seed=4,
        floor_cases=1,
        n_moves=2,
    )
    first = compare_with_reference(sim, tiny_posterior(sim), **kwargs)
    second = compare_with_reference(sim, tiny_posterior(sim), **kwargs)
    assert first["records"][0]["c2st_floor"] == second["records"][0]["c2st_floor"]


def test_sequential_experiment_compares_three_ways():
    sim = Simulator()
    lines = []
    out = sequential_experiment(
        sim,
        tiny_posterior(sim),
        (96, 160),
        n_cases=1,
        n_particles=64,
        seed=3,
        n_moves=2,
        log=lines.append,
    )
    assert [x["case"] for x in lines] == [0]
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
