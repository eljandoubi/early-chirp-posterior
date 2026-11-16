import json
from pathlib import Path

import pytest
import torch

from chirpvi.experiments import summarise_reference
from chirpvi.metrics import is_bimodal, mode_summary
from chirpvi.rechecks import mode_counts, paired_excess, width_spread

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def mode(weight, std):
    return dict(weight=weight, sky_std_0=std, sky_std_1=std)


def case(flow_mode, **extra):
    widths = dict(chirp_mass=0.1, coalescence_time=0.1, cos_inclination=0.1)
    row = dict(
        fraction=0.5,
        mode_ref=mode(0.5, 0.05),
        mode_flow=flow_mode,
        c2st=0.6,
        widths_ref=widths,
        widths_flow=widths,
    )
    return dict(row, **extra)


def test_a_uniform_flow_passes_the_weight_check_but_not_the_reference_test():
    gen = torch.Generator().manual_seed(941)
    uniform = torch.rand(2048, 5, generator=gen, dtype=torch.float64)
    summary = mode_summary(uniform)
    assert abs(summary["weight"] - 0.5) < 0.2 and not is_bimodal(summary)
    assert mode_counts([case(summary)]) == dict(
        bimodal_cases=1, weight_near_half=1, concentrated=0, passes_reference_test=0
    )


def test_mode_counts_separate_the_weight_and_the_concentration():
    rows = [
        case(mode(0.5, 0.05)),
        case(mode(0.65, 0.05)),
        case(mode(0.5, 0.45)),
        case(dict(weight=0.0, sky_std_0=0.05, sky_std_1=None)),
        case(mode(0.5, 0.05), mode_ref=mode(0.5, 0.45)),
    ]
    counts = mode_counts(rows)
    assert counts["bimodal_cases"] == 4
    assert counts["weight_near_half"] == 3
    assert counts["concentrated"] == 2
    assert counts["passes_reference_test"] == 1


def test_paired_excess_uses_only_the_cases_that_have_both_values():
    rows = [
        dict(c2st=0.60, c2st_floor=0.55),
        dict(c2st=0.52, c2st_floor=0.58),
        dict(c2st=0.90),
        dict(c2st=0.95, c2st_floor=None),
    ]
    out = paired_excess(rows)
    assert out["cases"] == 2
    assert out["median_excess"] == pytest.approx(-0.005)
    assert out["min_excess"] == pytest.approx(-0.06)
    assert out["max_excess"] == pytest.approx(0.05)
    assert out["median_c2st"] == pytest.approx(0.56)
    assert paired_excess([dict(c2st=0.9)])["median_excess"] is None


def test_width_spread_leaves_a_collapsed_reference_out():
    def row(ref, flow):
        keys = ("chirp_mass", "coalescence_time", "cos_inclination")
        return dict(
            widths_ref={k: ref for k in keys}, widths_flow={k: flow for k in keys}
        )

    out = width_spread([row(0.1, 0.1), row(0.1, 0.2), row(0.1, 0.05), row(0.0, 0.3)])
    mass = out["chirp_mass"]
    assert mass["cases"] == 3 and mass["median"] == pytest.approx(1.0)
    assert mass["min"] == pytest.approx(0.5) and mass["max"] == pytest.approx(2.0)
    assert mass["outside"] == 2


def test_the_summary_of_a_new_run_records_both_mode_criteria_and_the_paired_excess():
    broad = dict(weight=0.5, sky_std_0=0.45, sky_std_1=0.45)
    row = case(broad, c2st_floor=0.55, c2st=0.9, snr=10.0, c2st_symmetry=0.5)
    row.update(is_efficiency=0.1, ref_seconds=1.0, ref_evals=10, ref_distinct=0.9)
    table = summarise_reference([row], (0.5,))["0.5"]
    assert table["flow_keeps_both_modes"] == 1.0
    assert table["mode_counts"]["passes_reference_test"] == 0
    assert table["paired_excess"]["median_excess"] == pytest.approx(0.35)


def test_the_recheck_script_writes_one_entry_per_training_seed(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import recheck

    rows = [case(mode(0.5, 0.05), c2st_floor=0.55), case(mode(0.5, 0.45))]
    for row in rows:
        row.update(ref_stages=7, ref_distinct=0.99)
    for name in ("eval_reference.json", "eval_reference_seed1.json"):
        (tmp_path / name).write_text(json.dumps(dict(records=rows)))
    paths = recheck.recheck(tmp_path)
    assert [Path(p).name for p in paths] == [
        "modes_concentration.json",
        "paired_excess.json",
        "width_spread.json",
        "sampler_health.json",
    ]
    modes = json.loads((tmp_path / "modes_concentration.json").read_text())
    assert set(modes) == {"seed0", "seed1"}
    assert modes["seed1"]["source"] == "eval_reference_seed1.json"
    assert modes["seed0"]["total"]["passes_reference_test"] == 1
    paired = json.loads((tmp_path / "paired_excess.json").read_text())
    assert paired["seed0"]["fractions"]["0.5"]["cases"] == 1
    health = json.loads((tmp_path / "sampler_health.json").read_text())
    assert health["eval_reference.json"] == dict(max_stages=7, min_distinct_share=0.99)
