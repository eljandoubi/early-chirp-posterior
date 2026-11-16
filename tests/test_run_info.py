import json
import sys
from pathlib import Path

import torch

from chirpvi.experiments import grid_check
from chirpvi.simulator import Simulator
from chirpvi.util import environment

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
ENVIRONMENT_KEYS = {"python", "torch", "cpu_count", "device"}


def test_environment_names_the_versions_the_cpu_count_and_the_device():
    env = environment("cpu")
    assert set(env) == ENVIRONMENT_KEYS
    assert env["torch"] == torch.__version__ and env["device"] == "cpu"
    assert env["python"].count(".") == 2 and env["cpu_count"] >= 1


def test_a_script_stores_its_environment_and_settings(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import calibrate_snr

    monkeypatch.setattr(calibrate_snr, "results_dir", lambda args: tmp_path)
    argv = ["calibrate_snr.py", "--smoke", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    calibrate_snr.main()
    run = json.loads((tmp_path / "snr.json").read_text())["run"]
    assert set(run) == {"environment", "settings"}
    assert set(run["environment"]) == ENVIRONMENT_KEYS
    assert run["settings"] == dict(n=400, seed=0)


def test_the_check_reference_presets_name_the_grid_settings(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import check_reference

    for preset in check_reference.PRESETS.values():
        assert {"grid_particles", "grid_moves", "grid_ess"} <= set(preset)
    full = check_reference.PRESETS["full"]
    assert (full["grid_particles"], full["grid_moves"], full["grid_ess"]) == (
        2000,
        6,
        0.5,
    )


def test_the_grid_check_returns_the_settings_it_used():
    out = grid_check(
        Simulator(), n_obs=96, n_grid=20, n_particles=200, n_moves=3, ess_frac=0.6
    )
    expected = dict(n_obs=96, n_grid=20, particles=200, moves=3, ess_fraction=0.6)
    assert out["settings"] == expected and out["n_evals"] > 200


def test_report_skips_the_run_entry_of_a_calibration_file(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report

    sbc = dict(chirp_mass=dict(ks=0.1, p_value=0.5))
    row = dict(
        sbc=sbc,
        hpd={"0.5": 0.5, "0.68": 0.68, "0.9": 0.9, "0.95": 0.95},
        tarp=dict(max_deviation=0.1, mean_deviation=0.0),
    )
    data = {"0.5": row, "run": dict(environment={}, settings={})}
    (tmp_path / "eval_flow.json").write_text(json.dumps(data))
    report.calibration_tables(tmp_path)
    out = capsys.readouterr().out
    assert "| 0.5 | 0.100 (0.50) |" in out and "| run |" not in out
