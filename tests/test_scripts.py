import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_train_script_takes_a_peak_learning_rate(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import train

    assert train.parse(["--smoke"]).lr == 3e-4
    assert train.parse(["--smoke", "--lr", "1e-4"]).lr == 1e-4


def test_report_prints_one_cost_row_per_device(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report

    row = dict(
        n_obs=192,
        n_samples=2048,
        flow_seconds_one=0.01,
        flow_seconds_amortised=0.001,
        reference_seconds=3.0,
        reference_evals=400000,
        load_average=0.5,
    )
    for name, device in (("timing_t4", "Tesla T4"), ("timing_cpu", "cpu")):
        (tmp_path / f"{name}.json").write_text(json.dumps(dict(row, device=device)))
    report.cost(tmp_path)
    out = capsys.readouterr().out
    assert "Tesla T4" in out and "| cpu |" in out and "2048 posterior samples" in out


def test_figures_are_drawn_from_the_result_files(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import figures

    hist = dict(counts=[5] * 20, expected=[5.0] * 20)
    sbc = dict(chirp_mass=dict(ks=0.1, p_value=0.5, hist=hist))
    flow = {"0.5": dict(sbc=sbc), "1": dict(sbc=sbc)}
    widths = dict(chirp_mass=0.2, coalescence_time=0.4)
    summary = {f: dict(width_reference=widths, width_flow=widths) for f in ("0.5", "1")}
    assert Path(figures.sbc_figure(flow, tmp_path / "sbc.png")).stat().st_size > 1000
    path = figures.width_figure(dict(summary=summary), tmp_path / "width.png")
    assert Path(path).stat().st_size > 1000
