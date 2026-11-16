from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_train_script_takes_a_peak_learning_rate(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import train

    assert train.parse(["--smoke"]).lr == 3e-4
    assert train.parse(["--smoke", "--lr", "1e-4"]).lr == 1e-4


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
