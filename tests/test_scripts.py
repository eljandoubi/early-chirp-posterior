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


def test_report_orders_the_sequential_rows_by_variant_and_length(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report

    cell = dict(evals=1, seconds=1.0, stages=1, ess_first=1.0, c2st=0.5)
    keys = ["chain_flow@96", "scratch@128", "chain_sampler@96", "scratch@96"]
    summary = {k: cell for k in sorted(keys)}
    (tmp_path / "sequential.json").write_text(
        json.dumps(dict(summary=summary, c2st_floor=0.5))
    )
    report.sequential_table(tmp_path)
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if x.startswith(("| scratch", "| chain"))]
    found = [line.split("|")[1].strip() for line in lines]
    assert found == ["scratch@96", "scratch@128", "chain_sampler@96", "chain_flow@96"]
    assert "| proposal correction ESS |" in out


def test_report_titles_the_calibration_tables_of_another_seed(
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
    (tmp_path / "eval_flow_seed1.json").write_text(json.dumps({"0.5": row}))
    report.calibration_tables(tmp_path, "eval_flow_seed1.json")
    out = capsys.readouterr().out
    assert "(p value), seed1" in out and "TARP" in out and "HPD" in out


def test_report_titles_the_reference_tables_of_another_seed(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report

    widths = dict(chirp_mass=0.1, coalescence_time=0.2, cos_inclination=0.3)
    row = dict(
        median_snr=5.0,
        median_c2st=0.6,
        median_c2st_floor=0.55,
        median_is_efficiency=0.1,
        bimodal_cases=3,
        flow_keeps_both_modes=1.0,
        flow_mode_gap=0.01,
        reference_mode_gap=0.01,
        median_c2st_symmetry=0.5,
        width_reference=widths,
        width_ratio=widths,
    )
    (tmp_path / "eval_reference_seed1.json").write_text(
        json.dumps(dict(summary={"0.5": row}))
    )
    report.reference_tables(tmp_path, "eval_reference_seed1.json")
    out = capsys.readouterr().out
    assert "Flow against the reference, seed1" in out and "Mode recovery, seed1" in out


def test_report_puts_the_training_runs_side_by_side(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report

    def flow(p):
        joint = dict(ks=0.9, p_value=0.0, hist=dict(counts=[0] * 19 + [10]))
        sbc = dict(chirp_mass=dict(ks=0.02, p_value=p), log_likelihood=joint)
        hpd = {"0.5": 0.52, "0.9": 0.89}
        tarp = dict(max_deviation=0.03)
        return {f: dict(sbc=sbc, hpd=hpd, tarp=tarp) for f in ("0.5", "1")}

    ratio = dict(mass=2.0, time=2.5, cos=1.4)
    reference = dict(
        summary={f: dict(median_c2st=0.6, width_ratio=ratio) for f in ("0.75", "1")}
    )
    for seed in (0, 1):
        log = dict(summary=dict(best_val_nll=-5.0 - seed))
        (tmp_path / f"train_log_seed{seed}.json").write_text(json.dumps(log))
    for name, data in (
        ("eval_flow.json", flow(0.5)),
        ("eval_flow_seed1.json", flow(0.01)),
        ("eval_reference.json", reference),
        ("eval_reference_seed1.json", reference),
    ):
        (tmp_path / name).write_text(json.dumps(data))
    report.seed_summary(tmp_path)
    rows = [x for x in capsys.readouterr().out.splitlines() if x.startswith("| ")]
    assert rows[1].startswith(
        "| 0 | -5.000 | 0 of 2 | 0.020 | 0.900 to 0.900 | 10 of 10 |"
    )
    assert rows[2].startswith("| 1 | -6.000 | 2 of 2 |") and rows[2].endswith(
        "2.00 / 2.50 / 1.40 |"
    )


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
