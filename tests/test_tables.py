import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_tables_file_is_the_report_output(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr(sys, "argv", ["report.py"])
    import report

    report.main()
    assert capsys.readouterr().out == (ROOT / "results" / "tables.txt").read_text()


def test_every_readme_table_row_is_a_row_of_the_report_output():
    tables = set((ROOT / "results" / "tables.txt").read_text().splitlines())
    lines = (ROOT / "README.md").read_text().splitlines()
    rows = [line for line in lines if line.startswith("|")]
    assert rows and [row for row in rows if row not in tables] == []


def test_report_prints_the_derived_tables_and_skips_missing_files(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import report

    for fn in (report.modes_table, report.paired_table, report.spread_table):
        fn(tmp_path)
    assert capsys.readouterr().out == ""
    modes = dict(
        bimodal_cases=4, concentrated=3, passes_reference_test=3, weight_near_half=4
    )
    pair = dict(
        cases=5,
        median_c2st=0.6,
        median_floor=0.55,
        median_excess=0.05,
        min_excess=-0.02,
        max_excess=0.1,
    )
    cell = dict(cases=20, median=1.05, min=0.9, max=1.4, outside=7)
    files = dict(
        modes_concentration=dict(seed1=dict(fractions={"0.5": modes}, total=modes)),
        paired_excess=dict(seed0=dict(fractions={"0.5": pair})),
        width_spread=dict(
            seed0=dict(tolerance=0.11, fractions={"0.5": dict(a=cell, b=cell, c=cell)})
        ),
    )
    for name, data in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    for fn in (report.modes_table, report.paired_table, report.spread_table):
        fn(tmp_path)
    out = capsys.readouterr().out
    assert "for two modes, seed1" in out
    assert "| 0.5 | 4 | 4 | 3 |" in out and "| all | 4 | 4 | 3 |" in out
    assert "| 0.5 | 5 | 0.600 | 0.550 | 0.050 | -0.020 to 0.100 |" in out
    assert "1.05 (0.90 to 1.40), 7 of 20" in out and "more than 11%" in out


def test_report_prints_no_side_by_side_table_for_one_training_run(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import report

    log = dict(summary=dict(best_val_nll=-5.0))
    (tmp_path / "train_log_seed0.json").write_text(json.dumps(log))
    reference = dict(summary={"0.5": dict(median_c2st=0.6)})
    (tmp_path / "eval_reference.json").write_text(json.dumps(reference))
    report.seed_summary(tmp_path)
    assert capsys.readouterr().out == ""
