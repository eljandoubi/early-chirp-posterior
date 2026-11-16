from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_train_script_takes_a_peak_learning_rate(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import train

    assert train.parse(["--smoke"]).lr == 3e-4
    assert train.parse(["--smoke", "--lr", "1e-4"]).lr == 1e-4
