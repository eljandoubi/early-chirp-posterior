import json
import math

import torch

from chirpvi.util import (
    circular_mean,
    circular_std,
    unit_distance,
    wrap_unit,
    write_json,
)


def test_wrap_and_circular_statistics():
    x = torch.tensor([0.95, 0.05, 0.99, 0.01], dtype=torch.float64)
    assert (
        abs(circular_mean(x).item() % 1.0) < 1e-9
        or abs(circular_mean(x).item() - 1.0) < 1e-9
    )
    assert (
        wrap_unit(torch.tensor([-0.25, 1.25])) - torch.tensor([0.75, 0.25])
    ).abs().max() < 1e-6
    tight = torch.tensor([0.99, 0.0, 0.01], dtype=torch.float64)
    spread = torch.tensor([0.0, 0.33, 0.66], dtype=torch.float64)
    assert circular_std(tight).item() < 0.1 < circular_std(spread).item()
    assert math.isfinite(
        circular_std(torch.tensor([0.1, 0.1], dtype=torch.float64)).item()
    )


def test_unit_distance_is_circular_only_on_periodic_dimensions():
    a = torch.tensor([[0.05, 0.05]], dtype=torch.float64)
    b = torch.tensor([[0.95, 0.95]], dtype=torch.float64)
    d = unit_distance(a, b, torch.tensor([True, False]))
    assert abs(d[0, 0].item() - 0.1) < 1e-12 and abs(d[0, 1].item() - 0.9) < 1e-12


def test_write_json_cleans_values(tmp_path):
    data = {1: float("nan"), "a": (torch.tensor(2.0), [float("inf"), 3]), "b": {0.5: 1}}
    path = write_json(tmp_path / "out" / "x.json", data)
    back = json.loads(open(path).read())
    assert back == {"1": None, "a": [2.0, [None, 3]], "b": {"0.5": 1}}
