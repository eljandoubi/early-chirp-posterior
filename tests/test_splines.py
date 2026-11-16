import torch

from chirpvi.splines import (
    circle_forward,
    circle_inverse,
    identity_bias,
    n_params,
    rq_forward,
    rq_inverse,
)

F64 = torch.float64
K = 6


def random_raw(batch, periodic, seed=0, scale=1.0, dtype=F64):
    gen = torch.Generator().manual_seed(seed)
    raw = scale * torch.randn(batch, n_params(K, periodic), generator=gen, dtype=dtype)
    return raw


def test_identity_bias_gives_the_identity_map():
    raw = identity_bias(K, False).to(F64).expand(5, -1)
    x = torch.rand(5, dtype=F64)
    y, logdet = rq_forward(x, raw, K)
    assert (y - x).abs().max() < 1e-6 and logdet.abs().max() < 1e-6
    raw = identity_bias(K, True).to(F64).expand(5, -1)
    y, logdet = circle_forward(x, raw, K)
    assert (y - x).abs().max() < 1e-6 and logdet.abs().max() < 1e-6


def test_forward_and_inverse_round_trip():
    x = torch.rand(400, dtype=F64)
    for periodic in (False, True):
        raw = random_raw(400, periodic, scale=1.5)
        if periodic:
            y, ld = circle_forward(x, raw, K)
            back, ld_back = circle_inverse(y, raw, K)
        else:
            y, ld = rq_forward(x, raw, K)
            back, ld_back = rq_inverse(y, raw, K)
        assert (back - x).abs().max() < 1e-7
        assert (ld + ld_back).abs().max() < 1e-7


def test_float32_round_trip():
    x = torch.rand(400)
    raw = random_raw(400, False, scale=1.0, dtype=torch.float32)
    y, _ = rq_forward(x, raw, K)
    back, _ = rq_inverse(y, raw, K)
    assert (back - x).abs().max() < 2e-4


def test_log_det_matches_a_finite_difference():
    h = 1e-6
    x = 0.02 + 0.96 * torch.rand(300, dtype=F64)
    raw = random_raw(300, False, seed=1)
    _, ld = rq_forward(x, raw, K)
    up, _ = rq_forward(x + h, raw, K)
    dn, _ = rq_forward(x - h, raw, K)
    assert (torch.log((up - dn) / (2 * h)) - ld).abs().max() < 1e-5
    raw = random_raw(300, True, seed=2)
    _, ld = circle_forward(x, raw, K)
    up, _ = circle_forward(x + h, raw, K)
    dn, _ = circle_forward(x - h, raw, K)
    diff = torch.remainder(up - dn + 0.5, 1.0) - 0.5
    assert (torch.log(diff / (2 * h)) - ld).abs().max() < 1e-5


def test_spline_is_monotone_fixes_the_ends_and_integrates_to_one():
    raw = random_raw(1, False, seed=3, scale=2.0).expand(2001, -1)
    x = torch.linspace(0, 1, 2001, dtype=F64)
    y, ld = rq_forward(x, raw, K)
    assert y[0] == 0 and abs(y[-1] - 1) < 1e-12 and (y[1:] >= y[:-1]).all()
    area = torch.trapezoid(torch.exp(ld), x)
    assert abs(area - 1.0) < 1e-3


def test_circle_map_has_equal_slopes_at_the_seam():
    raw = random_raw(1, True, seed=4, scale=2.0).clone()
    raw[..., -2:] = 0.0
    raw = raw.expand(2, -1)
    _, ld = circle_forward(torch.tensor([1e-12, 1 - 1e-12], dtype=F64), raw, K)
    assert abs(ld[0] - ld[1]) < 1e-5


def test_circle_map_is_a_bijection_of_the_circle():
    raw = random_raw(1, True, seed=5, scale=1.5).expand(1000, -1)
    x = torch.linspace(0, 1, 1001, dtype=F64)[:-1]
    y, _ = circle_forward(x, raw, K)
    assert len(torch.unique(y.round(decimals=12))) == 1000
    order = torch.argsort(y)
    steps = torch.remainder(y[order][1:] - y[order][:-1], 1.0)
    assert (steps > 0).all()
