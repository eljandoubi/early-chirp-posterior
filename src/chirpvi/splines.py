"""Monotone rational-quadratic splines on [0, 1] (Durkan et al., 2019)."""

import torch
import torch.nn.functional as F

MIN_BIN = 1e-3
MIN_DERIV = 1e-3


def n_params(n_bins: int, periodic: bool) -> int:
    """Raw parameters per transformed dimension (a circle also has two shifts)."""
    return 3 * n_bins + 2 if periodic else 3 * n_bins + 1


def identity_bias(n_bins: int, periodic: bool) -> torch.Tensor:
    """Raw values that make the spline the identity map."""
    out = torch.zeros(n_params(n_bins, periodic))
    d0 = torch.log(torch.expm1(torch.tensor(1.0 - MIN_DERIV)))
    out[2 * n_bins : 3 * n_bins + (0 if periodic else 1)] = d0
    return out


def _knots(raw, n_bins, periodic):
    raw_w, raw_h = raw[..., :n_bins], raw[..., n_bins : 2 * n_bins]
    raw_d = raw[..., 2 * n_bins : 3 * n_bins + (0 if periodic else 1)]
    widths = MIN_BIN + (1.0 - MIN_BIN * n_bins) * F.softmax(raw_w, -1)
    heights = MIN_BIN + (1.0 - MIN_BIN * n_bins) * F.softmax(raw_h, -1)
    deriv = MIN_DERIV + F.softplus(raw_d)
    if periodic:
        deriv = torch.cat([deriv, deriv[..., :1]], -1)
    zero, one = torch.zeros_like(widths[..., :1]), torch.ones_like(widths[..., :1])
    cw = torch.cat([zero, torch.cumsum(widths, -1)[..., :-1], one], -1)
    ch = torch.cat([zero, torch.cumsum(heights, -1)[..., :-1], one], -1)
    return widths, heights, deriv, cw, ch


def _pick(table, idx):
    return table.gather(-1, idx[..., None]).squeeze(-1)


def _bin(values, edges):
    inner = edges[..., 1:-1].contiguous()
    return torch.searchsorted(inner, values[..., None].contiguous()).squeeze(-1)


def _slope(wk, hk, dk, dk1, xi):
    s = hk / wk
    xi1 = xi * (1.0 - xi)
    denom = s + (dk1 + dk - 2.0 * s) * xi1
    num = s**2 * (dk1 * xi**2 + 2.0 * s * xi1 + dk * (1.0 - xi) ** 2)
    return s, denom, num


def rq_forward(x, raw, n_bins, periodic=False):
    """Map x in [0, 1] and return (y, log dy/dx). raw has n_params entries."""
    w, h, d, cw, ch = _knots(raw, n_bins, periodic)
    x = x.clamp(0.0, 1.0)
    idx = _bin(x, cw)
    wk, hk, cwk, chk = _pick(w, idx), _pick(h, idx), _pick(cw, idx), _pick(ch, idx)
    dk, dk1 = _pick(d, idx), _pick(d[..., 1:], idx)
    xi = (x - cwk) / wk
    s, denom, num = _slope(wk, hk, dk, dk1, xi)
    y = chk + hk * (s * xi**2 + dk * xi * (1.0 - xi)) / denom
    return y, torch.log(num) - 2.0 * torch.log(denom)


def rq_inverse(y, raw, n_bins, periodic=False):
    """Inverse of rq_forward; returns (x, log dx/dy)."""
    w, h, d, cw, ch = _knots(raw, n_bins, periodic)
    y = y.clamp(0.0, 1.0)
    idx = _bin(y, ch)
    wk, hk, cwk, chk = _pick(w, idx), _pick(h, idx), _pick(cw, idx), _pick(ch, idx)
    dk, dk1 = _pick(d, idx), _pick(d[..., 1:], idx)
    s = hk / wk
    dy = y - chk
    shape = dk1 + dk - 2.0 * s
    a = dy * shape + hk * (s - dk)
    b = hk * dk - dy * shape
    c = -s * dy
    disc = (b**2 - 4.0 * a * c).clamp_min(0.0)
    xi = 2.0 * c / (-b - torch.sqrt(disc))
    x = xi * wk + cwk
    _, denom, num = _slope(wk, hk, dk, dk1, xi)
    return x, -(torch.log(num) - 2.0 * torch.log(denom))


def circle_forward(x, raw, n_bins):
    """Circle diffeomorphism: shift, periodic spline, shift. raw has n_params(True)."""
    shift_in, shift_out = raw[..., -2], raw[..., -1]
    x1 = torch.remainder(x + shift_in, 1.0)
    y1, logdet = rq_forward(x1, raw[..., :-2], n_bins, periodic=True)
    return torch.remainder(y1 + shift_out, 1.0), logdet


def circle_inverse(y, raw, n_bins):
    shift_in, shift_out = raw[..., -2], raw[..., -1]
    y1 = torch.remainder(y - shift_out, 1.0)
    x1, logdet = rq_inverse(y1, raw[..., :-2], n_bins, periodic=True)
    return torch.remainder(x1 - shift_in, 1.0), logdet
