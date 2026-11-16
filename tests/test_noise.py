import math

import torch

from chirpvi.config import Problem
from chirpvi.noise import Noise, psd


def test_whitener_is_lower_triangular_and_exact():
    noise = Noise(Problem())
    assert torch.triu(noise.whiten, 1).abs().max() == 0
    eye = noise.whiten @ noise.cov @ noise.whiten.T
    assert (eye - torch.eye(eye.shape[0], dtype=eye.dtype)).abs().max() < 1e-8


def test_truncation_commutes_with_whitening():
    noise = Noise(Problem())
    gen = torch.Generator().manual_seed(0)
    n = 100
    x = torch.randn(256, generator=gen, dtype=torch.float64)
    x[n:] = 0.0
    small = torch.linalg.cholesky(noise.cov[:n, :n])
    direct = torch.linalg.solve_triangular(small, x[:n, None], upper=False)[:, 0]
    assert (noise.whiten @ x)[:n].sub(direct).abs().max() < 1e-9


def test_coloured_noise_has_the_requested_spectrum():
    problem = Problem()
    noise = Noise(problem)
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(1500, problem.n, generator=gen, dtype=torch.float64)
    x = noise.colour(z) * torch.hann_window(problem.n, dtype=torch.float64)
    power = torch.fft.rfft(x).abs() ** 2
    scale = 2.0 / (problem.fs * (torch.hann_window(problem.n) ** 2).sum().item())
    est = power.mean(0) * scale
    freq = torch.fft.rfftfreq(problem.n, 1 / problem.fs)
    for hz in (25.0, 35.0, 45.0):
        k = int(round(hz * problem.n / problem.fs))
        ratio = est[k].item() / psd(freq[k], problem).item()
        assert abs(ratio - 1.0) < 0.15, (hz, ratio)


def test_the_noise_is_much_redder_below_the_band():
    problem = Problem()
    freq = torch.tensor([13.0, 35.0], dtype=torch.float64)
    s13, s35 = psd(freq, problem)
    assert 5 < (s13 / s35).item() < 12
    assert math.isfinite(psd(torch.tensor(0.0), problem).item())
