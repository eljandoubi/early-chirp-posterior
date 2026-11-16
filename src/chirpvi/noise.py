"""Coloured Gaussian noise with a known spectrum and its exact causal whitener."""

import math

import torch

from .config import Problem


def psd(freq: torch.Tensor, problem: Problem) -> torch.Tensor:
    """One-sided PSD: a low-frequency wall, a flat bucket, a rising edge."""
    f = freq.clamp_min(problem.floor_hz)
    return 1.0 + (problem.wall_hz / f) ** 4 + (f / problem.high_hz) ** 2


def autocovariance(problem: Problem, n_freq: int = 8192) -> torch.Tensor:
    """Autocovariance r[k] = integral of S(f) cos(2 pi f k / fs) over [0, fs / 2]."""
    df = problem.fs / 2 / n_freq
    freq = (torch.arange(n_freq, dtype=torch.float64) + 0.5) * df
    lags = torch.arange(problem.n, dtype=torch.float64)
    cos = torch.cos(2 * math.pi * lags[:, None] * freq[None, :] / problem.fs)
    return (cos * psd(freq, problem)[None, :]).sum(1) * df


class Noise:
    """Toeplitz covariance K = L L^T, noise = L z, whitener W = L^-1 (lower)."""

    def __init__(self, problem: Problem, device: str | torch.device = "cpu"):
        n = problem.n
        r = autocovariance(problem)
        lag = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs()
        self.cov = r[lag]
        chol = torch.linalg.cholesky(self.cov)
        eye = torch.eye(n, dtype=torch.float64)
        whiten = torch.linalg.solve_triangular(chol, eye, upper=False)
        self.cov, self.chol, self.whiten = (
            m.to(device) for m in (self.cov, chol, whiten)
        )

    def colour(self, z: torch.Tensor) -> torch.Tensor:
        """Map standard normal rows (..., n) to noise with covariance K."""
        return z @ self.chol.T
