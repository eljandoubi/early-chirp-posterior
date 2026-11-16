"""Conditional coupling flow on the unit cube with rational-quadratic splines."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .splines import (
    circle_forward,
    circle_inverse,
    identity_bias,
    n_params,
    rq_forward,
    rq_inverse,
)


def active_sets(dim: int, layers: int) -> list[tuple[int, ...]]:
    """Dimensions transformed by each layer: rotating windows over the dimensions."""
    k = max(1, dim // 2)
    return [tuple((n * k + j) % dim for j in range(k)) for n in range(layers)]


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(x)))


class Embedding(nn.Module):
    """Residual MLP of the whitened strain (zero after n_obs) and the fraction f."""

    def __init__(self, n_samples: int, width: int, out: int, blocks: int):
        super().__init__()
        self.inp = nn.Linear(2 * n_samples + 1, width)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.out = nn.Linear(width, out)

    def forward(self, w, frac):
        x = torch.cat([w.flatten(1), frac[:, None]], 1)
        return self.out(F.gelu(self.blocks(F.gelu(self.inp(x)))))


class Coupling(nn.Module):
    """Spline transform of the active dimensions given the others and the context."""

    def __init__(self, dim, active, periodic, context, hidden, n_bins):
        super().__init__()
        self.active = list(active)
        self.passive = [i for i in range(dim) if i not in self.active]
        self.periodic = [i in periodic for i in range(dim)]
        self.n_bins = n_bins
        self.sizes = [n_params(n_bins, self.periodic[i]) for i in self.active]
        n_in = sum(2 if self.periodic[i] else 1 for i in self.passive) + context
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, sum(self.sizes)),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            bias = [identity_bias(n_bins, self.periodic[i]) for i in self.active]
            self.net[-1].bias.copy_(torch.cat(bias))

    def raw(self, x, context):
        feats = []
        for i in self.passive:
            if self.periodic[i]:
                angle = 2 * math.pi * x[:, i]
                feats += [torch.cos(angle), torch.sin(angle)]
            else:
                feats.append(x[:, i])
        out = self.net(torch.cat([torch.stack(feats, 1), context], 1))
        return torch.split(out, self.sizes, dim=1)

    def transform(self, x, context, forward):
        raws = self.raw(x, context)
        y = x.clone()
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for raw, i in zip(raws, self.active, strict=True):
            if self.periodic[i]:
                fn = circle_forward if forward else circle_inverse
                yi, ld = fn(x[:, i], raw, self.n_bins)
            else:
                fn = rq_forward if forward else rq_inverse
                yi, ld = fn(x[:, i], raw, self.n_bins)
            y[:, i] = yi
            logdet = logdet + ld
        return y, logdet

    def forward(self, x, context):
        return self.transform(x, context, True)

    def inverse(self, y, context):
        return self.transform(y, context, False)


class ConditionalFlow(nn.Module):
    """q(u | w, f) on [0, 1]^dim with a uniform base: log q is the sum of log dets."""

    def __init__(
        self,
        n_samples: int,
        dim: int = 5,
        periodic: tuple = (2, 4),
        layers: int = 8,
        n_bins: int = 8,
        hidden: int = 256,
        context: int = 128,
        embed_width: int = 512,
        embed_blocks: int = 2,
    ):
        super().__init__()
        self.config = dict(
            n_samples=n_samples,
            dim=dim,
            periodic=tuple(periodic),
            layers=layers,
            n_bins=n_bins,
            hidden=hidden,
            context=context,
            embed_width=embed_width,
            embed_blocks=embed_blocks,
        )
        self.dim = dim
        self.embedding = Embedding(n_samples, embed_width, context, embed_blocks)
        self.layers = nn.ModuleList(
            Coupling(dim, a, periodic, context, hidden, n_bins)
            for a in active_sets(dim, layers)
        )

    def log_prob_given(self, u, context):
        logdet = torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)
        for layer in self.layers:
            u, ld = layer(u, context)
            logdet = logdet + ld
        return logdet

    def log_prob(self, u, w, frac):
        return self.log_prob_given(u, self.embedding(w, frac))

    def sample_given(self, context, generator=None):
        """Samples and their log density for each row of context."""
        z = torch.rand(
            context.shape[0],
            self.dim,
            generator=generator,
            dtype=context.dtype,
            device=context.device,
        )
        logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, context)
            logdet = logdet + ld
        return z, -logdet

    @torch.no_grad()
    def sample_many(self, w, frac, n, generator=None):
        """n samples for each of B observations: (B, n, dim) and log q (B, n)."""
        context = self.embedding(w, frac).repeat_interleave(n, dim=0)
        u, log_q = self.sample_given(context, generator)
        return u.reshape(w.shape[0], n, self.dim), log_q.reshape(w.shape[0], n)

    @torch.no_grad()
    def sample(self, w, frac, n, generator=None):
        """n samples for one observation w (2, n_samples) at fraction frac."""
        frac = torch.as_tensor(frac, dtype=w.dtype, device=w.device).reshape(1)
        u, log_q = self.sample_many(w[None], frac, n, generator)
        return u[0], log_q[0]
