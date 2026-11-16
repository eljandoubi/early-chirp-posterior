"""Online training of the amortised flow and a posterior wrapper for inference."""

import math
import time

import torch

from .flow import ConditionalFlow
from .simulator import Simulator
from .util import load_average, make_generator


def training_batch(sim: Simulator, batch: int, n_min: int, generator=None):
    """Prior draws, whitened data at random fractions: (u, w, fraction)."""
    theta = sim.sample_prior(batch, generator)
    n_obs = torch.randint(
        n_min, sim.problem.n + 1, (batch,), generator=generator, device=sim.device
    )
    w = sim.observe(theta, n_obs, generator)
    return sim.to_unit(theta).float(), w.float(), (n_obs / sim.problem.n).float()


@torch.no_grad()
def mean_nll(flow, data, chunk: int = 1024) -> float:
    total, count = 0.0, 0
    for i in range(0, data[0].shape[0], chunk):
        u, w, frac = (d[i : i + chunk] for d in data)
        total += -flow.log_prob(u, w, frac).sum().item()
        count += u.shape[0]
    return total / count


def train_flow(
    sim: Simulator,
    flow: ConditionalFlow,
    steps: int,
    batch: int = 512,
    lr: float = 1e-3,
    lr_end: float = 1e-5,
    n_min: int = 32,
    seed: int = 0,
    val_size: int = 4096,
    val_every: int = 500,
    minutes: float | None = None,
    log=None,
) -> dict:
    """Maximise log q(u | w, f) on fresh simulations; keep the best validation weights.

    With minutes set, the first 30 steps are timed and the step count is chosen to
    fill the budget. Returns the history and a summary.
    """
    flow.to(sim.device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    gen = make_generator(seed, sim.device)
    val = training_batch(
        sim, val_size, n_min, make_generator(seed + 10_000, sim.device)
    )
    total = max(steps, 31) if minutes else steps
    history, best, best_state, run = [], math.inf, None, 0.0
    t0 = time.perf_counter()
    step = 0
    while step < total:
        if minutes and step == 30:
            per_step = (time.perf_counter() - t0) / 30
            total = max(31, int(0.95 * minutes * 60 / per_step))
        progress = step / max(total, 1)
        rate = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * progress))
        for group in opt.param_groups:
            group["lr"] = rate
        u, w, frac = training_batch(sim, batch, n_min, gen)
        loss = -flow.log_prob(u, w, frac).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        run = loss.item() if step == 0 else 0.98 * run + 0.02 * loss.item()
        step += 1
        if step % val_every == 0 or step == total:
            nll = mean_nll(flow, val)
            history.append(
                dict(
                    step=step,
                    train_nll=run,
                    val_nll=nll,
                    lr=rate,
                    seconds=time.perf_counter() - t0,
                )
            )
            if log:
                log(history[-1])
            if nll < best:
                best = nll
                best_state = {
                    k: v.detach().clone() for k, v in flow.state_dict().items()
                }
    if best_state is not None:
        flow.load_state_dict(best_state)
    summary = dict(
        steps=step,
        batch=batch,
        seconds=time.perf_counter() - t0,
        best_val_nll=best,
        parameters=sum(p.numel() for p in flow.parameters()),
        load_average=load_average(),
        device=str(sim.device),
    )
    return dict(history=history, summary=summary)


def save_flow(path, flow: ConditionalFlow, extra: dict | None = None) -> None:
    torch.save(
        dict(config=flow.config, state=flow.state_dict(), extra=extra or {}), path
    )


def load_flow(path, device="cpu") -> ConditionalFlow:
    blob = torch.load(path, map_location=device, weights_only=False)
    flow = ConditionalFlow(**blob["config"])
    flow.load_state_dict(blob["state"])
    return flow.to(device).eval()


class FlowPosterior:
    """Posterior samples in parameter units from a trained flow."""

    def __init__(self, flow: ConditionalFlow, sim: Simulator):
        self.flow, self.sim = flow.eval(), sim

    def sample_many(self, w, n_obs, n: int, generator=None):
        """Samples (B, n, 5) and log q (B, n) in unit cube coordinates."""
        frac = torch.as_tensor(n_obs, device=self.sim.device) / self.sim.problem.n
        u, log_q = self.flow.sample_many(w.float(), frac.float(), n, generator)
        return self.sim.from_unit(u.double()), log_q.double()

    def sample(self, w, n_obs: int, n: int, generator=None):
        theta, log_q = self.sample_many(w[None], torch.tensor([n_obs]), n, generator)
        return theta[0], log_q[0]

    @torch.no_grad()
    def log_prob(self, theta, w, n_obs):
        """Log density of rows of theta in unit coordinates, one observation each."""
        frac = torch.as_tensor(n_obs, device=self.sim.device) / self.sim.problem.n
        u = self.sim.to_unit(theta).float()
        return self.flow.log_prob(u, w.float(), frac.float()).double()
