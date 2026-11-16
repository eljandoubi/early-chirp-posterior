"""Experiments behind the scripts: calibration, reference comparison, cost."""

import math
import statistics
import time

import torch

from .config import n_obs_of
from .metrics import (
    FUNCTION_NAMES,
    LINEAR,
    c2st,
    central_coverage,
    hpd_coverage,
    importance_efficiency,
    is_bimodal,
    mode_summary,
    pair_image,
    sbc_ranks,
    tarp,
    tarp_summary,
    uniform_ks,
    widths,
)
from .reference import (
    Likelihood,
    angle_grid_posterior,
    angle_posterior,
    reference_posterior,
)
from .sequential import sequential_path
from .simulator import Simulator
from .training import FlowPosterior
from .util import load_average, make_generator

LEVELS = (0.5, 0.68, 0.9, 0.95)


def make_streams(sim: Simulator, n: int, seed: int):
    """Prior draws and their full whitened streams (n, 2, N)."""
    gen = make_generator(seed, sim.device)
    theta = sim.sample_prior(n, gen)
    full = torch.full((n,), sim.problem.n, device=sim.device)
    return theta, sim.observe(theta, full, gen)


def truncate(sim: Simulator, w_full: torch.Tensor, n_obs: int) -> torch.Tensor:
    return w_full * sim.mask(torch.full((w_full.shape[0],), n_obs))


def median(values) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(values) if values else float("nan")


def grid_check(sim: Simulator, n_obs=192, n_grid=120, n_particles=1500, seed=0):
    """Sampler against the exact (phase, sky) grid posterior, other parameters fixed."""
    gen = make_generator(seed, sim.device)
    theta = sim.sample_prior(1, gen)
    theta[0, 0], theta[0, 1], theta[0, 2] = 21.0, 2.34, 1.0
    theta[0, 3], theta[0, 4] = 0.4, 0.9
    w = sim.observe(theta, torch.tensor([n_obs], device=sim.device), gen)[0]
    prob, log_z = angle_grid_posterior(sim, w, n_obs, theta, n_grid)
    res = angle_posterior(sim, w, n_obs, theta, n_particles, gen)
    out = dict(grid_log_evidence=log_z, sampler_log_evidence=res.log_evidence)
    for axis, name in ((0, "phase"), (1, "sky")):
        span = float(sim.hi[2 + 2 * axis] - sim.lo[2 + 2 * axis])
        edges = torch.arange(1, n_grid + 1, dtype=torch.float64) / n_grid * span
        cdf = torch.cumsum(prob.sum(1 - axis), 0).cpu()
        ecdf = (res.samples[:, axis][:, None].cpu() <= edges[None, :]).double().mean(0)
        out[f"ks_{name}"] = (cdf - ecdf).abs().max().item()
    image = pair_image(sim.to_unit(theta))
    lik = Likelihood(sim, w)
    ell_a = lik.at(theta, n_obs)[0]
    ell_b = lik.at(sim.from_unit(image), n_obs)[0]
    out["symmetry_error"] = (ell_a - ell_b).abs().item()
    first = ((res.samples[:, 1] - 0.9 + math.pi / 4) % math.pi) < math.pi / 2
    out["mode_weight"] = first.double().mean().item()
    out["n_evals"] = res.n_evals
    return out


def calibration(
    sim,
    posterior: FlowPosterior,
    fractions,
    n_test,
    n_samples,
    seed,
    chunk=100,
    loglik_cases=None,
):
    """SBC, HPD and central coverage and TARP for the flow at each fraction."""
    out = {}
    for f in fractions:
        n_obs = n_obs_of(f, sim.problem)
        theta, w_full = make_streams(sim, n_test, seed + n_obs)
        w = truncate(sim, w_full, n_obs)
        n_vec = torch.full((n_test,), n_obs, device=sim.device)
        gen = make_generator(seed + 7 * n_obs, sim.device)
        parts = [
            posterior.sample_many(
                w[i : i + chunk], n_vec[i : i + chunk], n_samples, gen
            )
            for i in range(0, n_test, chunk)
        ]
        samples = torch.cat([p[0] for p in parts])
        log_q = torch.cat([p[1] for p in parts])
        log_q_true = posterior.log_prob(theta, w, n_vec)
        u_true, u_samp = sim.to_unit(theta), sim.to_unit(samples)
        ranks = sbc_ranks(u_true, u_samp)
        sbc = {
            name: uniform_ks(ranks[:, k], n_samples)
            for k, name in enumerate(FUNCTION_NAMES)
        }
        cases = n_test if loglik_cases is None else min(loglik_cases, n_test)
        ell_ranks = []
        for i in range(cases):
            (ell,) = Likelihood(sim, w[i]).at(
                torch.cat([theta[i : i + 1], samples[i]]), n_obs
            )
            ell_ranks.append((ell[1:] < ell[0]).sum())
        sbc["log_likelihood"] = uniform_ks(torch.stack(ell_ranks), n_samples)
        reference = torch.rand(
            n_test, 5, generator=gen, dtype=u_true.dtype, device=sim.device
        )
        levels, ecp = tarp(u_true, u_samp, reference, sim.periodic)
        central = central_coverage(
            u_true[:, list(LINEAR)], u_samp[..., list(LINEAR)], LEVELS
        )
        out[f"{f:g}"] = dict(
            n_obs=n_obs,
            n_test=n_test,
            n_samples=n_samples,
            sbc=sbc,
            hpd={str(c): v for c, v in hpd_coverage(log_q_true, log_q, LEVELS).items()},
            central={str(c): v.tolist() for c, v in central.items()},
            tarp=tarp_summary(levels, ecp),
            mean_nll=-log_q_true.mean().item(),
        )
    return out


def summarise_reference(records, fractions):
    table = {}
    for f in fractions:
        rows = [r for r in records if r["fraction"] == f]
        both = [r for r in rows if is_bimodal(r["mode_ref"])]
        keep = [abs(r["mode_flow"]["weight"] - 0.5) < 0.2 for r in both]
        ratio = {
            k: median([r["widths_flow"][k] / r["widths_ref"][k] for r in rows])
            for k in rows[0]["widths_ref"]
        }
        table[f"{f:g}"] = dict(
            cases=len(rows),
            median_snr=median([r["snr"] for r in rows]),
            median_c2st=median([r["c2st"] for r in rows]),
            median_c2st_floor=median([r.get("c2st_floor") for r in rows]),
            median_c2st_symmetry=median([r["c2st_symmetry"] for r in rows]),
            median_is_efficiency=median([r["is_efficiency"] for r in rows]),
            bimodal_cases=len(both),
            flow_keeps_both_modes=(sum(keep) / len(keep)) if keep else None,
            flow_mode_gap=median([abs(r["mode_flow"]["weight"] - 0.5) for r in both]),
            reference_mode_gap=median(
                [abs(r["mode_ref"]["weight"] - 0.5) for r in both]
            ),
            width_reference={
                k: median([r["widths_ref"][k] for r in rows]) for k in ratio
            },
            width_flow={k: median([r["widths_flow"][k] for r in rows]) for k in ratio},
            width_ratio=ratio,
            median_reference_seconds=median([r["ref_seconds"] for r in rows]),
            median_reference_evals=median([r["ref_evals"] for r in rows]),
        )
    return table


def compare_with_reference(
    sim, posterior, fractions, n_cases, n_particles, seed, floor_cases=3, n_moves=6
):
    """Flow against the exact reference on injections shared across fractions."""
    theta, w_full = make_streams(sim, n_cases, seed)
    records = []
    for f in fractions:
        n_obs = n_obs_of(f, sim.problem)
        w = truncate(sim, w_full, n_obs)
        for j in range(n_cases):
            gen = make_generator(seed * 1000 + 31 * n_obs + j, sim.device)
            ref = reference_posterior(
                sim, w[j], n_obs, n_particles, n_moves, generator=gen
            )
            flow_theta, log_q = posterior.sample(w[j], n_obs, n_particles, gen)
            u_ref, u_flow = sim.to_unit(ref.samples), sim.to_unit(flow_theta)
            (ell,) = Likelihood(sim, w[j]).at(flow_theta, n_obs)
            rec = dict(
                fraction=f,
                case=j,
                n_obs=n_obs,
                snr=sim.snr(theta[j : j + 1], torch.tensor([n_obs])).item(),
                c2st=c2st(
                    u_flow.cpu(), u_ref.cpu(), sim.periodic.cpu(), make_generator(j)
                ),
                c2st_symmetry=c2st(
                    u_flow.cpu(),
                    pair_image(u_flow).cpu(),
                    sim.periodic.cpu(),
                    make_generator(j + 50),
                ),
                is_efficiency=importance_efficiency(ell, log_q),
                mode_ref=mode_summary(u_ref),
                mode_flow=mode_summary(u_flow),
                widths_ref=widths(u_ref),
                widths_flow=widths(u_flow),
                ref_seconds=ref.seconds,
                ref_evals=ref.n_evals,
                ref_stages=ref.stages,
                ref_log_evidence=ref.log_evidence,
            )
            if j < floor_cases:
                second = reference_posterior(
                    sim,
                    w[j],
                    n_obs,
                    n_particles,
                    n_moves,
                    generator=make_generator(j + 99, sim.device),
                )
                rec["c2st_floor"] = c2st(
                    u_ref.cpu(), sim.to_unit(second.samples).cpu(), sim.periodic.cpu()
                )
            records.append(rec)
    return dict(records=records, summary=summarise_reference(records, fractions))


def sequential_experiment(sim, posterior, path, n_cases, n_particles, seed, n_moves=6):
    """Cost and quality of three ways to reach each fraction along one stream."""
    theta, w_full = make_streams(sim, n_cases, seed)
    rows = []
    for j in range(n_cases):
        w = w_full[j]
        gen = make_generator(seed * 1000 + j, sim.device)
        scratch = {
            n: reference_posterior(sim, w, n, n_particles, n_moves, generator=gen)
            for n in path
        }
        extra = reference_posterior(
            sim, w, path[-1], n_particles, n_moves, generator=gen
        )
        chains = {
            "chain_sampler": sequential_path(
                sim, w, list(path), scratch[path[0]].samples, None, n_moves, gen
            )
        }
        w0 = truncate(sim, w[None], path[0])[0]
        flow_theta, log_q = posterior.sample(w0, path[0], n_particles, gen)
        chains["chain_flow"] = sequential_path(
            sim, w, list(path), flow_theta, log_q, n_moves, gen
        )
        for name, records in chains.items():
            for rec in records:
                res, n = rec["result"], rec["n_obs"]
                rows.append(
                    dict(
                        case=j,
                        variant=name,
                        n_obs=n,
                        evals=res.n_evals,
                        seconds=res.seconds,
                        stages=res.stages,
                        ess_first=res.ess_first,
                        c2st=c2st(
                            sim.to_unit(res.samples).cpu(),
                            sim.to_unit(scratch[n].samples).cpu(),
                            sim.periodic.cpu(),
                        ),
                        mode_gap=abs(
                            mode_summary(sim.to_unit(res.samples))["weight"] - 0.5
                        ),
                    )
                )
        for n in path[1:]:
            res = scratch[n]
            rows.append(
                dict(
                    case=j,
                    variant="scratch",
                    n_obs=n,
                    evals=res.n_evals,
                    seconds=res.seconds,
                    stages=res.stages,
                    ess_first=1.0,
                    c2st=None,
                    mode_gap=abs(
                        mode_summary(sim.to_unit(res.samples))["weight"] - 0.5
                    ),
                )
            )
        rows.append(
            dict(
                case=j,
                variant="scratch_floor",
                n_obs=path[-1],
                evals=extra.n_evals,
                seconds=extra.seconds,
                stages=extra.stages,
                ess_first=1.0,
                c2st=c2st(
                    sim.to_unit(extra.samples).cpu(),
                    sim.to_unit(scratch[path[-1]].samples).cpu(),
                    sim.periodic.cpu(),
                ),
                mode_gap=None,
            )
        )
    table = {}
    for variant in ("scratch", "chain_sampler", "chain_flow"):
        for n in path[1:]:
            sel = [r for r in rows if r["variant"] == variant and r["n_obs"] == n]
            table[f"{variant}@{n}"] = {
                k: median([r[k] for r in sel])
                for k in ("evals", "seconds", "stages", "ess_first", "c2st", "mode_gap")
            }
    floor = [r["c2st"] for r in rows if r["variant"] == "scratch_floor"]
    return dict(rows=rows, summary=table, c2st_floor=median(floor), path=list(path))


def timing(sim, posterior, n_obs, n_samples, n_particles, repeats=5):
    """Seconds for n_samples draws: flow (one, 100 amortised) and the sampler."""
    theta, w_full = make_streams(sim, 100, 5)
    w = truncate(sim, w_full, n_obs)
    gen = make_generator(1, sim.device)

    def clock(fn):
        fn()
        times = []
        for _ in range(repeats):
            if sim.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if sim.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    n_vec = torch.full((100,), n_obs, device=sim.device)
    one = clock(lambda: posterior.sample(w[0], n_obs, n_samples, gen))
    many = clock(lambda: posterior.sample_many(w, n_vec, n_samples, gen)) / 100
    ref = [
        reference_posterior(sim, w[0], n_obs, n_particles, generator=gen)
        for _ in range(3)
    ]
    name = torch.cuda.get_device_name() if sim.device.type == "cuda" else "cpu"
    return dict(
        device=name,
        n_obs=n_obs,
        n_samples=n_samples,
        flow_seconds_one=one,
        flow_seconds_amortised=many,
        reference_seconds=median([r.seconds for r in ref]),
        reference_evals=ref[0].n_evals,
        load_average=load_average(),
        threads=torch.get_num_threads(),
    )
