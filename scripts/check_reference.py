"""Check the reference sampler: exact grid posterior, symmetry, two independent runs.

The grid check samples the two angles with 2000 particles, 6 moves and an
effective sample share of 0.5 (50,000 likelihood evaluations; the smoke run uses
400 particles). The two-run comparison uses the particles, moves and share of its
own preset, and the result file stores both sets of settings.
"""

from common import parser, results_dir, run_info, size

from chirpvi.config import n_obs_of
from chirpvi.experiments import grid_check, make_streams, median, truncate
from chirpvi.metrics import c2st, mode_summary
from chirpvi.reference import reference_posterior
from chirpvi.simulator import Simulator
from chirpvi.util import make_generator, write_json

PRESETS = {
    "smoke": dict(
        n_grid=60,
        grid_particles=400,
        grid_moves=6,
        grid_ess=0.5,
        n_cases=2,
        particles=200,
        fractions=(0.75,),
        moves=4,
        ess=0.5,
    ),
    "full": dict(
        n_grid=160,
        grid_particles=2000,
        grid_moves=6,
        grid_ess=0.5,
        n_cases=12,
        particles=2048,
        fractions=(0.5, 0.75, 1.0),
        moves=20,
        ess=0.7,
    ),
}


def main():
    args = parser(__doc__).parse_args()
    cfg = size(args, PRESETS)
    sim = Simulator(device=args.device)
    grid = grid_check(
        sim,
        192,
        cfg["n_grid"],
        cfg["grid_particles"],
        args.seed,
        cfg["grid_moves"],
        cfg["grid_ess"],
    )
    theta, w_full = make_streams(sim, cfg["n_cases"], args.seed + 1)
    runs = []
    for f in cfg["fractions"]:
        n_obs = n_obs_of(f, sim.problem)
        w = truncate(sim, w_full, n_obs)
        for j in range(cfg["n_cases"]):
            args_ref = (sim, w[j], n_obs, cfg["particles"], cfg["moves"], cfg["ess"])
            a = reference_posterior(
                *args_ref, generator=make_generator(10 + j, sim.device)
            )
            b = reference_posterior(
                *args_ref, generator=make_generator(500 + j, sim.device)
            )
            plain = reference_posterior(
                *args_ref,
                generator=make_generator(900 + j, sim.device),
                symmetrize=False,
            )
            ua, ub, up = (sim.to_unit(r.samples) for r in (a, b, plain))
            runs.append(
                dict(
                    fraction=f,
                    case=j,
                    c2st=c2st(ua, ub, sim.periodic, make_generator(j)),
                    plain_c2st=c2st(ua, up, sim.periodic, make_generator(j + 70)),
                    plain_mode_weight=mode_summary(up)["weight"],
                    evidence_gap=abs(a.log_evidence - b.log_evidence),
                    stages=a.stages,
                    evals=a.n_evals,
                    seconds=a.seconds,
                )
            )
    summary = {
        f"{f:g}": dict(
            median_c2st=median([r["c2st"] for r in runs if r["fraction"] == f]),
            max_c2st=max(r["c2st"] for r in runs if r["fraction"] == f),
            median_evidence_gap=median(
                [r["evidence_gap"] for r in runs if r["fraction"] == f]
            ),
            median_plain_c2st=median(
                [r["plain_c2st"] for r in runs if r["fraction"] == f]
            ),
            median_plain_mode_gap=median(
                [abs(r["plain_mode_weight"] - 0.5) for r in runs if r["fraction"] == f]
            ),
        )
        for f in cfg["fractions"]
    }
    out = dict(
        grid=grid,
        two_run=runs,
        two_run_summary=summary,
        particles=cfg["particles"],
        moves=cfg["moves"],
        ess_fraction=cfg["ess"],
        run=run_info(args, cfg),
    )
    print(write_json(results_dir(args) / "reference_check.json", out))


if __name__ == "__main__":
    main()
