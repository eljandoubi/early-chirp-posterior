"""Evaluate the trained flow: calibration, agreement with the reference, cost."""

from common import checkpoint_path, parser, results_dir, size

from chirpvi.config import FRACTIONS, n_obs_of
from chirpvi.experiments import calibration, compare_with_reference, timing
from chirpvi.simulator import Simulator
from chirpvi.training import FlowPosterior, load_flow
from chirpvi.util import write_json

PRESETS = {
    "smoke": dict(
        fractions=(0.5, 1.0),
        n_test=40,
        n_samples=50,
        chunk=20,
        loglik_cases=10,
        n_cases=2,
        particles=128,
        floor_cases=1,
        moves=3,
        ess=0.5,
        time_samples=100,
        repeats=2,
    ),
    "full": dict(
        fractions=FRACTIONS,
        n_test=1000,
        n_samples=200,
        chunk=100,
        loglik_cases=300,
        n_cases=20,
        particles=2048,
        floor_cases=5,
        moves=20,
        ess=0.7,
        time_samples=2048,
        repeats=5,
    ),
}


def main():
    p = parser(__doc__)
    p.add_argument(
        "--part",
        choices=("calibration", "reference", "timing", "all"),
        default="all",
        help="run one part (each writes its own file)",
    )
    args = p.parse_args()
    cfg = size(args, PRESETS)
    out = results_dir(args)
    sim = Simulator(device=args.device)
    posterior = FlowPosterior(load_flow(checkpoint_path(args), args.device), sim)
    if args.part in ("calibration", "all"):
        res = calibration(
            sim,
            posterior,
            cfg["fractions"],
            cfg["n_test"],
            cfg["n_samples"],
            args.seed,
            cfg["chunk"],
            cfg["loglik_cases"],
        )
        print(write_json(out / "eval_flow.json", res))
    if args.part in ("reference", "all"):
        res = compare_with_reference(
            sim,
            posterior,
            cfg["fractions"],
            cfg["n_cases"],
            cfg["particles"],
            args.seed + 1,
            cfg["floor_cases"],
            cfg["moves"],
            cfg["ess"],
        )
        print(write_json(out / "eval_reference.json", res))
    if args.part in ("timing", "all"):
        n_obs = n_obs_of(0.75, sim.problem)
        res = timing(
            sim,
            posterior,
            n_obs,
            cfg["time_samples"],
            cfg["particles"],
            cfg["repeats"],
            cfg["moves"],
            cfg["ess"],
        )
        print(write_json(out / "timing.json", res))


if __name__ == "__main__":
    main()
