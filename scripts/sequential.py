"""Sequential variant: cost and quality of reusing the posterior at one fraction."""

from common import checkpoint_path, parser, results_dir, run_info, size

from chirpvi.config import FRACTIONS, n_obs_of
from chirpvi.experiments import sequential_experiment
from chirpvi.simulator import Simulator
from chirpvi.training import FlowPosterior, load_flow
from chirpvi.util import write_json

PRESETS = {
    "smoke": dict(
        path=(96, 160, 256),
        n_cases=1,
        particles=128,
        moves=3,
        ess=0.5,
        strict=False,
    ),
    "full": dict(
        path=tuple(n_obs_of(f, Simulator().problem) for f in FRACTIONS),
        n_cases=10,
        particles=2048,
        moves=20,
        ess=0.7,
        strict=True,
    ),
}


def main():
    args = parser(__doc__).parse_args()
    cfg = size(args, PRESETS)
    sim = Simulator(device=args.device)
    posterior = FlowPosterior(load_flow(checkpoint_path(args), args.device), sim)
    res = sequential_experiment(
        sim,
        posterior,
        cfg["path"],
        cfg["n_cases"],
        cfg["particles"],
        args.seed + 2,
        cfg["moves"],
        cfg["ess"],
        log=print,
        strict=cfg["strict"],
    )
    res["run"] = run_info(args, cfg)
    print(write_json(results_dir(args) / "sequential.json", res))


if __name__ == "__main__":
    main()
