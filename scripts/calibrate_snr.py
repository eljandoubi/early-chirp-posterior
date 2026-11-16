"""Signal to noise ratio by observed fraction for the chosen amplitude constant."""

import torch
from common import parser, results_dir, size

from chirpvi.config import FRACTIONS, n_obs_of
from chirpvi.noise import psd
from chirpvi.simulator import Simulator
from chirpvi.util import make_generator, write_json

PRESETS = {"smoke": dict(n=400), "full": dict(n=20000)}


def main():
    args = parser(__doc__).parse_args()
    sim = Simulator(device=args.device)
    theta = sim.sample_prior(
        size(args, PRESETS)["n"], make_generator(args.seed, sim.device)
    )
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    table = {}
    for f in FRACTIONS:
        n_obs = n_obs_of(f, sim.problem)
        snr = sim.snr(theta, torch.full((theta.shape[0],), n_obs))
        q = torch.quantile(snr, quantiles.to(snr.device))
        table[f"{f:g}"] = dict(
            n_obs=n_obs, p10=q[0].item(), median=q[1].item(), p90=q[2].item()
        )
    freq = sim.frequency(theta)
    hz = torch.tensor([13.0, 35.0], dtype=torch.float64)
    out = dict(
        amplitude=sim.problem.amplitude,
        snr=table,
        frequency_start_hz=[freq[:, 0].min().item(), freq[:, 0].max().item()],
        frequency_end_hz=[freq[:, -1].min().item(), freq[:, -1].max().item()],
        psd_ratio_13_over_35_hz=(
            psd(hz[0], sim.problem) / psd(hz[1], sim.problem)
        ).item(),
        n_prior_draws=theta.shape[0],
    )
    print(write_json(results_dir(args) / "snr.json", out))


if __name__ == "__main__":
    main()
