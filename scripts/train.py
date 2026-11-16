"""Train the amortised flow on online simulations (fresh data at every step)."""

import torch
from common import checkpoint_path, parser, results_dir, size

from chirpvi.flow import ConditionalFlow
from chirpvi.simulator import Simulator
from chirpvi.training import save_flow, train_flow
from chirpvi.util import write_json

PRESETS = {
    "smoke": dict(
        model=dict(
            layers=2, n_bins=4, hidden=32, context=16, embed_width=64, embed_blocks=1
        ),
        steps=60,
        batch=32,
        val_size=256,
        val_every=20,
    ),
    "full": dict(model={}, steps=20000, batch=512, val_size=4096, val_every=500),
}


def main():
    p = parser(__doc__)
    p.add_argument("--minutes", type=float, default=None, help="fill this time budget")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="default 512 (smoke 32)")
    args = p.parse_args()
    cfg = size(args, PRESETS)
    sim = Simulator(device=args.device)
    torch.manual_seed(args.seed)
    flow = ConditionalFlow(sim.problem.n, **cfg["model"])
    out = train_flow(
        sim,
        flow,
        steps=args.steps or cfg["steps"],
        batch=args.batch or cfg["batch"],
        seed=args.seed,
        val_size=cfg["val_size"],
        val_every=cfg["val_every"],
        minutes=args.minutes,
        log=print,
    )
    save_flow(checkpoint_path(args), flow, dict(seed=args.seed, **out["summary"]))
    log = dict(model=flow.config, seed=args.seed, minutes_requested=args.minutes, **out)
    print(write_json(results_dir(args) / f"train_log_seed{args.seed}.json", log))


if __name__ == "__main__":
    main()
