"""Draw the two figures of the README from the JSON files in results/."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
plt.switch_backend("Agg")


def sbc_figure(ev: dict, path) -> str:
    """Rank histograms (count over uniform count) per fraction and function."""
    names = sorted(
        next(iter(ev.values()))["sbc"], key=lambda n: (n == "log_likelihood", n)
    )
    fig, axes = plt.subplots(
        len(ev),
        len(names),
        figsize=(1.5 * len(names), 0.95 * len(ev) + 0.6),
        sharey=True,
        squeeze=False,
    )
    for i, (frac, row) in enumerate(ev.items()):
        for j, name in enumerate(names):
            ax = axes[i, j]
            hist = row["sbc"][name]["hist"]
            ratio = [
                c / e for c, e in zip(hist["counts"], hist["expected"], strict=True)
            ]
            band = 2 / math.sqrt(sum(hist["expected"]) / len(ratio))
            ax.axhspan(1 - band, 1 + band, color="0.88", lw=0)
            ax.bar(range(len(ratio)), ratio, width=0.9, color="C0")
            ax.set_ylim(0, 3)
            ax.set_xticks([])
            ax.set_yticks([0, 1, 2, 3] if j == 0 else [])
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f"f = {frac}", fontsize=7)
            if i == 0:
                ax.set_title(name.replace("_", " "), fontsize=7)
    fig.suptitle(
        "SBC rank histograms of the flow: count over uniform count "
        "(grey: 2 s.d., bars clipped at 3)",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.3, w_pad=0.2)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def width_figure(ev: dict, path) -> str:
    """Posterior std over prior std against the observed fraction."""
    summary = ev["summary"]
    fractions = [float(f) for f in summary]
    keys = list(next(iter(summary.values()))["width_reference"])
    fig, axes = plt.subplots(
        1, len(keys), figsize=(3.1 * len(keys), 2.8), sharey=True, squeeze=False
    )
    for ax, key in zip(axes[0], keys, strict=True):
        ref = [v["width_reference"][key] for v in summary.values()]
        flow = [v["width_flow"][key] for v in summary.values()]
        ax.plot(fractions, ref, "o-", label="reference")
        ax.plot(fractions, flow, "s--", label="flow")
        ax.set_yscale("log")
        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.set_xlabel("observed fraction f", fontsize=8)
    axes[0, 0].set_ylabel("posterior std / prior std", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def main():
    results = ROOT / "results"
    flow = json.loads((results / "eval_flow.json").read_text())
    reference = json.loads((results / "eval_reference.json").read_text())
    print(sbc_figure(flow, results / "sbc_ranks.png"))
    print(width_figure(reference, results / "posterior_width.png"))


if __name__ == "__main__":
    main()
