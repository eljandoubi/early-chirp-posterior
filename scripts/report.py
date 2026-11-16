"""Print the README tables from the JSON files in results/."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ("0.5", "0.68", "0.9", "0.95")


def load(folder: Path, name: str):
    path = folder / name
    return json.loads(path.read_text()) if path.exists() else None


def num(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def table(title, headers, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


def setup(folder):
    snr = load(folder, "snr.json")
    if snr:
        rows = [
            [f, v["n_obs"], num(v["p10"], 1), num(v["median"], 1), num(v["p90"], 1)]
            for f, v in snr["snr"].items()
        ]
        table(
            "Signal to noise by observed fraction",
            ["f", "samples", "p10", "median", "p90"],
            rows,
        )


def checks(folder):
    ref = load(folder, "reference_check.json")
    if not ref:
        return
    g = ref["grid"]
    table(
        "Reference sampler against the exact grid posterior of phase and sky angle",
        ["KS phase", "KS sky", "mode weight", "log evidence gap", "symmetry error"],
        [
            [
                num(g["ks_phase"]),
                num(g["ks_sky"]),
                num(g["mode_weight"]),
                num(abs(g["sampler_log_evidence"] - g["grid_log_evidence"])),
                f"{g['symmetry_error']:.1e}",
            ]
        ],
    )
    rows = [
        [
            f,
            num(v["median_c2st"]),
            num(v["max_c2st"]),
            num(v["median_evidence_gap"]),
            num(v["median_plain_c2st"]),
            num(v["median_plain_mode_gap"]),
        ]
        for f, v in ref["two_run_summary"].items()
    ]
    table(
        "Two independent reference runs, and the sampler without the symmetry move",
        [
            "f",
            "median C2ST",
            "max C2ST",
            "median log evidence gap",
            "plain vs symmetrized C2ST",
            "plain mode gap",
        ],
        rows,
    )


def training(folder):
    for path in sorted(folder.glob("train_log_seed*.json")):
        s = json.loads(path.read_text())["summary"]
        table(
            f"Training ({path.stem})",
            ["steps", "minutes", "best validation NLL", "parameters", "device"],
            [
                [
                    s["steps"],
                    num(s["seconds"] / 60, 1),
                    num(s["best_val_nll"]),
                    s["parameters"],
                    s["device"],
                ]
            ],
        )


def calibration_tables(folder):
    ev = load(folder, "eval_flow.json")
    if not ev:
        return
    names = list(next(iter(ev.values()))["sbc"])
    rows = [
        [f] + [f"{v['sbc'][n]['ks']:.3f} ({v['sbc'][n]['p_value']:.2f})" for n in names]
        for f, v in ev.items()
    ]
    table("SBC: Kolmogorov distance of ranks (p value)", ["f"] + names, rows)
    rows = [[f] + [num(v["hpd"][c] - float(c)) for c in LEVELS] for f, v in ev.items()]
    table("HPD coverage minus level", ["f"] + list(LEVELS), rows)
    rows = [
        [f, num(v["tarp"]["max_deviation"]), num(v["tarp"]["mean_deviation"])]
        for f, v in ev.items()
    ]
    table(
        "TARP (negative mean deviation: overconfident)",
        ["f", "max deviation", "mean deviation"],
        rows,
    )


def reference_tables(folder):
    ev = load(folder, "eval_reference.json")
    if not ev:
        return
    s = ev["summary"]
    rows = [
        [
            f,
            num(v["median_snr"], 1),
            num(v["median_c2st"]),
            num(v["median_c2st_floor"]),
            num(v["median_is_efficiency"]),
        ]
        for f, v in s.items()
    ]
    table(
        "Flow against the reference",
        ["f", "median SNR", "C2ST", "C2ST floor", "IS efficiency"],
        rows,
    )
    rows = [
        [
            f,
            v["bimodal_cases"],
            num(v["flow_keeps_both_modes"], 2),
            num(v["flow_mode_gap"]),
            num(v["reference_mode_gap"]),
            num(v["median_c2st_symmetry"]),
        ]
        for f, v in s.items()
    ]
    table(
        "Mode recovery",
        [
            "f",
            "bimodal cases",
            "flow keeps both",
            "flow gap",
            "reference gap",
            "C2ST flow vs its image",
        ],
        rows,
    )
    rows = []
    for f, v in s.items():
        r, q = v["width_reference"], v["width_ratio"]
        rows.append([f] + [num(r[k]) for k in r] + [num(q[k], 2) for k in q])
    table(
        "Posterior width over prior width (reference) and flow over reference",
        ["f", "mass", "time", "cos i", "ratio mass", "ratio time", "ratio cos i"],
        rows,
    )


def sequential_table(folder):
    sq = load(folder, "sequential.json")
    if not sq:
        return
    order = {"scratch": 0, "chain_sampler": 1, "chain_flow": 2}

    def position(item):
        variant, n = item[0].split("@")
        return order[variant], int(n)

    rows = [
        [
            k,
            num(v["evals"], 0),
            num(v["seconds"], 2),
            num(v["stages"], 0),
            num(v["ess_first"], 2),
            num(v["c2st"]),
        ]
        for k, v in sorted(sq["summary"].items(), key=position)
    ]
    table(
        f"Sequential (C2ST floor {num(sq['c2st_floor'])})",
        [
            "variant@samples",
            "evaluations",
            "seconds",
            "stages",
            "first ESS fraction",
            "C2ST vs scratch",
        ],
        rows,
    )


def cost(folder):
    paths = sorted(folder.glob("timing_*.json")) or [folder / "timing.json"]
    rows = []
    for path in paths:
        t = load(folder, path.name)
        if t:
            rows.append(
                [
                    t["device"],
                    num(t["flow_seconds_one"], 3),
                    num(t["flow_seconds_amortised"], 4),
                    num(t["reference_seconds"], 2),
                    t["reference_evals"],
                    num(t["load_average"], 1),
                ]
            )
    if rows:
        table(
            f"Cost of {t['n_samples']} posterior samples ({t['n_obs']} observed)",
            [
                "device",
                "flow one",
                "flow amortised",
                "reference",
                "reference evaluations",
                "load average",
            ],
            rows,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="read results/smoke")
    args = p.parse_args()
    folder = ROOT / "results" / "smoke" if args.smoke else ROOT / "results"
    for fn in (
        setup,
        checks,
        training,
        calibration_tables,
        reference_tables,
        sequential_table,
        cost,
    ):
        fn(folder)


if __name__ == "__main__":
    main()
