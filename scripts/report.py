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


def seed_tag(name: str, stem: str) -> str:
    """Title suffix for a file like eval_flow_seed1.json; empty for the base file."""
    return "" if name == f"{stem}.json" else ", " + name[len(stem) + 1 : -5]


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


def calibration_tables(folder, name="eval_flow.json"):
    ev = load(folder, name)
    if not ev:
        return
    tag = seed_tag(name, "eval_flow")
    names = list(next(iter(ev.values()))["sbc"])
    rows = [
        [f] + [f"{v['sbc'][n]['ks']:.3f} ({v['sbc'][n]['p_value']:.2f})" for n in names]
        for f, v in ev.items()
    ]
    table(f"SBC: Kolmogorov distance of ranks (p value){tag}", ["f"] + names, rows)
    rows = [[f] + [num(v["hpd"][c] - float(c)) for c in LEVELS] for f, v in ev.items()]
    table(f"HPD coverage minus level{tag}", ["f"] + list(LEVELS), rows)
    rows = [
        [f, num(v["tarp"]["max_deviation"]), num(v["tarp"]["mean_deviation"])]
        for f, v in ev.items()
    ]
    table(
        f"TARP (negative mean deviation: overconfident){tag}",
        ["f", "max deviation", "mean deviation"],
        rows,
    )


def reference_tables(folder, name="eval_reference.json"):
    ev = load(folder, name)
    if not ev:
        return
    tag = seed_tag(name, "eval_reference")
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
        f"Flow against the reference{tag}",
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
        f"Mode recovery{tag}",
        [
            "f",
            "bimodal cases",
            "flow weight within 0.2 of half",
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
        f"Posterior width over prior width (reference) and flow over reference{tag}",
        ["f", "mass", "time", "cos i", "ratio mass", "ratio time", "ratio cos i"],
        rows,
    )


def seed_label(seed: str) -> str:
    return "" if seed == "seed0" else f", {seed}"


def modes_table(folder):
    data = load(folder, "modes_concentration.json")
    for seed, v in (data or {}).items():
        rows = [[f, *mode_cells(c)] for f, c in v["fractions"].items()]
        rows.append(["all", *mode_cells(v["total"])])
        table(
            f"Flow against the reference test for two modes{seed_label(seed)}",
            [
                "f",
                "bimodal cases",
                "weight within 0.2 of half",
                "passes reference test",
            ],
            rows,
        )


def mode_cells(c):
    return [c["bimodal_cases"], c["weight_near_half"], c["passes_reference_test"]]


def paired_table(folder):
    data = load(folder, "paired_excess.json")
    for seed, v in (data or {}).items():
        rows = [
            [
                f,
                p["cases"],
                num(p["median_c2st"]),
                num(p["median_floor"]),
                num(p["median_excess"]),
                f"{num(p['min_excess'])} to {num(p['max_excess'])}",
            ]
            for f, p in v["fractions"].items()
        ]
        table(
            f"Flow C2ST above the floor, paired injections{seed_label(seed)}",
            ["f", "paired", "C2ST", "floor", "median excess", "excess range"],
            rows,
        )


def spread_table(folder):
    data = load(folder, "width_spread.json")
    for seed, v in (data or {}).items():
        rows = []
        for f, names in v["fractions"].items():
            cells = [
                f"{num(s['median'], 2)} ({num(s['min'], 2)} to {num(s['max'], 2)}), "
                f"{s['outside']} of {s['cases']}"
                for s in names.values()
            ]
            rows.append([f, *cells])
        table(
            "Flow over reference width per injection: median (range), cases "
            f"off by more than {v['tolerance']:.0%}{seed_label(seed)}",
            ["f", "mass", "time", "cos i"],
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
            "proposal correction ESS",
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


def evaluation(folder, stem, seed):
    """Result file of one training seed; the first seed's file has no suffix."""
    data = load(folder, f"{stem}_seed{seed}.json")
    if data is None and seed == 0:
        data = load(folder, f"{stem}.json")
    return data


def flow_cells(flow):
    marginal = [
        (v["ks"], v["p_value"])
        for row in flow.values()
        for name, v in row["sbc"].items()
        if name != "log_likelihood"
    ]
    joint = [row["sbc"]["log_likelihood"] for row in flow.values()]
    top = flow[max(flow, key=float)]["sbc"]["log_likelihood"]["hist"]["counts"]
    gaps = [row["hpd"][c] - float(c) for row in flow.values() for c in row["hpd"]]
    tarp = [row["tarp"]["max_deviation"] for row in flow.values()]
    return [
        f"{sum(p < 0.05 for _, p in marginal)} of {len(marginal)}",
        num(max(k for k, _ in marginal)),
        f"{num(min(v['ks'] for v in joint))} to {num(max(v['ks'] for v in joint))}",
        f"{top[-1]} of {sum(top)}",
        f"{num(min(gaps))} to {num(max(gaps))}",
        num(max(tarp)),
    ]


def reference_cells(ref):
    s = ref["summary"]
    ratio = " / ".join(num(x, 2) for x in s["1"]["width_ratio"].values())
    return [num(s["0.75"]["median_c2st"]), num(s["1"]["median_c2st"]), ratio]


def seed_summary(folder):
    rows = []
    for path in sorted(folder.glob("train_log_seed*.json")):
        seed = int(path.stem.removeprefix("train_log_seed"))
        best = json.loads(path.read_text())["summary"]["best_val_nll"]
        flow = evaluation(folder, "eval_flow", seed)
        ref = evaluation(folder, "eval_reference", seed)
        rows.append(
            [seed, num(best)]
            + (flow_cells(flow) if flow else ["n/a"] * 6)
            + (reference_cells(ref) if ref else ["n/a"] * 3)
        )
    if len(rows) > 1:
        table(
            "Training runs side by side",
            [
                "seed",
                "best validation NLL",
                "marginal SBC p under 0.05",
                "largest marginal distance",
                "log likelihood distance",
                "log likelihood top bin at f = 1",
                "HPD minus level",
                "TARP max deviation",
                "C2ST at f = 0.75",
                "C2ST at f = 1",
                "width ratio at f = 1 (mass, time, cos i)",
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
        modes_table,
        paired_table,
        spread_table,
        sequential_table,
        cost,
    ):
        fn(folder)
    seed_summary(folder)
    for path in sorted(folder.glob("eval_flow_seed*.json")):
        calibration_tables(folder, path.name)
    for path in sorted(folder.glob("eval_reference_seed*.json")):
        reference_tables(folder, path.name)


if __name__ == "__main__":
    main()
