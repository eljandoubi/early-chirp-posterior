"""Derived numbers from the per-case records of the reference comparison.

Reads results/eval_reference*.json and writes modes_concentration.json,
paired_excess.json and width_spread.json next to them, and sampler_health.json
from those records, reference_check.json and sequential.json.
"""

import json
from pathlib import Path

from common import ROOT

from chirpvi.rechecks import by_fraction, mode_counts, paired_excess, width_spread
from chirpvi.util import write_json

TOLERANCE = 0.11


def sources(folder: Path) -> dict:
    """Result file of each training seed: the file without a suffix is seed 0."""
    found = {"seed0": folder / "eval_reference.json"}
    for path in sorted(folder.glob("eval_reference_seed*.json")):
        found[path.stem.removeprefix("eval_reference_")] = path
    return found


def sampler_health(folder: Path) -> dict:
    """Largest stage count and smallest diversity or proposal ESS of the saved runs."""
    out = {}
    for path in sources(folder).values():
        records = json.loads(path.read_text())["records"]
        out[path.name] = dict(
            max_stages=max(r["ref_stages"] for r in records),
            min_distinct_share=min(r["ref_distinct"] for r in records),
        )
    check = folder / "reference_check.json"
    if check.exists():
        runs = json.loads(check.read_text())["two_run"]
        out[check.name] = dict(max_stages=max(r["stages"] for r in runs))
    path = folder / "sequential.json"
    if path.exists():
        rows = json.loads(path.read_text())["rows"]
        out[path.name] = dict(
            max_stages=max(r["stages"] for r in rows),
            min_proposal_ess=min(r["ess_first"] for r in rows),
        )
    return out


def recheck(folder: Path) -> list[str]:
    modes, paired, spread = {}, {}, {}
    for seed, path in sources(folder).items():
        groups = by_fraction(json.loads(path.read_text())["records"])
        counts = {f: mode_counts(rows) for f, rows in groups.items()}
        total = {
            k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))
        }
        modes[seed] = dict(source=path.name, fractions=counts, total=total)
        paired[seed] = dict(
            source=path.name,
            fractions={f: paired_excess(rows) for f, rows in groups.items()},
        )
        spread[seed] = dict(
            source=path.name,
            tolerance=TOLERANCE,
            fractions={f: width_spread(rows, TOLERANCE) for f, rows in groups.items()},
        )
    return [
        write_json(folder / "modes_concentration.json", modes),
        write_json(folder / "paired_excess.json", paired),
        write_json(folder / "width_spread.json", spread),
        write_json(folder / "sampler_health.json", sampler_health(folder)),
    ]


def main():
    for path in recheck(ROOT / "results"):
        print(path)


if __name__ == "__main__":
    main()
