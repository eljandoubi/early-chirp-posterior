"""Numbers derived from the per-case records of the reference comparison."""

import statistics

from .metrics import is_bimodal

WIDTH_NAMES = ("chirp_mass", "coalescence_time", "cos_inclination")


def by_fraction(records) -> dict:
    """Records grouped by observed fraction, keyed like the result tables."""
    groups = {}
    for r in records:
        groups.setdefault(f"{r['fraction']:g}", []).append(r)
    return groups


def mode_counts(rows) -> dict:
    """The cases whose reference is bimodal and how many of them the flow passes.

    weight_near_half: the flow puts between 0.3 and 0.7 of its weight on each
    half-circle of the sky angle (a uniform flow passes). concentrated: the sky
    angle spread inside each half stays under 0.3 rad. passes_reference_test: the
    weight is between 0.4 and 0.6 and both halves are concentrated, the test that
    makes the reference bimodal.
    """
    flows = [r["mode_flow"] for r in rows if is_bimodal(r["mode_ref"])]
    return dict(
        bimodal_cases=len(flows),
        weight_near_half=sum(abs(m["weight"] - 0.5) < 0.2 for m in flows),
        concentrated=sum(is_bimodal(m, 0.0, 1.0) for m in flows),
        passes_reference_test=sum(is_bimodal(m) for m in flows),
    )


def paired_excess(rows) -> dict:
    """Flow C2ST minus the reference floor on the cases that have both values."""
    pairs = [
        (r["c2st"], r["c2st_floor"]) for r in rows if r.get("c2st_floor") is not None
    ]
    gaps = [flow - floor for flow, floor in pairs]
    return dict(
        cases=len(pairs),
        median_c2st=statistics.median(flow for flow, _ in pairs) if pairs else None,
        median_floor=statistics.median(floor for _, floor in pairs) if pairs else None,
        median_excess=statistics.median(gaps) if gaps else None,
        min_excess=min(gaps) if gaps else None,
        max_excess=max(gaps) if gaps else None,
    )


def width_spread(rows, tolerance: float = 0.11) -> dict:
    """Per-injection flow over reference width ratios: median, range, outliers."""
    out = {}
    for name in WIDTH_NAMES:
        ratios = [
            r["widths_flow"][name] / r["widths_ref"][name]
            for r in rows
            if r["widths_ref"][name] > 0
        ]
        out[name] = dict(
            cases=len(ratios),
            median=statistics.median(ratios) if ratios else None,
            min=min(ratios) if ratios else None,
            max=max(ratios) if ratios else None,
            outside=sum(abs(x - 1) > tolerance for x in ratios),
        )
    return out
