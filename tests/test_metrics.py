import math

import pytest
import torch

from chirpvi.metrics import (
    c2st,
    central_coverage,
    hpd_coverage,
    importance_efficiency,
    is_bimodal,
    kolmogorov_pvalue,
    mode_labels,
    mode_summary,
    pair_image,
    rank_histogram,
    rank_summary,
    sbc_ranks,
    tarp,
    tarp_summary,
    uniform_ks,
    widths,
)

PERIODIC = torch.tensor([False, False, True, False, True])


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_ranks_of_a_calibrated_posterior_are_uniform():
    g = gen(1)
    truth = torch.rand(600, 5, generator=g, dtype=torch.float64)
    samples = torch.rand(600, 99, 5, generator=g, dtype=torch.float64)
    ranks = sbc_ranks(truth, samples)
    assert ranks.shape == (600, 7) and ranks.max() <= 99
    for k in range(7):
        assert uniform_ks(ranks[:, k], 99)["p_value"] > 0.001


def test_ranks_of_an_overconfident_posterior_are_not_uniform():
    g = gen(2)
    truth = torch.rand(600, 5, generator=g, dtype=torch.float64)
    samples = 0.5 + 0.05 * (
        torch.rand(600, 99, 5, generator=g, dtype=torch.float64) - 0.5
    )
    ranks = sbc_ranks(truth, samples)
    assert uniform_ks(ranks[:, 0], 99)["p_value"] < 1e-6


def test_rank_histogram_counts_match_the_uniform_expectation_for_every_rank_once():
    out = rank_histogram(torch.arange(201), 200)
    assert sum(out["counts"]) == 201 and len(out["counts"]) == 20
    assert out["counts"] == [int(e) for e in out["expected"]]
    assert sorted(set(out["counts"])) == [10, 11]


def test_rank_histogram_puts_extreme_ranks_in_the_end_bins():
    out = rank_histogram(torch.tensor([0, 0, 200]), 200)
    assert out["counts"][0] == 2 and out["counts"][-1] == 1
    assert sum(out["expected"]) == pytest.approx(3)


def test_rank_summary_keeps_the_distance_and_adds_the_histogram():
    ranks = torch.arange(100)
    out = rank_summary(ranks, 99)
    assert out["ks"] == uniform_ks(ranks, 99)["ks"]
    assert sum(out["hist"]["counts"]) == 100


def test_kolmogorov_pvalue_limits():
    assert kolmogorov_pvalue(0.0, 100) == 1.0
    assert kolmogorov_pvalue(0.5, 1000) < 1e-10
    assert abs(kolmogorov_pvalue(1.358 / math.sqrt(1000), 1000) - 0.05) < 0.01


def test_hpd_coverage_matches_the_level_for_a_calibrated_gaussian():
    g = gen(3)
    truth = torch.randn(3000, generator=g, dtype=torch.float64)
    samples = torch.randn(3000, 200, generator=g, dtype=torch.float64)
    levels = (0.5, 0.68, 0.9, 0.95)
    cov = hpd_coverage(-0.5 * truth**2, -0.5 * samples**2, levels)
    assert all(abs(cov[c] - c) < 0.03 for c in levels)


def test_hpd_coverage_flags_an_overconfident_density():
    g = gen(11)
    truth = torch.randn(3000, generator=g, dtype=torch.float64)
    narrow = 0.4 * torch.randn(3000, 200, generator=g, dtype=torch.float64)
    low = hpd_coverage(
        -0.5 * (truth / 0.4) ** 2, -0.5 * (narrow / 0.4) ** 2, (0.5, 0.9)
    )
    assert low[0.5] < 0.4 and low[0.9] < 0.8


def test_central_coverage_of_calibrated_samples():
    g = gen(4)
    truth = torch.randn(2000, 3, generator=g, dtype=torch.float64)
    samples = torch.randn(2000, 200, 3, generator=g, dtype=torch.float64)
    cov = central_coverage(truth, samples, (0.68, 0.95))
    assert (cov[0.68] - 0.68).abs().max() < 0.04
    assert (cov[0.95] - 0.95).abs().max() < 0.03


def test_tarp_flags_overconfidence_and_passes_calibration():
    g = gen(5)
    truth = torch.rand(500, 5, generator=g, dtype=torch.float64)
    ref = torch.rand(500, 5, generator=g, dtype=torch.float64)
    good = torch.rand(500, 100, 5, generator=g, dtype=torch.float64)
    bad = 0.5 + 0.1 * (good - 0.5)
    assert tarp_summary(*tarp(truth, good, ref, PERIODIC))["max_deviation"] < 0.08
    summary = tarp_summary(*tarp(truth, bad, ref, PERIODIC))
    assert summary["max_deviation"] > 0.2 and summary["mean_deviation"] < 0


def test_c2st_separates_shifted_samples_and_not_identical_ones():
    g = gen(6)
    a = torch.rand(600, 5, generator=g, dtype=torch.float64)
    b = torch.rand(600, 5, generator=g, dtype=torch.float64)
    same = c2st(a, b, PERIODIC, gen(7), steps=150)
    assert 0.42 < same < 0.58
    shifted = b.clone()
    shifted[:, 0] = 0.3 * shifted[:, 0]
    assert c2st(a, shifted, PERIODIC, gen(8), steps=150) > 0.72


def two_mode_samples(n, centre=0.2, spread=0.01, weight=0.5, seed=0):
    g = gen(seed)
    u = torch.rand(n, 5, generator=g, dtype=torch.float64)
    second = torch.rand(n, generator=g, dtype=torch.float64) < weight
    u[:, 4] = (
        centre
        + 0.5 * second
        + spread * torch.randn(n, generator=g, dtype=torch.float64)
    ) % 1.0
    return u


def test_mode_labels_split_the_two_images_and_detect_a_lost_mode():
    both = mode_summary(two_mode_samples(2000))
    assert abs(both["weight"] - 0.5) < 0.05 and both["sky_std_0"] < 0.1
    assert is_bimodal(both)
    lost = mode_summary(two_mode_samples(2000, weight=0.0))
    assert abs(lost["weight"] - 0.5) > 0.4 and not is_bimodal(lost)
    labels = mode_labels(two_mode_samples(500, centre=0.97))
    assert 0.4 < labels.double().mean() < 0.6


def test_pair_image_is_an_involution_up_to_the_period():
    u = torch.rand(20, 5, generator=gen(9), dtype=torch.float64)
    twice = pair_image(pair_image(u))
    assert (twice - u).abs().max() < 1e-12
    assert (pair_image(u)[:, [0, 1, 3]] == u[:, [0, 1, 3]]).all()


def test_widths_and_importance_efficiency():
    g = gen(10)
    u = torch.rand(20000, 5, generator=g, dtype=torch.float64)
    assert all(abs(v - 1) < 0.03 for v in widths(u).values())
    narrow = 0.5 + 0.05 * (u - 0.5)
    assert widths(narrow)["chirp_mass"] < 0.1
    log_q = torch.randn(5000, generator=g, dtype=torch.float64)
    assert abs(importance_efficiency(log_q, log_q) - 1.0) < 1e-12
    assert importance_efficiency(5 * log_q, log_q) < 0.2
