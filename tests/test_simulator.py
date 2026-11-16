import math

import pytest
import torch

from chirpvi.config import Problem
from chirpvi.simulator import Simulator


def corner_thetas(sim):
    lo, hi = sim.lo, sim.hi
    rows = []
    for mc in (lo[0], hi[0]):
        for tc in (lo[1], hi[1]):
            rows.append(torch.stack([mc, tc, lo[2] + 1.0, lo[3] + 0.3, lo[4] + 0.4]))
    return torch.stack(rows)


def test_observations_are_reproducible_and_causal():
    sim = Simulator()
    theta = sim.sample_prior(4, torch.Generator().manual_seed(0))
    full = sim.observe(theta, torch.full((4,), 256), torch.Generator().manual_seed(5))
    cut = sim.observe(theta, torch.full((4,), 100), torch.Generator().manual_seed(5))
    again = sim.observe(theta, torch.full((4,), 100), torch.Generator().manual_seed(5))
    assert torch.equal(cut, again)
    assert torch.equal(cut[..., :100], full[..., :100])
    assert cut[..., 100:].abs().max() == 0


def test_pair_symmetry_leaves_both_channels_unchanged():
    sim = Simulator()
    theta = sim.sample_prior(8, torch.Generator().manual_seed(1))
    image = theta.clone()
    image[:, 2] = (image[:, 2] + math.pi) % (2 * math.pi)
    image[:, 4] = (image[:, 4] + math.pi / 2) % math.pi
    assert (sim.signal(theta) - sim.signal(image)).abs().max() < 1e-9
    assert (sim.pair_image(theta) - image).abs().max() < 1e-12
    twice = sim.pair_image(sim.pair_image(theta))
    assert (twice - theta).abs().max() < 1e-12


def test_phase_derivative_matches_the_frequency_law():
    sim = Simulator()
    theta = corner_thetas(sim)
    phase, _ = sim.phase_and_amplitude(theta)
    dphi = (phase[:, 1:] - phase[:, :-1]) * sim.problem.fs
    freq = sim.frequency(theta)
    mid = 0.5 * (freq[:, 1:] + freq[:, :-1])
    assert ((dphi / (2 * math.pi)) / mid - 1).abs().max() < 2e-3


def test_frequency_stays_below_nyquist_over_the_prior():
    sim = Simulator()
    theta = corner_thetas(sim)
    assert sim.frequency(theta).max() < 0.85 * sim.problem.fs / 2


def test_face_on_source_has_constant_amplitude():
    sim = Simulator()
    theta = sim.sample_prior(3, torch.Generator().manual_seed(2))
    theta[:, 3] = 1.0
    sig = sim.signal(theta)
    _, amp = sim.phase_and_amplitude(theta)
    assert (sig.pow(2).sum(1).sqrt() - amp).abs().max() < 1e-9


def test_signal_is_linear_in_the_amplitude_constant():
    base = Simulator(Problem(amplitude=1.0))
    big = Simulator(Problem(amplitude=3.0))
    theta = base.sample_prior(2, torch.Generator().manual_seed(3))
    assert (big.signal(theta) - 3 * base.signal(theta)).abs().max() < 1e-9


def test_amplitude_is_the_chirp_mass_to_the_five_fourths():
    sim = Simulator()
    theta = torch.tensor(
        [[15.0, 2.4, 0.0, 0.0, 0.0], [30.0, 2.4, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    _, amp = sim.phase_and_amplitude(theta)
    assert torch.allclose(amp[1] / amp[0], torch.full_like(amp[0], 2**1.25))
    assert amp[0, 0].item() == pytest.approx(21.0 * (15 / 20) ** 1.25 * 2.4**-0.25)


def test_amplitude_is_the_time_to_merger_to_the_minus_one_quarter():
    sim = Simulator()
    theta = torch.tensor([[20.0, 2.4, 0.0, 0.0, 0.0]], dtype=torch.float64)
    _, amp = sim.phase_and_amplitude(theta)
    tau = 2.4 - sim.t
    ratio = (tau[0] / tau[200]) ** 0.25
    assert amp[0, 200].item() / amp[0, 0].item() == pytest.approx(ratio.item())


def test_polarisation_factors_of_face_on_edge_on_and_tilted_orbits():
    sim = Simulator()
    for cosi in (-1.0, -0.5, 0.0, 0.5, 1.0):
        theta = torch.tensor([[20.0, 2.4, 0.7, cosi, 0.0]], dtype=torch.float64)
        phase, amp = sim.phase_and_amplitude(theta)
        a, phi = amp[0, 0].item(), phase[0, 0].item()
        plus = a * 0.5 * (1.0 + cosi * cosi) * math.cos(phi)
        cross = a * cosi * math.sin(phi)
        # alpha = 0 and t = 0: channel 1 is plus and channel 2 is cross
        sig = sim.signal(theta)
        assert sig[0, 0, 0].item() == pytest.approx(plus, abs=1e-12)
        assert sig[0, 1, 0].item() == pytest.approx(cross, abs=1e-12)


def test_snr_grows_with_the_observed_fraction():
    sim = Simulator()
    theta = sim.sample_prior(16, torch.Generator().manual_seed(4))
    snr = [sim.snr(theta, torch.full((16,), n)) for n in (64, 128, 256)]
    assert (snr[1] >= snr[0]).all() and (snr[2] >= snr[1]).all()
    assert (snr[2] > snr[0]).all()


def test_prior_samples_stay_inside_the_bounds():
    sim = Simulator()
    theta = sim.sample_prior(500, torch.Generator().manual_seed(6))
    assert (theta >= sim.lo).all() and (theta <= sim.hi).all()
    assert torch.allclose(sim.from_unit(sim.to_unit(theta)), theta)
