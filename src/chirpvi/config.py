"""Constants of the toy problem."""

import math
from dataclasses import dataclass

T_SUN = 4.925491e-6
NAMES = ("chirp_mass", "coalescence_time", "phase", "cos_inclination", "sky_angle")
PERIODIC = (2, 4)
PERIODS = (2 * math.pi, math.pi)


@dataclass(frozen=True)
class Problem:
    """Grid, priors, antenna rotation and noise spectrum of the chirp problem."""

    fs: float = 128.0
    n: int = 256
    amplitude: float = 21.0
    omega: float = 0.5
    wall_hz: float = 25.0
    floor_hz: float = 3.0
    high_hz: float = 60.0
    mass: tuple = (15.0, 30.0)
    merger_margin: tuple = (0.2, 0.5)

    @property
    def duration(self) -> float:
        return self.n / self.fs

    def bounds(self) -> tuple[tuple, tuple]:
        """Lower and upper prior bounds of (Mc, tc, phic, cosi, alpha)."""
        end = self.duration
        lo = (self.mass[0], end + self.merger_margin[0], 0.0, -1.0, 0.0)
        hi = (self.mass[1], end + self.merger_margin[1], PERIODS[0], 1.0, PERIODS[1])
        return lo, hi


FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75, 1.0)


def n_obs_of(fraction: float, problem: Problem) -> int:
    """Number of observed samples at a fraction of the buffer."""
    return int(round(fraction * problem.n))
