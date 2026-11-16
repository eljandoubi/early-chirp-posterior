"""Seeds, circular helpers and JSON output shared by modules and scripts."""

import json
import math
import os
import platform
from pathlib import Path

import torch


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    """Wrap to [0, 1)."""
    return x - torch.floor(x)


def unit_distance(a: torch.Tensor, b: torch.Tensor, periodic: torch.Tensor):
    """Per-dimension distance in unit cube coordinates, circular where periodic."""
    d = (a - b).abs()
    return torch.where(periodic, torch.minimum(d, 1.0 - d), d)


def circular_mean(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular mean of unit coordinates (angle fractions), in [0, 1)."""
    ang = 2 * math.pi * x
    mean = torch.atan2(torch.sin(ang).mean(dim), torch.cos(ang).mean(dim))
    return wrap_unit(mean / (2 * math.pi))


def circular_std(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular standard deviation in radians of unit coordinates."""
    ang = 2 * math.pi * x
    r = torch.hypot(torch.sin(ang).mean(dim), torch.cos(ang).mean(dim))
    return torch.sqrt(-2 * torch.log(r.clamp_min(1e-12)))


def clean(x):
    """Make a result tree valid JSON: string keys, lists, no NaN or infinity."""
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path: str | os.PathLike, data: dict) -> str:
    """Write a result file with sorted keys and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=1, sort_keys=True) + "\n")
    return str(path)


def load_average() -> float:
    return os.getloadavg()[0]


def environment(device: str | torch.device) -> dict:
    """Python and torch versions, CPU count and device name of this run."""
    device = torch.device(device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cpu_count=os.cpu_count(),
        device=name,
    )
