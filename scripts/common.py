"""Shared command line handling for the scripts."""

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny CPU run")
    mode.add_argument("--full", action="store_true", help="full sizes, writes results/")
    default = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default)
    p.add_argument("--seed", type=int, default=0)
    return p


def results_dir(args) -> Path:
    return ROOT / "results" / "smoke" if args.smoke else ROOT / "results"


def checkpoint_path(args) -> Path:
    name = "flow_smoke.pt" if args.smoke else f"flow_seed{args.seed}.pt"
    path = ROOT / "checkpoints" / name
    path.parent.mkdir(exist_ok=True)
    return path


def size(args, presets: dict) -> dict:
    return presets["smoke" if args.smoke else "full"]
