#!/usr/bin/env python
"""
Validate the NVML-based energy meter against its theoretical
FLOP-based estimate, on a small sample batch.

The script only needs a *base* experiment config (the kind under
configs/experiments/), because it builds a model and runs a few
forward passes. Baseline configs (configs/baselines/) are not
accepted here.

Usage:
    python scripts/analysis/calibration.py \
        --config configs/experiments/tiroberta_tidg_base.yaml \
        --seed 42 \
        --num-runs 100
"""

import sys
from pathlib import Path

# Make the repository root importable when this script is launched directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch

from config.config_loader import load_config
from config.experiment import ExperimentConfig
from src.data.dataloader import TigrinyaDataLoader
from src.models.peft_factory import PEFTModelFactory
from src.evaluation.energy_meter import EnergyMeter


def load_model_and_batch(config_path: str, seed: int = 42):
    """Load a base experiment config, build the model, and grab one batch."""
    config_path = Path(config_path).resolve()
    config_dict = load_config(config_path, base_path=config_path.parent)
    config_dict["seed"] = seed
    config = ExperimentConfig(**config_dict)

    # Data loader
    data_loader = TigrinyaDataLoader(config, use_cache=True)
    _, _, test_loader = data_loader.create_dataloaders()
    sample_batch = next(iter(test_loader))

    device = torch.device(config.device)
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

    # Model
    model = PEFTModelFactory.create_model(
        model_config=config.model,
        peft_config=config.peft,
        num_labels=config.task.num_labels,
        device=device,
    )
    model.eval()

    return model, sample_batch, device

def calibrate_energy_meter(model, sample_batch, device, num_runs=100):
    meter = EnergyMeter(model, device, enable_power_measurement=True)

    theo = meter.measure_energy(sample_batch, num_runs=num_runs, measure_power=False)
    real = meter.measure_energy(sample_batch, num_runs=num_runs, measure_power=True)

    if real is not None:
        factor = real["energy_per_inference_joules"] / theo["energy_per_inference_joules"]
        print(f"Calibration factor (measured/theoretical): {factor:.3f}")
        print(f"  Theoretical: {theo['energy_per_inference_joules']:.6f} J")
        print(f"  Measured:    {real['energy_per_inference_joules']:.6f} J")
        print(f"  Measurement method: {real['measurement_method']}")
        return factor
    else:
        print("Real measurement failed. Check NVML permissions or hardware.")
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibrate the energy meter.")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/experiments/tiroberta_tidg_base.yaml",
        help="Path to a *base* experiment config (under configs/experiments/).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-runs", type=int, default=100)
    args = ap.parse_args()

    model, batch, device = load_model_and_batch(args.config, seed=args.seed)
    calibrate_energy_meter(model, batch, device, num_runs=args.num_runs)