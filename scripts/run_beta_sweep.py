#!/usr/bin/env python 
"""
Sweep beta_1 (per-parameter EMA) and beta_2 (per-module EMA) for
SAP-PEFT on TiroBERTa/TiDG and Afro-XLM-R/TiDG.

The two EMA-based importance signals controlled by these betas are
both live in this release: `ema_beta` (per-parameter smoothing in
`GradientImportanceEstimator.beta`) and `importance_smoothing`
(per-module smoothing, threaded through to
`GradientImportanceEstimator.ema_beta`). Running the sweep therefore
produces distinct results per value, as documented in the paper's
sensitivity table.

Usage (Windows):
    python scripts\\run_beta_sweep.py

Usage (selective):
    python scripts\\run_beta_sweep.py --sweeps beta1 --backbones tiroberta
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


BACKBONES = {
    "tiroberta": "configs/baselines/tiroberta_tidg_sap_peft.yaml",
    "afro_xlmr": "configs/baselines/afro_xlmr_tidg_sap_peft.yaml",
}

SWEEPS = {
    "beta1": {
        # peft.ema_beta -> PEFTConfig.ema_beta
        #              -> ProgressivePEFTConfig.ema_beta
        #              -> GradientImportanceEstimator.beta
        # (used in per-parameter EMA; after Patch 1, also in module score)
        "config_key": "peft.ema_beta",
        "values": [0.0, 0.5, 0.99],
    },
    "beta2": {
        # peft.importance_smoothing -> PEFTConfig.importance_smoothing
        #                           -> ProgressivePEFTConfig.importance_smoothing
        #                           -> GradientImportanceEstimator.ema_beta
        # (used in per-module EMA; requires Patch 2 to propagate)
        "config_key": "peft.importance_smoothing",
        "values": [0.9, 0.999],
    },
}


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def value_tag(value: float) -> str:
    """0.0 -> 000, 0.5 -> 050, 0.99 -> 099. Alphanumeric only."""
    return f"{int(round(value * 100)):03d}"


def apply_override(data: dict, key_path: str, value: float) -> None:
    """Write the override as a flat dotted key, matching the runner's
    expectation in baseline_config._run_experiment."""
    for method in data["baseline_methods"]:
        method["config_overrides"][key_path] = value


def tag_methods(data: dict, sweep_name: str, value: float) -> None:
    tag = value_tag(value)
    for method in data["baseline_methods"]:
        method["name"] = f"{method['name']}_{sweep_name}_{tag}"


def run_one(
    base_config: Path,
    backbone: str,
    sweep_name: str,
    value: float,
    output_root: Path,
    seeds: list[int],
) -> None:
    cfg = load_yaml(base_config)
    apply_override(cfg, SWEEPS[sweep_name]["config_key"], value)
    tag_methods(cfg, sweep_name, value)

    out_dir = output_root / backbone / sweep_name / value_tag(value)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_config = out_dir / "sweep_config.yaml"
    save_yaml(cfg, temp_config)

    cmd = [
        sys.executable, "scripts/run_baseline_comparison.py",
        "--config", str(temp_config),
        "--sequential",
        "--seeds", ",".join(str(s) for s in seeds),
        "--output-dir", str(out_dir / "results"),
    ]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=["beta1", "beta2"],
                    choices=list(SWEEPS))
    ap.add_argument("--backbones", nargs="+",
                    default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--output-root", default="results/sweeps")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    for backbone in args.backbones:
        base_config = Path(BACKBONES[backbone])
        if not base_config.exists():
            print(f"WARNING: {base_config} not found; skipping {backbone}")
            continue
        for sweep_name in args.sweeps:
            for value in SWEEPS[sweep_name]["values"]:
                print(f"\n===== {backbone} / {sweep_name} = {value} =====")
                run_one(base_config, backbone, sweep_name, value,
                        output_root, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())