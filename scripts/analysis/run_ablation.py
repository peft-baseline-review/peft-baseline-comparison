#!/usr/bin/env python
"""
Ablation Study Runner for Progressive PEFT.

Generates a series of ablation experiments by patching the base SAP-PEFT
YAML one hyperparameter at a time, then runs each variant with multiple
random seeds via the standard baseline comparison runner.

Each variant is written to a temporary YAML config first (using
make_ablation_config.py), so it can be inspected, rerun, or archived
independently of the sweep.

Usage:
    # Run every axis
    python scripts/analysis/run_ablation.py --config configs/baselines/tiroberta_tidg_sap_peft.yaml

    # Run a single axis
    python scripts/analysis/run_ablation.py \
        --config configs/baselines/tiroberta_tidg_sap_peft.yaml \
        --param temperature

    # Dry run (generate configs, do not launch training)
    python scripts/analysis/run_ablation.py \
        --config configs/baselines/tiroberta_tidg_sap_peft.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Ablation grids
# ---------------------------------------------------------------------------
# Each entry: (parameter_path, short_name, [values])
# The parameter_path uses flat dot notation and is applied as a
# config_overrides entry by make_ablation_config.py.
ABLATION_GRIDS = [
    ("peft.temperature",                  "temperature", [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    ("peft.allocation_update_frequency",  "update_freq", [50, 100, 200, 500]),
    ("peft.min_rank",                     "min_rank",    [1, 2, 4, 8]),
    ("peft.max_rank",                     "max_rank",    [16, 32, 64, 128]),
    ("peft.importance_smoothing",         "beta2",       [0.90, 0.95, 0.99]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_variant_config(
    base_config: Path,
    overrides: list[tuple[str, object]],
    name: str,
    out_path: Path,
) -> None:
    """Patch the base SAP-PEFT YAML and write a variant config."""
    with base_config.open("r") as f:
        cfg = yaml.safe_load(f)

    if "baseline_methods" not in cfg or not cfg["baseline_methods"]:
        raise SystemExit(f"No baseline_methods in {base_config}")

    method = cfg["baseline_methods"][0]
    method.setdefault("config_overrides", {})
    for key, val in overrides:
        method["config_overrides"][key] = val

    method["name"] = name
    method["description"] = f"SAP-PEFT ablation: {name}"
    method["experiment_name"] = f"SAP-PEFT ablation: {name}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  wrote {out_path}")


def run_experiment(config_path: Path, seeds: list[int], dry_run: bool) -> None:
    """Run the standard baseline runner on a single variant config."""
    seed_str = ",".join(str(s) for s in seeds)
    cmd = [
        sys.executable,
        "scripts/run_baseline_comparison.py",
        "--config", str(config_path),
        "--sequential",
        "--seeds", seed_str,
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED: {config_path}")
    else:
        print(f"  OK: {config_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run ablation sweeps for SAP-PEFT.")
    ap.add_argument(
        "--config", type=Path,
        default=Path("configs/baselines/tiroberta_tidg_sap_peft.yaml"),
        help="Base SAP-PEFT baseline YAML (default: TiroBERTa/TiDG)",
    )
    ap.add_argument(
        "--param", type=str, default=None,
        help="Run only this ablation axis (e.g. temperature). "
             "If omitted, all axes are run.",
    )
    ap.add_argument(
        "--seeds", type=str, default="42,43,44,45,46",
        help="Comma-separated random seeds.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("ablation_tmp_configs"),
        help="Directory for generated variant configs.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Generate configs but do not launch training.",
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if not args.config.exists():
        raise SystemExit(f"Base config not found: {args.config}")

    grids = ABLATION_GRIDS
    if args.param:
        grids = [g for g in grids if g[1] == args.param]
        if not grids:
            raise SystemExit(
                f"Unknown axis '{args.param}'. "
                f"Available: {[g[1] for g in ABLATION_GRIDS]}"
            )

    stem = args.config.stem  # e.g., "tiroberta_tidg_sap_peft"

    for param_path, param_name, values in grids:
        print(f"\n{'=' * 70}")
        print(f"Ablation axis: {param_name}  ({param_path})")
        print(f"Values: {values}")
        print(f"{'=' * 70}")

        for val in values:
            val_str = str(val).replace(".", "_")
            name = f"sap_peft_{param_name}_{val_str}"
            out = args.out_dir / f"{stem}__{param_name}_{val_str}.yaml"

            build_variant_config(
                base_config=args.config,
                overrides=[(param_path, val)],
                name=name,
                out_path=out,
            )
            run_experiment(out, seeds, dry_run=args.dry_run)

    print(f"\nDone. Generated configs are in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())