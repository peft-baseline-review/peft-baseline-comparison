#!/usr/bin/env python
"""
Generate a temporary ablation config by patching a base SAP-PEFT YAML.

Usage:
    python scripts/analysis/make_ablation_config.py \
        --base configs/baselines/tiroberta_tidg_sap_peft.yaml \
        --override peft.temperature=3.0 \
        --name sap_peft_temperature_3_0 \
        --out ablation_tmp_configs/tiroberta_T_3_0.yaml
"""

from __future__ import annotations
import argparse
import yaml
from pathlib import Path


def parse_override(s: str):
    """Parse key=value where value is parsed as YAML scalar."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected key=value, got {s!r}")
    key, val = s.split("=", 1)
    try:
        val_parsed = yaml.safe_load(val)
    except Exception:
        val_parsed = val
    return key, val_parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="Path to base SAP-PEFT baseline YAML")
    ap.add_argument("--override", action="append", default=[],
                    type=parse_override,
                    help="Override as key=value (repeatable). Dotted keys allowed.")
    ap.add_argument("--name", required=True, type=str,
                    help="Name for the ablation method")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output YAML path")
    args = ap.parse_args()

    with args.base.open("r") as f:
        cfg = yaml.safe_load(f)

    if "baseline_methods" not in cfg or not cfg["baseline_methods"]:
        raise SystemExit(f"No baseline_methods in {args.base}")

    method = cfg["baseline_methods"][0]
    method.setdefault("config_overrides", {})

    for key, val in args.override:
        method["config_overrides"][key] = val

    method["name"] = args.name
    method["description"] = f"SAP-PEFT ablation: {args.name}"
    method["experiment_name"] = f"SAP-PEFT ablation: {args.name}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()