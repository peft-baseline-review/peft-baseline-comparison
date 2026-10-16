#!/usr/bin/env python
"""
Extract the sensitivity table from ablation runs.

Reads every raw_results.json produced by a SAP-PEFT ablation sweep
and emits a CSV and a LaTeX snippet for Table 9.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import csv


def parse_variant(name: str):
    """Extract (axis, value) from a method folder name like
    'sap_peft_temperature_3_0'."""
    if not name.startswith("sap_peft_"):
        return None
    parts = name[len("sap_peft_"):].rsplit("_", 1)
    if len(parts) != 2:
        return None
    axis, value = parts
    # The value may contain underscores for decimal points; convert
    value = value.replace("_", ".")
    try:
        value = float(value)
    except ValueError:
        return None
    return axis, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/baselines"))
    ap.add_argument("--pattern", type=str, default="sap_peft_temperature")
    ap.add_argument("--output", type=Path,
                    default=Path("sensitivity_temperature.csv"))
    args = ap.parse_args()

    rows = []
    for raw in args.results_dir.rglob("raw_results.json"):
        with raw.open() as f:
            data = json.load(f)
        for method_name, seeds in data.items():
            parsed = parse_variant(method_name)
            if parsed is None:
                continue
            axis, value = parsed
            if axis != args.pattern.replace("sap_peft_", ""):
                continue
            accs = [s["accuracy"] for s in seeds if s.get("success", False)]
            if len(accs) < 2:
                continue
            m = mean(accs)
            s = stdev(accs)
            # 95% CI half-width for n seeds
            n = len(accs)
            t_val = {5: 2.776, 4: 3.182, 3: 4.303}.get(n, 2.776)
            ci = t_val * s / (n ** 0.5)
            rows.append({
                "axis": axis,
                "value": value,
                "accuracy_mean": m,
                "ci_half_width": ci,
                "n_seeds": n,
            })

    rows.sort(key=lambda r: r["value"])

    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis","value","accuracy_mean","ci_half_width","n_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {args.output}")
    print()
    print("LaTeX snippet:")
    for r in rows:
        print(f"  {r['value']:.2f} & {r['accuracy_mean']*100:.2f} & {r['ci_half_width']*100:.2f} \\\\")


if __name__ == "__main__":
    raise SystemExit(main())