#!/usr/bin/env python
"""After all regens, print a clean summary for the paper."""
import json
from pathlib import Path

EXPERIMENTS = [
    ("tiroberta_tidg",      "2026-03-10_00-26-37"),
    ("tiroberta_sentiment", "2026-03-10_13-48-15"),
    ("afro_xlmr_tidg",      "2026-03-14_00-56-04"),
    ("afro_xlmr_sentiment", "2026-03-01_12-49-29"),
    ("gpt2_tidg",           "2026-03-14_13-57-52"),
    ("gpt2_sentiment",      "2026-03-05_07-51-48"),
]

print(f"{'experiment':25s} {'HV (new)':>10s} {'HV (old)':>10s} {'delta':>10s}  frontier members")
print("-" * 100)

hvs = []
for exp, ts in EXPERIMENTS:
    run = Path(f"results/baselines/{exp}/{ts}")
    new = json.load((run / "statistical_analysis.json").open())
    old_path = run / "statistical_analysis.json.bak"
    if old_path.exists():
        old = json.load(old_path.open())
        old_hv = old["pareto_frontier"]["hypervolume"]
    else:
        old_hv = float("nan")

    hv = new["pareto_frontier"]["hypervolume"]
    frontier = new["pareto_frontier"]["frontier_methods"]
    hvs.append((exp, hv))

    delta = hv - old_hv
    print(f"{exp:25s} {hv:>10.5f} {old_hv:>10.5f} {delta:>+10.5f}  {len(frontier)} methods: {frontier}")

print()
print("=" * 100)
print("KEY NUMBERS FOR §5.7:")
print("=" * 100)
min_exp, min_hv = min(hvs, key=lambda x: x[1])
max_exp, max_hv = max(hvs, key=lambda x: x[1])
spread = max_hv / min_hv
import math
print(f"  Minimum HV: {min_hv:.5f}  ({min_exp})")
print(f"  Maximum HV: {max_hv:.5f}  ({max_exp})")
print(f"  Spread:     {spread:.0f}×  ({math.log10(spread):.2f} orders of magnitude)")
print()
print("  News experiments only (TiDG):")
for exp, hv in hvs:
    if "tidg" in exp:
        print(f"    {exp}: {hv:.5f}")