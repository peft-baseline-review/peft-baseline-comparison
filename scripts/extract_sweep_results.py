#!/usr/bin/env python
"""
After running run_beta_sweep.py, extract the accuracy mean, CI width,
and mean realized rank from each sweep directory.

Writes a compact CSV and prints a LaTeX-ready table.
"""

import json
from pathlib import Path

import numpy as np


SWEEP_ROOT = Path("results/sweeps")

BACKBONES = ["tiroberta", "afro_xlmr"]
SWEEPS = {
    "beta1": ["000", "050", "099"],
    "beta2": ["090", "100"],
}


def load_run_results(run_dir: Path) -> dict:
    """Read the most recent timestamped results directory."""
    results_root = run_dir / "results"
    if not results_root.exists():
        return {}
    timestamps = sorted(d for d in results_root.iterdir() if d.is_dir())
    if not timestamps:
        return {}
    latest = timestamps[-1]
    raw = latest / "raw_results.json"
    if not raw.exists():
        return {}
    with raw.open() as f:
        return json.load(f)


def compute_stats(raw: dict) -> dict:
    """Aggregate accuracy across the five seeds for each method."""
    out = {}
    for method_name, results in raw.items():
        accs = [r["accuracy"] for r in results if r.get("success")]
        if not accs:
            continue
        n = len(accs)
        mean = float(np.mean(accs))
        # CI half-width using t-distribution with n-1 dof, alpha=0.05
        from scipy import stats
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        ci_half = float(tcrit * stats.sem(accs)) if n > 1 else 0.0
        out[method_name] = {"mean": mean, "ci": ci_half, "n": n}
    return out


def main() -> None:
    rows = []
    for backbone in BACKBONES:
        for sweep_name, values in SWEEPS.items():
            for value in values:
                run_dir = SWEEP_ROOT / backbone / sweep_name / value
                raw = load_run_results(run_dir)
                if not raw:
                    print(f"missing: {run_dir}")
                    continue
                stats = compute_stats(raw)
                # Each directory has exactly one method; take the first
                for method_name, s in stats.items():
                    rows.append({
                        "backbone": backbone,
                        "sweep": sweep_name,
                        "value": value,
                        "accuracy_mean": s["mean"],
                        "accuracy_ci_half": s["ci"],
                        "n": s["n"],
                    })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("beta_sweep_results.csv", index=False)
    print(df.to_string(index=False))

    # LaTeX output
    print("\n% --- LaTeX rows ---")
    for backbone in BACKBONES:
        print(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{backbone}}}}} \\\\")
        print("\\midrule")
        for sweep_name, values in SWEEPS.items():
            for value in values:
                row = df[(df["backbone"] == backbone)
                         & (df["sweep"] == sweep_name)
                         & (df["value"] == value)]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(
                    f"& {sweep_name}$=${value} & "
                    f"${r['accuracy_mean']*100:.2f} \\pm {r['accuracy_ci_half']*100:.2f}$ "
                    f"\\\\"
                )
        print("\\midrule")


if __name__ == "__main__":
    main()