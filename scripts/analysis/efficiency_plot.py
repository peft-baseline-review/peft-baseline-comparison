#!/usr/bin/env python
"""
Single-panel efficiency scatter (trainable parameters vs. accuracy).

Usage:
    python scripts/analysis/efficiency_plot.py \
        --json results/baselines/<experiment>/<timestamp>/statistical_analysis.json \
        --output figures/efficiency_scatter \
        --title "Efficiency vs. Performance"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True,
                    help="Path to statistical_analysis.json")
    ap.add_argument("--output", type=Path, default=Path("efficiency_scatter"),
                    help="Output basename (no extension); writes .png and .pdf")
    ap.add_argument("--title", type=str,
                    default="Efficiency vs. Performance",
                    help="Plot title")
    args = ap.parse_args()

    with args.json.open() as f:
        data = json.load(f)

    perf = data["performance_summary"]
    pareto_methods = set(data["pareto_frontier"]["frontier_methods"])

    methods, params_m, accuracy = [], [], []
    for method, stats in perf.items():
        methods.append(method)
        params_m.append(stats["params_millions"])
        accuracy.append(stats["accuracy_mean"] * 100)

    colors = list(plt.cm.tab10(np.linspace(0, 1, len(methods))))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (m, x, y) in enumerate(zip(methods, params_m, accuracy)):
        if m in pareto_methods:
            marker, edgecolor, lw, size = "o", "black", 1.5, 100
        else:
            marker, edgecolor, lw, size = "s", colors[i], 1, 80
        ax.scatter(x, y, marker=marker, s=size, color=colors[i],
                   edgecolor=edgecolor, linewidth=lw, alpha=0.8, label=m)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", ncol=2, fontsize=8, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()