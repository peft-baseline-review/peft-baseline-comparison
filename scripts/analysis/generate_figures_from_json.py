#!/usr/bin/env python
"""
Generate publication-quality figures from statistical_analysis.json.

Usage:
    python scripts/analysis/generate_figures_from_json.py \
        --json results/baselines/<experiment>/<timestamp>/statistical_analysis.json \
        --output-dir figures/<experiment>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


# Map long method names to short, readable versions
NAME_MAP = {
    "tiroberta_tidg_lora": "LoRA",
    "tiroberta_tidg_full_finetuning": "Full FT",
    "tiroberta_tidg_progressive_adaptive": "Prog-Adapt",
    "tiroberta_tidg_sap_peft": "SAP-PEFT",
    "tiroberta_tidg_dora": "DoRA",
    "tiroberta_tidg_adalora": "AdaLoRA",
    "tiroberta_tidg_vanilla": "Vanilla",
    "tiroberta_tidg_progressive_linear": "Prog-Linear",
    "tiroberta_tidg_progressive_fixed": "Prog-Fixed",
    "afro_xlmr_lora": "LoRA",
    "afro_xlmr_full_finetuning": "Full FT",
    "afro_xlmr_progressive_adaptive": "Prog-Adapt",
    "afro_xlmr_sap_peft": "SAP-PEFT",
    "afro_xlmr_dora": "DoRA",
    "afro_xlmr_adalora": "AdaLoRA",
    "afro_xlmr_vanilla": "Vanilla",
    "afro_xlmr_progressive_linear": "Prog-Linear",
    "afro_xlmr_progressive_fixed": "Prog-Fixed",
    "gpt2_lora": "LoRA",
    "gpt2_full_finetuning": "Full FT",
    "gpt2_progressive_adaptive": "Prog-Adapt",
    "gpt2_sap_peft": "SAP-PEFT",
    "gpt2_dora": "DoRA",
    "gpt2_adalora": "AdaLoRA",
    "gpt2_vanilla": "Vanilla",
    "gpt2_progressive_linear": "Prog-Linear",
    "gpt2_progressive_fixed": "Prog-Fixed",
}

COLOR_MAP = {
    "LoRA": "#1f77b4",
    "Full FT": "#ff7f0e",
    "Prog-Adapt": "#2ca02c",
    "SAP-PEFT": "#d62728",
    "DoRA": "#9467bd",
    "AdaLoRA": "#8c564b",
    "Vanilla": "#e377c2",
    "Prog-Linear": "#7f7f7f",
    "Prog-Fixed": "#bcbd22",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
})


def short_name(name):
    return NAME_MAP.get(name, name)


def get_color(name):
    return COLOR_MAP.get(short_name(name), "#cccccc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True,
                    help="Path to statistical_analysis.json")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory for output figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.json.open() as f:
        data = json.load(f)

    perf = data["performance_summary"]
    pareto = data.get("pareto_frontier")
    pairwise = data.get("pairwise_tests", {})
    raw = data["pareto_frontier"]["raw_objectives"]

    methods = list(perf.keys())
    short_names = [short_name(m) for m in methods]
    params_m = [perf[m]["params_millions"] for m in methods]
    acc_percent = [perf[m]["accuracy_mean"] * 100 for m in methods]
    energy_j = [perf[m]["energy_mean_j"] for m in methods]
    lat_mean = [perf[m]["latency_mean_ms"] for m in methods]
    lat_ci = [perf[m]["latency_ci_width"] for m in methods]

    out = args.output_dir

    # 1. Efficiency scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    handles, labels = [], []
    for m, x, y, s in zip(methods, params_m, acc_percent, short_names):
        sc = ax.scatter(x, y, s=80, color=get_color(m), edgecolor="black", alpha=0.8)
        handles.append(sc)
        labels.append(s)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    unique = {}
    for h, l in zip(handles, labels):
        unique.setdefault(l, h)
    ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out / "efficiency_scatter.pdf", bbox_inches="tight")
    plt.savefig(out / "efficiency_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Energy scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    handles, labels = [], []
    for m, x, y, s in zip(methods, energy_j, acc_percent, short_names):
        sc = ax.scatter(x, y, s=80, color=get_color(m), edgecolor="black", alpha=0.8)
        handles.append(sc)
        labels.append(s)
    ax.set_xlabel("Energy per Sample (J)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    unique = {}
    for h, l in zip(handles, labels):
        unique.setdefault(l, h)
    ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out / "energy_scatter.pdf", bbox_inches="tight")
    plt.savefig(out / "energy_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Latency bar
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, lat_mean, yerr=lat_ci, capsize=5,
           color=[get_color(m) for m in methods],
           error_kw={"linewidth": 1.5, "ecolor": "black"})
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("Inference Latency (ms)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "latency_barplot.pdf", bbox_inches="tight")
    plt.savefig(out / "latency_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Pareto parallel coordinates
    if pareto:
        norm_obj = pareto["normalized_objectives"]
        frontier_orig = pareto["frontier_methods"]
        frontier_short = [short_name(m) for m in frontier_orig]

        methods_p = list(norm_obj.keys())
        short_names_p = [short_name(m) for m in methods_p]
        values = np.array([[norm_obj[m]["accuracy"],
                            norm_obj[m]["param_efficiency"],
                            norm_obj[m]["speed"],
                            norm_obj[m]["memory_efficiency"]] for m in methods_p])
        is_frontier = [short_name(m) in frontier_short for m in methods_p]
        xp = np.arange(4)
        x_labels = ["Accuracy", "Parameter\nEfficiency", "Speed", "Memory\nEfficiency"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for m, vals, frontier in zip(short_names_p, values, is_frontier):
            if not frontier:
                ax.plot(xp, vals, color="lightgray", linewidth=1.0, alpha=0.3)
        legend_handles = []
        for m, vals, frontier in zip(short_names_p, values, is_frontier):
            if frontier:
                line = ax.plot(xp, vals, color=get_color(m), linewidth=2.5,
                               alpha=0.9, label=m)
                legend_handles.append(line[0])
        ax.legend(handles=legend_handles, loc="upper right",
                  frameon=True, title="Pareto-optimal")
        ax.set_xticks(xp)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Normalized Value (higher is better)")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / "pareto_parallel.pdf", bbox_inches="tight")
        plt.savefig(out / "pareto_parallel.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 5. P-value heatmap
    wilcox = pairwise.get("wilcoxon", [])
    mcnemar = pairwise.get("mcnemar", [])
    if wilcox or mcnemar:
        n = len(short_names)
        pval = np.ones((n, n))
        for res in wilcox:
            i1 = short_names.index(short_name(res["method1"]))
            i2 = short_names.index(short_name(res["method2"]))
            p = res.get("p_value_corrected", 1.0)
            pval[i1, i2] = pval[i2, i1] = p
        for res in mcnemar:
            i1 = short_names.index(short_name(res["method1"]))
            i2 = short_names.index(short_name(res["method2"]))
            p = res.get("p_value_corrected", 1.0)
            pval[i1, i2] = pval[i2, i1] = p
        np.fill_diagonal(pval, np.nan)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pval, annot=True, fmt=".2e", cmap="Reds_r",
                    square=True, cbar_kws={"label": "Corrected p-value"},
                    xticklabels=short_names, yticklabels=short_names, ax=ax)
        ax.set_title("Pairwise Statistical Significance (Holm-corrected p-values)")
        plt.tight_layout()
        plt.savefig(out / "pvalue_heatmap.pdf", bbox_inches="tight")
        plt.savefig(out / "pvalue_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 6. Memory bar plot
    peak_memory_gb = []
    for m in methods:
        mem_eff = raw[m]["memory_efficiency"]
        peak_memory_gb.append((1.0 / mem_eff) / 1024.0)
    order = np.argsort(peak_memory_gb)
    peak_memory_gb = [peak_memory_gb[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(short_names))
    ax.bar(x, peak_memory_gb, color="#4c72b0", edgecolor="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Peak Memory Usage")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "memory_barplot.pdf", bbox_inches="tight")
    plt.savefig(out / "memory_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()