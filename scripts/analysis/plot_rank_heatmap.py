#!/usr/bin/env python
"""
Per-layer rank evolution heatmaps for the progressive variants.

Discovers report JSON files produced by SAP-PEFT and Progressive-Adaptive /
Linear / Fixed runs, then writes one heatmap per run.

Usage:
    python scripts/analysis/plot_rank_heatmap.py \
        --experiments-root experiments \
        --output-dir figures/rank_heatmaps
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_rank_heatmap(rank_history, title, save_path,
                      agg_func="max", vmin=2, vmax=32):
    """Generate a heatmap of rank evolution over epochs (rows=layers,
    columns=epochs)."""
    if not rank_history:
        print(f"Warning: Empty rank_history for {title}")
        return

    # 1. Discover every layer that appears in any epoch
    all_layers = set()
    for entry in rank_history:
        for module in entry["allocation"]:
            m = re.search(r"layer\.(\d+)", module)
            if m:
                all_layers.add(int(m.group(1)))
    layers = sorted(all_layers)

    if not layers:
        print(f"Warning: no 'layer.N' modules found in rank_history for {title}")
        return

    # 2. Build (num_layers x num_epochs) matrix, one column per epoch
    epochs = [entry["epoch"] for entry in rank_history]
    matrix = np.full((len(layers), len(epochs)), np.nan, dtype=float)

    for col, entry in enumerate(rank_history):
        per_layer = {}
        for module, rank in entry["allocation"].items():
            m = re.search(r"layer\.(\d+)", module)
            if not m:
                continue
            layer = int(m.group(1))
            per_layer.setdefault(layer, []).append(rank)

        for layer, ranks in per_layer.items():
            row = layers.index(layer)
            if agg_func == "max":
                matrix[row, col] = max(ranks)
            elif agg_func == "mean":
                matrix[row, col] = float(np.mean(ranks))
            elif agg_func == "median":
                matrix[row, col] = float(np.median(ranks))
            else:
                raise ValueError("agg_func must be 'max', 'mean', or 'median'")

    # 3. Plot
    _, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                   interpolation="nearest",
                   vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer Index (0 = bottom)")
    ax.set_title(title)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    plt.colorbar(im, ax=ax, label="Rank")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path} ({matrix.shape[0]} layers x {matrix.shape[1]} epochs)")


def infer_title(report_path: Path) -> str:
    """Derive a human-readable title from the report file path."""
    parts = [p.lower() for p in report_path.parts]
    text = " ".join(parts)
    if "sap" in text:
        return "SAP-PEFT (structure-adaptive)"
    if "progressive adaptive" in text or "progressive_adaptive" in text:
        return "Progressive-Adaptive (budget-based)"
    if "progressive linear" in text or "progressive_linear" in text:
        return "Progressive-Linear (budget-based)"
    if "progressive fixed" in text or "progressive_fixed" in text:
        return "Progressive-Fixed (budget-based)"
    return report_path.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-root", type=Path, default=Path("experiments"),
                    help="Root of the experiments/ tree")
    ap.add_argument("--output-dir", type=Path, default=Path("figures/rank_heatmaps"),
                    help="Directory for output heatmaps")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_files = sorted(args.experiments_root.rglob("*_report.json"))
    if not report_files:
        print(f"No report JSONs found under {args.experiments_root}")
        return

    for report_path in report_files:
        try:
            with report_path.open() as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {report_path}: {e}")
            continue

        rank_history = data.get("rank_history", [])
        if not rank_history:
            continue

        title = infer_title(report_path)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
        out = args.output_dir / f"rank_{safe}.pdf"
        plot_rank_heatmap(rank_history, title=title, save_path=out,
                          agg_func="max", vmin=2, vmax=32)


if __name__ == "__main__":
    main()