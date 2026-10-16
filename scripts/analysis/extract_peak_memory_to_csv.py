#!/usr/bin/env python
"""
Extract peak GPU memory per method from raw_results.json files.

Usage:
    python scripts/analysis/extract_peak_memory_to_csv.py \
        --results-root results/baselines \
        --output peak_memory.csv

    # Or specify individual files:
    python scripts/analysis/extract_peak_memory_to_csv.py \
        --json results/baselines/tiroberta_tidg/<ts>/raw_results.json \
        --json results/baselines/afro_xlmr_tidg/<ts>/raw_results.json \
        --output peak_memory.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def get_method_display(method_key: str) -> str:
    mapping = [
        ("full_finetuning", "Full Fine-Tuning"),
        ("adalora", "AdaLoRA"),
        ("progressive_adaptive", "Progressive-Adaptive"),
        ("progressive_linear", "Progressive-Linear"),
        ("progressive_fixed", "Progressive-Fixed"),
        ("sap_peft", "SAP-PEFT"),
        ("dora", "DoRA"),
        ("lora", "LoRA"),
        ("vanilla", "Vanilla"),
    ]
    for key, display in mapping:
        if key in method_key.lower():
            return display
    return method_key


def get_model_from_path(path_str: str) -> str:
    low = path_str.lower()
    if "afro_xlmr" in low:
        return "Afro-XLM-R"
    if "tiroberta" in low:
        return "TiroBERTa"
    if "gpt2" in low:
        return "GPT-2"
    return "Unknown"


def get_task_from_path(path_str: str) -> str:
    low = path_str.lower()
    if "tidg" in low:
        return "news"
    if "sentiment" in low:
        return "sentiment"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Directory to discover raw_results.json under")
    ap.add_argument("--json", type=Path, action="append", default=[],
                    help="Explicit raw_results.json path (repeatable)")
    ap.add_argument("--output", type=Path, default=Path("peak_memory.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    json_paths = list(args.json)
    if args.results_root is not None:
        json_paths.extend(sorted(args.results_root.rglob("raw_results.json")))

    if not json_paths:
        print("No input paths. Pass --results-root or --json.")
        return

    seed_data = defaultdict(dict)
    for json_path in json_paths:
        path = Path(json_path)
        if not path.exists():
            print(f"Warning: file not found: {json_path}")
            continue
        model = get_model_from_path(str(path))
        task = get_task_from_path(str(path))
        if model == "Unknown" or task == "unknown":
            print(f"Skipping {path}: model/task not detected")
            continue
        with path.open() as f:
            data = json.load(f)
        for method_key, seeds in data.items():
            method = get_method_display(method_key)
            if method == method_key:
                continue
            for seed_entry in seeds:
                if not seed_entry.get("success", False):
                    continue
                seed = seed_entry.get("seed")
                if seed is None:
                    continue
                mem = seed_entry.get("peak_memory_gb")
                if mem is not None and mem > 0:
                    key = (model, method, task, seed)
                    seed_data.setdefault(key, mem)

    agg = defaultdict(list)
    for (model, method, task, seed), mem in seed_data.items():
        agg[(model, method, task)].append(mem)

    rows = []
    for (model, method, task), vals in agg.items():
        n = len(vals)
        mean = float(np.mean(vals))
        if n > 1:
            sem = stats.sem(vals)
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
            ci_width = float((ci[1] - ci[0]) / 2)
        else:
            ci_width = 0.0
        rows.append({
            "Model": model, "Task": task, "Method": method,
            "Mean_GB": round(mean, 3),
            "CI_Half_Width_GB": round(ci_width, 3),
            "N": n,
        })

    if not rows:
        print("No data extracted.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

    news_df = df[df["Task"] == "news"]
    print("\n--- News Task Peak Memory (mean ± 95% CI) ---")
    for model in ["TiroBERTa", "Afro-XLM-R", "GPT-2"]:
        print(f"\n{model}:")
        for method in ["Full Fine-Tuning", "LoRA", "DoRA", "AdaLoRA",
                       "Progressive-Adaptive", "Progressive-Linear",
                       "Progressive-Fixed", "SAP-PEFT", "Vanilla"]:
            subset = news_df[(news_df["Model"] == model) & (news_df["Method"] == method)]
            if not subset.empty:
                row = subset.iloc[0]
                print(f"  {method}: {row['Mean_GB']:.2f} ± "
                      f"{row['CI_Half_Width_GB']:.2f} (n={row['N']})")
            else:
                print(f"  {method}: data missing")


if __name__ == "__main__":
    main()