#!/usr/bin/env python3
"""
extract_training_time_carbon.py

Reads raw_results.json files from baseline runner output, computes mean and std
of training_time_hours across seeds, generates multi-panel bar plots for
training time (GPU hours) and estimated carbon emissions (kg CO₂).
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
GPU_POWER_KW = 0.15
CARBON_INTENSITY = 0.233  # kg CO₂ per kWh

# Model name mapping
MODEL_MAP = {
    "afro_xlmr": "Afro-XLM-R",
    "tiroberta": "TiroBERTa",
    "gpt2": "GPT-2",
}

# Task name mapping
TASK_MAP = {
    "tidg": "News",
    "sentiment": "Sentiment",
}

# Method display name mapping
METHOD_MAP = {
    "afro_xlmr_lora": "LoRA",
    "afro_xlmr_full_finetuning": "Full Fine-Tuning",
    "afro_xlmr_vanilla": "Vanilla",
    "afro_xlmr_dora": "DoRA",
    "afro_xlmr_adalora": "AdaLoRA",
    "afro_xlmr_progressive_adaptive": "Progressive-Adaptive",
    "afro_xlmr_progressive_linear": "Progressive-Linear",
    "afro_xlmr_progressive_fixed": "Progressive-Fixed",
    "afro_xlmr_sap_peft": "SAP-PEFT",
    "gpt2_lora": "LoRA",
    "gpt2_full_finetuning": "Full Fine-Tuning",
    "gpt2_vanilla": "Vanilla",
    "gpt2_dora": "DoRA",
    "gpt2_adalora": "AdaLoRA",
    "gpt2_progressive_adaptive": "Progressive-Adaptive",
    "gpt2_progressive_linear": "Progressive-Linear",
    "gpt2_progressive_fixed": "Progressive-Fixed",
    "gpt2_sap_peft": "SAP-PEFT",
    "tiroberta_tidg_lora": "LoRA",
    "tiroberta_tidg_full_finetuning": "Full Fine-Tuning",
    "tiroberta_tidg_vanilla": "Vanilla",
    "tiroberta_tidg_dora": "DoRA",
    "tiroberta_tidg_adalora": "AdaLoRA",
    "tiroberta_tidg_progressive_adaptive": "Progressive-Adaptive",
    "tiroberta_tidg_progressive_linear": "Progressive-Linear",
    "tiroberta_tidg_progressive_fixed": "Progressive-Fixed",
    "tiroberta_tidg_sap_peft": "SAP-PEFT",
    "tiroberta_sentiment_lora": "LoRA",
    "tiroberta_sentiment_full_finetuning": "Full Fine-Tuning",
    "tiroberta_sentiment_vanilla": "Vanilla",
    "tiroberta_sentiment_dora": "DoRA",
    "tiroberta_sentiment_adalora": "AdaLoRA",
    "tiroberta_sentiment_progressive_adaptive": "Progressive-Adaptive",
    "tiroberta_sentiment_progressive_linear": "Progressive-Linear",
    "tiroberta_sentiment_progressive_fixed": "Progressive-Fixed",
    "tiroberta_sentiment_sap_peft": "SAP-PEFT"
}

# Order of methods for plotting
METHOD_ORDER = [
    "Full Fine-Tuning",
    "LoRA",
    "DoRA",
    "AdaLoRA",
    "Progressive-Adaptive",
    "Progressive-Linear",
    "Progressive-Fixed",
    "SAP-PEFT",
    "Vanilla",
]

def find_raw_results_files(root: Path):
    """Return list of all raw_results.json files in the directory tree."""
    print(list(root.rglob("raw_results.json")))
    return list(root.rglob("raw_results.json"))

def extract_data(json_path: Path):
    """Extract training_time_hours per method from a raw_results.json file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Determine model and task from the directory name (e.g., "afro_xlmr_tidg" or "gpt2_sentiment")
    model_task = json_path.parent.parent.name  # e.g., "afro_xlmr_tidg"
    parts = model_task.split("_")
    
    # Identify model (first part or known pattern)
    if parts[0] == "afro":
        model = "Afro-XLM-R"
    elif parts[0] == "tiroberta":
        model = "TiroBERTa"
    elif parts[0] == "gpt2":
        model = "GPT-2"
    else:
        model = parts[0].capitalize()
    
    # Identify task (last part, usually "tidg" or "sentiment")
    task_raw = parts[-1].lower()
    if task_raw == "tidg":
        task = "News"
    elif task_raw == "sentiment":
        task = "Sentiment"
    else:
        task = task_raw.capitalize()
    
    results = []
    for method_key, seeds in data.items():
        method_name = METHOD_MAP.get(method_key, method_key)
        # Optionally filter out mismatched tasks (if method_key contains "sentiment" but task is News)
        if "sentiment" in method_key.lower() and task == "News":
            continue
        if "tidg" in method_key.lower() and task == "Sentiment":
            continue
        for seed_entry in seeds:
            training_time = seed_entry.get("training_time_hours", None)
            if training_time is not None:
                results.append({
                    "model": model,
                    "task": task,
                    "method": method_name,
                    "training_time_hours": training_time,
                    "carbon_kg": training_time * GPU_POWER_KW * CARBON_INTENSITY
                })
    return results
def aggregate_and_plot(results, output_dir):
    """Group results by (model, task, method) and plot bar charts."""
    grouped = defaultdict(list)
    for entry in results:
        key = (entry["model"], entry["task"], entry["method"])
        grouped[key].append(entry["training_time_hours"])
    # Compute means and stds
    stats = defaultdict(dict)
    for (model, task, method), times in grouped.items():
        stats[(model, task, method)] = {
            "mean": np.mean(times),
            "std": np.std(times, ddof=1) if len(times) > 1 else 0,
            "carbon_mean": np.mean([t * GPU_POWER_KW * CARBON_INTENSITY for t in times]),
            "carbon_std": np.std([t * GPU_POWER_KW * CARBON_INTENSITY for t in times], ddof=1) if len(times) > 1 else 0,
            "n": len(times),
        }
    # Plot for each model and task
    models = sorted(set(m for (m, _, _) in stats.keys()))
    tasks = ["News", "Sentiment"]  # only those present
    for model in models:
        for task in tasks:
            # Gather data for this model+task
            data = []
            methods_in_order = []
            for method in METHOD_ORDER:
                key = (model, task, method)
                if key in stats:
                    data.append(stats[key])
                    methods_in_order.append(method)
            if not data:
                continue
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Training time bar plot
            x = np.arange(len(methods_in_order))
            means = [d["mean"] for d in data]
            stds = [d["std"] for d in data]
            ax1.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("Set2", len(methods_in_order)))
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods_in_order, rotation=45, ha="right")
            ax1.set_ylabel("Training Time (GPU hours)")
            ax1.set_title(f"{model} - {task}")
            ax1.grid(True, axis='y', alpha=0.3)
            # Carbon emissions bar plot
            carbon_means = [d["carbon_mean"] for d in data]
            carbon_stds = [d["carbon_std"] for d in data]
            ax2.bar(x, carbon_means, yerr=carbon_stds, capsize=5, color=sns.color_palette("Set2", len(methods_in_order)))
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods_in_order, rotation=45, ha="right")
            ax2.set_ylabel("Estimated CO₂ Emissions (kg)")
            ax2.set_title(f"{model} - {task}")
            ax2.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            output_path = output_dir / f"training_time_carbon_{model}_{task}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_results_dir', type=str, required=True,
                        help='Root directory containing baseline run folders (e.g., results/baselines)')
    parser.add_argument('--output_dir', type=str, default='figures/training_time',
                        help='Output directory for PDF figures')
    args = parser.parse_args()

    root = Path(args.raw_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_raw_results_files(root)
    if not json_files:
        print("No raw_results.json files found. Please check the path.")
        return

    all_results = []
    for jf in json_files:
        all_results.extend(extract_data(jf))
    if not all_results:
        print("No training time data extracted.")
        return

    aggregate_and_plot(all_results, output_dir)

if __name__ == "__main__":
    main()