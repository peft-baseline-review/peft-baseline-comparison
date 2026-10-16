import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Path to your experiments folder (adjust if needed)
EXPERIMENTS_ROOT = Path("experiments")

# Mapping from desired output metric names to JSON keys
METRIC_KEYS = {
    "ece": "ece",
    "ace": "adaptive_calibration_error",
    "brier_score": "brier_score",
    "classwise_ece_mean": "classwise_ece_mean"
}

# Mapping from full model name to short name for the paper
MODEL_NAME_MAP = {
    "fgaim/tiroberta-base": "TiroBERTa",
    "Davlan/afro-xlmr-base": "Afro-XLM-R",
    "openai-community/gpt2": "GPT-2",
}

def extract_calibration_metrics(json_path):
    """Extract calibration metrics from an experiment_results.json file."""
    with open(json_path) as f:
        data = json.load(f)
    model_metrics = data.get("results", {}).get("model_metrics", {})
    result = {}
    for out_name, json_key in METRIC_KEYS.items():
        result[out_name] = model_metrics.get(json_key, None)
    return result

def get_method_display(json_path):
    """Extract PEFT method name and return a display string, distinguishing progressive variants."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        config = data.get("results", {}).get("experiment_report", {}).get("configuration", {})
        peft = config.get("peft", {})
        method = peft.get("method", "unknown").lower()
        if method == "progressive":
            strategy = peft.get("allocation_strategy", "budget")
            schedule = peft.get("rank_schedule", "adaptive")
            if strategy == "structure":
                return "SAP-PEFT"
            elif schedule == "adaptive":
                return "Progressive-Adaptive"
            elif schedule == "linear":
                return "Progressive-Linear"
            elif schedule == "fixed":
                return "Progressive-Fixed"
            else:
                return f"Progressive ({schedule})"
        else:
            # Map standard methods
            return method.upper() if method in ["lora", "dora", "adalora"] else method.capitalize()
    except Exception as e:
        print(f"Error extracting method from {json_path}: {e}")
        return "unknown"

def get_model_short_name(json_path):
    """Extract model name and map to short name."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        config = data.get("results", {}).get("experiment_report", {}).get("configuration", {})
        model = config.get("model", {}).get("name", "")
        return MODEL_NAME_MAP.get(model, model)
    except Exception as e:
        print(f"Error extracting model from {json_path}: {e}")
        return "unknown"

def get_task_from_path(json_path):
    """Infer task from directory path (e.g., 'sentiment' or 'news')."""
    parts = json_path.parts
    for part in parts:
        if part == "sentiment":
            return "sentiment"
        elif part == "news":
            return "news"
    return "unknown"

def main():
    # Find all experiment_results.json files
    all_jsons = list(EXPERIMENTS_ROOT.rglob("experiment_results.json"))
    print(f"Found {len(all_jsons)} experiment result files.")

    # Aggregate: key = (model_short, task, method) -> list of metric dicts
    agg = defaultdict(list)

    for json_path in all_jsons:
        try:
            metrics = extract_calibration_metrics(json_path)
            # Skip if all metrics are None
            if not any(v is not None for v in metrics.values()):
                continue
            # Get method and model
            method = get_method_display(json_path)
            model_short = get_model_short_name(json_path)
            task = get_task_from_path(json_path)
            key = (model_short, task, method)
            agg[key].append(metrics)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    # Compute means and stds
    rows = []
    for (model, task, method), metrics_list in agg.items():
        if not metrics_list:
            continue
        row = {"Model": model, "Task": task, "Method": method}
        for metric_name in METRIC_KEYS.keys():
            vals = [d[metric_name] for d in metrics_list if d[metric_name] is not None]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_std"] = std
            else:
                row[f"{metric_name}_mean"] = None
                row[f"{metric_name}_std"] = None
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    # Save to CSV
    df.to_csv("calibration_metrics.csv", index=False)
    print("\nSaved to calibration_metrics.csv")
    print(df.to_string())

    # Optionally generate LaTeX table for news task
    news_df = df[df["Task"] == "news"]
    if not news_df.empty:
        # Sort by Model, Method for readability
        news_df = news_df.sort_values(["Model", "Method"])
        # Select relevant columns
        news_df = news_df[["Model", "Method", "ece_mean", "ece_std", "ace_mean", "ace_std",
                           "brier_score_mean", "brier_score_std", "classwise_ece_mean_mean", "classwise_ece_mean_std"]]
        # Print CSV-like output (or use to_latex if pandas supports)
        print("\n--- LaTeX table for news task (mean ± std) ---")
        print(news_df.to_csv(index=False))

if __name__ == "__main__":
    main()  