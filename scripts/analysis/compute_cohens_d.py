#!/usr/bin/env python3
"""
compute_cohens_d.py

Computes paired Cohen's d between any two methods using per-seed accuracy
from raw_results.json files.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE_DIR = Path("results/baselines")
METHOD1 = "SAP-PEFT"
METHOD2 = "LoRA"

# Comprehensive map covering every experiment/backbone combination.
METHOD_MAP = {
    # TiroBERTa / news
    "tiroberta_tidg_full_finetuning":      "Full Fine-Tuning",
    "tiroberta_tidg_lora":                 "LoRA",
    "tiroberta_tidg_dora":                 "DoRA",
    "tiroberta_tidg_adalora":              "AdaLoRA",
    "tiroberta_tidg_progressive_adaptive": "Progressive-Adaptive",
    "tiroberta_tidg_progressive_linear":   "Progressive-Linear",
    "tiroberta_tidg_progressive_fixed":    "Progressive-Fixed",
    "tiroberta_tidg_sap_peft":             "SAP-PEFT",
    "tiroberta_tidg_vanilla":              "Vanilla",
    # TiroBERTa / sentiment
    "tiroberta_sentiment_full_finetuning":      "Full Fine-Tuning",
    "tiroberta_sentiment_lora":                 "LoRA",
    "tiroberta_sentiment_dora":                 "DoRA",
    "tiroberta_sentiment_adalora":              "AdaLoRA",
    "tiroberta_sentiment_progressive_adaptive": "Progressive-Adaptive",
    "tiroberta_sentiment_progressive_linear":   "Progressive-Linear",
    "tiroberta_sentiment_progressive_fixed":    "Progressive-Fixed",
    "tiroberta_sentiment_sap_peft":             "SAP-PEFT",
    "tiroberta_sentiment_vanilla":              "Vanilla",
    # Afro-XLM-R / news
    "afro_xlmr_tidg_full_finetuning":      "Full Fine-Tuning",
    "afro_xlmr_tidg_lora":                 "LoRA",
    "afro_xlmr_tidg_dora":                 "DoRA",
    "afro_xlmr_tidg_adalora":              "AdaLoRA",
    "afro_xlmr_tidg_progressive_adaptive": "Progressive-Adaptive",
    "afro_xlmr_tidg_progressive_linear":   "Progressive-Linear",
    "afro_xlmr_tidg_progressive_fixed":    "Progressive-Fixed",
    "afro_xlmr_tidg_sap_peft":             "SAP-PEFT",
    "afro_xlmr_tidg_vanilla":              "Vanilla",
    # Afro-XLM-R / sentiment
    "afro_xlmr_sentiment_full_finetuning":      "Full Fine-Tuning",
    "afro_xlmr_sentiment_lora":                 "LoRA",
    "afro_xlmr_sentiment_dora":                 "DoRA",
    "afro_xlmr_sentiment_adalora":              "AdaLoRA",
    "afro_xlmr_sentiment_progressive_adaptive": "Progressive-Adaptive",
    "afro_xlmr_sentiment_progressive_linear":   "Progressive-Linear",
    "afro_xlmr_sentiment_progressive_fixed":    "Progressive-Fixed",
    "afro_xlmr_sentiment_sap_peft":             "SAP-PEFT",
    "afro_xlmr_sentiment_vanilla":              "Vanilla",
    # GPT-2 / news
    "gpt2_tidg_full_finetuning":      "Full Fine-Tuning",
    "gpt2_tidg_lora":                 "LoRA",
    "gpt2_tidg_dora":                 "DoRA",
    "gpt2_tidg_adalora":              "AdaLoRA",
    "gpt2_tidg_progressive_adaptive": "Progressive-Adaptive",
    "gpt2_tidg_progressive_linear":   "Progressive-Linear",
    "gpt2_tidg_progressive_fixed":    "Progressive-Fixed",
    "gpt2_tidg_sap_peft":             "SAP-PEFT",
    "gpt2_tidg_vanilla":              "Vanilla",
    # GPT-2 / sentiment (keys without task for the runner)
    "gpt2_full_finetuning":      "Full Fine-Tuning",
    "gpt2_lora":                 "LoRA",
    "gpt2_dora":                 "DoRA",
    "gpt2_adalora":              "AdaLoRA",
    "gpt2_progressive_adaptive": "Progressive-Adaptive",
    "gpt2_progressive_linear":   "Progressive-Linear",
    "gpt2_progressive_fixed":    "Progressive-Fixed",
    "gpt2_sap_peft":             "SAP-PEFT",
    "gpt2_vanilla":              "Vanilla",
    # Afro-XLM-R without task suffix (alternate runner key format)
    "afro_xlmr_full_finetuning":      "Full Fine-Tuning",
    "afro_xlmr_lora":                 "LoRA",
    "afro_xlmr_dora":                 "DoRA",
    "afro_xlmr_adalora":              "AdaLoRA",
    "afro_xlmr_progressive_adaptive": "Progressive-Adaptive",
    "afro_xlmr_progressive_linear":   "Progressive-Linear",
    "afro_xlmr_progressive_fixed":    "Progressive-Fixed",
    "afro_xlmr_sap_peft":             "SAP-PEFT",
    "afro_xlmr_vanilla":              "Vanilla",
}

def get_model_from_path(json_path: Path) -> str:
    s = str(json_path).lower()
    if "afro_xlmr" in s or "afro-xlmr" in s: return "Afro-XLM-R"
    if "tiroberta" in s: return "TiroBERTa"
    if "gpt2" in s or "gpt-2" in s: return "GPT-2"
    return "Unknown"

def get_task_from_path(json_path: Path) -> str:
    s = str(json_path).lower()
    if "tidg" in s: return "news"
    if "sentiment" in s: return "sentiment"
    return "unknown"

def extract_per_seed_accuracies(json_path: Path):
    data = {}
    model = get_model_from_path(json_path)
    task = get_task_from_path(json_path)
    if model == "Unknown" or task == "unknown":
        return data
    with open(json_path, 'r') as f:
        content = json.load(f)
    for method_key, seeds in content.items():
        display = METHOD_MAP.get(method_key.lower())
        if display is None:
            continue
        for entry in seeds:
            if not entry.get("success", False):
                continue
            seed = entry.get("seed")
            acc = entry.get("accuracy")
            if seed is not None and acc is not None:
                data[(model, task, display, seed)] = acc
    return data

def main():
    json_files = list(BASE_DIR.rglob("raw_results.json"))
    if not json_files:
        print(f"No raw_results.json files found under {BASE_DIR}.")
        return

    all_data = {}
    for jf in json_files:
        all_data.update(extract_per_seed_accuracies(jf))

    grouped = defaultdict(lambda: defaultdict(dict))
    for (model, task, method, seed), acc in all_data.items():
        grouped[(model, task)][method][seed] = acc

    results = []
    for (model, task), methods in grouped.items():
        if METHOD1 not in methods or METHOD2 not in methods:
            print(f"Skipping {model} {task}: missing {METHOD1} or {METHOD2}")
            continue
        common = sorted(set(methods[METHOD1]) & set(methods[METHOD2]))
        if len(common) < 2:
            print(f"Skipping {model} {task}: only {len(common)} common seed(s)")
            continue
        v1 = np.array([methods[METHOD1][s] for s in common])
        v2 = np.array([methods[METHOD2][s] for s in common])
        diff = v1 - v2
        sd = float(np.std(diff, ddof=1))
        d = float(np.mean(diff) / sd) if sd > 0 else 0.0
        results.append({
            "Model": model,
            "Task": task,
            f"{METHOD1}_mean": float(np.mean(v1)),
            f"{METHOD2}_mean": float(np.mean(v2)),
            "Mean_diff": float(np.mean(diff)),
            "Std_diff": sd,
            "Cohen_d": d,
            "N_seeds": len(common),
        })

    df = pd.DataFrame(results)
    print("\nPaired Cohen's d between SAP-PEFT and LoRA:\n")
    print(df.round(3).to_string(index=False))
    df.to_csv("cohens_d_results.csv", index=False)
    print("\nSaved to cohens_d_results.csv")

if __name__ == "__main__":
    main()