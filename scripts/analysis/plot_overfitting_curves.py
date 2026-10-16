#!/usr/bin/env python3
"""
auto_plot_overfitting_final.py

Scans experiments directory, groups JSON files by run directory,
and generates mean ± std training/validation curves with concise titles.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("Set2")

# ----------------------------------------------------------------------
# Text sanitisation
# ----------------------------------------------------------------------
def sanitize_text(text: str) -> str:
    """Replace non-breaking hyphens (U+2011) and other dashes with regular hyphen."""
    text = text.replace('\u2011', '-')
    text = text.replace('-', '-')
    return text

# ----------------------------------------------------------------------
# Naming helpers
# ----------------------------------------------------------------------
def clean_method_name(method_folder: str) -> str:
    """Map folder or experiment name to paper method name."""
    mapping = {
        "sap_peft": "SAP-PEFT",
        "progressive": "SAP-PEFT",          # generic progressive folder used for SAP
        "full_finetuning": "Full Fine-Tuning",
        "full": "Full Fine-Tuning",
        "lora": "LoRA",
        "dora": "DoRA",
        "adalora": "AdaLoRA",
        "vanilla": "Vanilla",
        "dy_lora": "DyLoRA",
        "sa_lora": "SaLoRA",
        "progressive_adaptive": "Progressive-Adaptive",
        "progressive_linear": "Progressive-Linear",
        "progressive_fixed": "Progressive-Fixed",
    }
    mf = method_folder.lower()
    if mf in mapping:
        return mapping[mf]
    for key, val in mapping.items():
        if key.lower() == mf:
            return val
    return method_folder.replace('_', ' ').title()

def clean_model_name(model_str: str) -> str:
    """Convert various model name variations to standard form."""
    mapping = {
        "tiroberta": "TiroBERTa",
        "afro_xlmr": "Afro-XLM-R",
        "afro-xlmr": "Afro-XLM-R",
        "gpt2": "GPT-2",
        "gpt-2": "GPT-2",
        "xlm-roberta": "Afro-XLM-R",
        "roberta": "TiroBERTa",
    }
    lower = model_str.lower()
    if lower in mapping:
        return mapping[lower]
    # Heuristic: if it contains "xlm" or "roberta", guess
    if "xlm" in lower:
        return "Afro-XLM-R"
    if "roberta" in lower:
        return "TiroBERTa"
    return model_str.title()

def make_title(run_dir: Path) -> str:
    """Extract model, method, task from run_dir path and return clean title."""
    parts = run_dir.parts
    try:
        exp_idx = parts.index('experiments')
        # Experiment name (e.g., "Afro-XLM-R SAP-PEFT" or "TiroBERTa TiDG LoRA")
        exp_name = parts[exp_idx + 1]
        task = parts[exp_idx + 2] if len(parts) > exp_idx + 2 else ""
        # Determine model
        if "TiroBERTa" in exp_name:
            model = "TiroBERTa"
        elif "Afro" in exp_name or "XLM" in exp_name:
            model = "Afro-XLM-R"
        elif "GPT" in exp_name:
            model = "GPT-2"
        else:
            # fallback: use the folder name of the architecture (e.g., "xlm-roberta")
            arch_folder = parts[exp_idx + 3] if len(parts) > exp_idx + 3 else ""
            model = clean_model_name(arch_folder)
        # Determine method
        method = "Unknown"
        # First check the folder name (last part of run_dir may be e.g., "lora", "progressive")
        method_folder = parts[-1] if len(parts) > exp_idx + 4 else ""
        method = clean_method_name(method_folder)
        if method == method_folder:  # not mapped
            # Try to infer from experiment name
            if "SAP-PEFT" in exp_name or "SAP" in exp_name:
                method = "SAP-PEFT"
            elif "Full Fine-tuning" in exp_name or "Full Fine-Tuning" in exp_name:
                method = "Full Fine-Tuning"
            elif "LoRA" in exp_name:
                method = "LoRA"
            elif "DoRA" in exp_name:
                method = "DoRA"
            elif "AdaLoRA" in exp_name:
                method = "AdaLoRA"
            elif "Vanilla" in exp_name:
                method = "Vanilla"
            elif "DyLoRA" in exp_name:
                method = "DyLoRA"
            elif "SaLoRA" in exp_name:
                method = "SaLoRA"
            elif "Progressive Adaptive" in exp_name:
                method = "Progressive-Adaptive"
            elif "Progressive Linear" in exp_name:
                method = "Progressive-Linear"
            elif "Progressive Fixed" in exp_name:
                method = "Progressive-Fixed"
            else:
                method = method_folder.replace('_', ' ').title()
        # Task name
        if task.lower() == "tidg":
            task = "News"
        else:
            task = task.title()
        title = f"{model} - {method} - {task}"
    except (ValueError, IndexError):
        title = str(run_dir)
    return sanitize_text(title)

def safe_filename(run_dir: Path) -> str:
    """Generate a short, safe filename from the title."""
    title = make_title(run_dir)
    # Remove problematic characters, replace spaces and dashes
    fname = title.replace(" - ", "_").replace(" ", "_").replace("-", "-")
    fname = re.sub(r'[\\/*?:"<>|]', '_', fname)
    return f"{fname}_curves.pdf"

# ----------------------------------------------------------------------
# Data loading and aggregation
# ----------------------------------------------------------------------
def load_history(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: cannot read {json_path}: {e}")
        return None
    history = data.get("training_history")
    if history is None:
        print(f"Warning: 'training_history' missing in {json_path}")
        return None
    required = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    if not all(k in history for k in required):
        print(f"Warning: missing keys in training_history of {json_path}")
        return None
    return history

def aggregate_across_seeds(json_paths: List[Path]) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    histories = []
    epoch_lengths = []
    for p in json_paths:
        hist = load_history(p)
        if hist is not None:
            histories.append(hist)
            epoch_lengths.append(len(hist['train_loss']))
    if len(histories) < 2:
        print(f"Only {len(histories)} valid histories (need ≥2) - skipping")
        return None
    min_epochs = min(epoch_lengths)
    print(f"Truncating to {min_epochs} epochs (seeds had {epoch_lengths})")
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1']
    aggregated = {}
    for metric in metrics:
        values = []
        for h in histories:
            if metric in h:
                values.append(h[metric][:min_epochs])
            else:
                # Fallback: if F1 missing, use accuracy (should not happen)
                continue
        if not values:
            continue
        arr = np.array(values)
        aggregated[metric] = (np.mean(arr, axis=0), np.std(arr, axis=0))
    if not aggregated:
        return None
    return aggregated

def plot_curves(aggregated: Dict, output_path: Path, title: str):
    epochs = range(1, len(aggregated['train_loss'][0]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    train_loss_mean, train_loss_std = aggregated['train_loss']
    val_loss_mean, val_loss_std = aggregated['val_loss']
    ax1.plot(epochs, train_loss_mean, label='Train', linewidth=2)
    ax1.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
    ax1.plot(epochs, val_loss_mean, label='Validation', linewidth=2)
    ax1.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    train_acc_mean, train_acc_std = aggregated['train_acc']
    val_acc_mean, val_acc_std = aggregated['val_acc']
    ax2.plot(epochs, train_acc_mean, label='Train', linewidth=2)
    ax2.fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.2)
    ax2.plot(epochs, val_acc_mean, label='Validation', linewidth=2)
    ax2.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

# ----------------------------------------------------------------------
# Directory scanning
# ----------------------------------------------------------------------
def find_run_groups(experiments_root: Path) -> Dict[Path, List[Path]]:
    groups = defaultdict(list)
    for json_file in experiments_root.rglob("experiment_results.json"):
        timestamp_dir = json_file.parent
        run_dir = timestamp_dir.parent
        groups[run_dir].append(json_file)
    return groups

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_root', type=str, required=True,
                        help='Root directory containing experiment subdirectories (e.g., "./experiments")')
    parser.add_argument('--output_dir', type=str, default='figures/overfitting',
                        help='Directory where PDFs will be saved')
    parser.add_argument('--min_seeds', type=int, default=2,
                        help='Minimum number of seeds required to generate a plot')
    args = parser.parse_args()

    root = Path(args.experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiments root not found: {root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = find_run_groups(root)
    print(f"Found {len(groups)} run groups")

    for run_dir, json_files in groups.items():
        if len(json_files) < args.min_seeds:
            print(f"Skipping {run_dir} - only {len(json_files)} seeds (need ≥{args.min_seeds})")
            continue
        print(f"Processing {run_dir} with {len(json_files)} seeds")
        aggregated = aggregate_across_seeds(json_files)
        if aggregated is None:
            print(f"Skipping {run_dir} due to insufficient valid data")
            continue
        output_path = output_dir / safe_filename(run_dir)
        title = make_title(run_dir)
        plot_curves(aggregated, output_path, title)

if __name__ == '__main__':
    main()