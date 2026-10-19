# Anonymous Code Release

This repository accompanies an anonymous submission under double-blind
review. It contains the implementation, configuration system, and
reproduction scripts for every table and figure in the paper.

## Overview

A unified baseline comparison framework for parameter-efficient
fine-tuning (PEFT) on Tigrinya text classification, including:

- Standard PEFT methods: LoRA, DoRA, AdaLoRA, and full fine-tuning
- A frozen-base vanilla classifier as a negative control
- Progressive PEFT: adaptive rank allocation with three schedules
  (adaptive, linear, fixed) and two allocation strategies
  (budget, structure; the latter is our SAP-PEFT variant)
- Statistical analysis: Wilcoxon signed-rank, McNemar, Holm
  correction, Pareto frontier, Cohen's kappa, MCC, bootstrap
  confidence intervals, and statistical power analysis

The framework is evaluated on three backbones (TiroBERTa, Afro-XLM-R,
GPT-2) across two tasks (binary sentiment, 7-class news genre).


## Repository layout

```
scripts/
    run_baseline_comparison.py     entry point for baseline suites
    collect_kappa_mcc.py           kappa and MCC aggregation
    run_beta_sweep.py              hyperparameter sweep orchestrator
    extract_sweep_results.py       sweep result aggregator
    reproduce_all.ps1              end-to-end reproduction (PowerShell)
    scrub.ps1                      anonymization helper
    analysis/                      post-hoc analysis scripts
experiment_runner.py               single-experiment orchestrator
config/                            Pydantic configuration system
src/data/                          dataset loading, caching, tokenization
src/models/                        PEFT factory and Progressive PEFT wrapper
src/training/                      trainer (AMP, grad accum, early stop)
src/evaluation/                    metrics, calibration, energy, reporting
src/utils/                         MLflow, hyperparameter adapter, risk mitigation
configs/baselines/                 one YAML per reported comparison
configs/experiments/               base experiment YAMLs
configs/model/ configs/task/       fragment YAMLs pulled in via !include
configs/common/                    shared evaluation and training defaults
data/processed/                    Tigrinya datasets (see Data section)
```

## Requirements

- Python 3.12
- CUDA 12.x for GPU training (CPU-only execution is supported but
  substantially slower)
- See `requirements.txt` for the exact pinned dependency set

## Installation

All commands below assume your terminal is in the repository root
(the directory containing this file).

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, allow local
scripts for your user account once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### PyTorch CUDA wheels

The pinned `torch==2.9.1` and `torchvision==0.24.1` in
`requirements.txt` are resolved from PyPI. On Linux the default
wheels are already CUDA-enabled. On Windows, install the CUDA 12.6
build first:

```powershell
pip install torch==2.9.1 torchvision==0.24.1 `
    --index-url https://download.pytorch.org/whl/cu126
```

Then run `pip install -r requirements.txt` as above.

## Data

The processed Tigrinya datasets ship with the repository:

- `data/processed/sentiment/` - binary sentiment
  - `train.csv`, `dev.csv`, `test.csv`
  - delimiter `;`
  - columns `text`, `label`
- `data/processed/tidg/` - 7-class news genre
  - `train.csv`, `val.csv`, `test.csv`
  - delimiter `,`
  - columns `cleaned_text`, `label`

## Quick smoke test

To verify the pipeline runs end-to-end without training every method,
run a two-seed comparison of the single-method vanilla config on the
fastest backbone/task combination (TiroBERTa on TiDG). On an RTX 4090
Laptop this finishes in roughly 8-10 minutes total.

Linux / macOS:

```bash
python scripts/run_baseline_comparison.py \
    --config configs/baselines/tiroberta_tidg_vanilla_only.yaml \
    --sequential \
    --seeds 42,43 \
    --output-dir results/smoke
```

Windows PowerShell:

```powershell
python scripts\run_baseline_comparison.py `
    --config configs\baselines\tiroberta_tidg_vanilla_only.yaml `
    --sequential `
    --seeds 42,43 `
    --output-dir results\smoke
```

A successful run prints two `✅` lines and writes a timestamped
directory under `results/smoke/` containing `baseline_report.md`,
`baseline_summary.csv`, `statistical_analysis.json`, and per-seed
`predictions.json` files.

For a deeper check that also exercises the progressive PEFT code
path, substitute the SAP-PEFT config:

```bash
python scripts/run_baseline_comparison.py \
    --config configs/baselines/tiroberta_tidg_sap_peft.yaml \
    --sequential \
    --seeds 42,43 \
    --output-dir results/smoke
```

## Reproducing the main results

Example: reproducing the TiroBERTa/sentiment baseline (results
reported in Section 5 of the paper):

```bash
python scripts/run_baseline_comparison.py \
    --config configs/baselines/tiroberta_sentiment_baseline.yaml \
    --sequential \
    --seeds 42,43,44,45,46
```

Windows PowerShell:

```powershell
python scripts\run_baseline_comparison.py `
    --config configs\baselines\tiroberta_sentiment_baseline.yaml `
    --sequential `
    --seeds 42,43,44,45,46
```

Results land in
`results/baselines/tiroberta_sentiment/<timestamp>/`, containing:

- `baseline_report.md`
- `baseline_summary.csv`
- `statistical_analysis.json`
- `raw_results.json`
- a full figure suite (`accuracy_barplot.png`,
  `pvalue_heatmap.png`, `pareto_parallel.png`,
  `energy_scatter.png`, and others)

### Output locations

Each run writes to two locations:

- `results/baselines/<experiment>/<timestamp>/` - per-run summary
  (`raw_results.json`, `statistical_analysis.json`, plots).
- `experiments/<Model Name>/<task>/<architecture>/<method>/<timestamp>/`
  - per-seed detail (`experiment_results.json`,
  `predictions.json`, checkpoints).

The `results/` tree is what the sections above describe. The
`experiments/` tree is what `collect_kappa_mcc.py` walks, because it
reads per-seed metrics from individual `experiment_results.json`
files.

### All reported configurations

| Paper item | Config file |
|---|---|
| TiroBERTa news | `configs/baselines/tiroberta_tidg_baseline.yaml` |
| TiroBERTa sentiment | `configs/baselines/tiroberta_sentiment_baseline.yaml` |
| Afro-XLM-R news | `configs/baselines/afro_xlmr_tidg_baseline.yaml` |
| Afro-XLM-R sentiment | `configs/baselines/afro_xlmr_sentiment_baseline.yaml` |
| GPT-2 news | `configs/baselines/gpt2_tidg_baseline.yaml` |
| GPT-2 sentiment | `configs/baselines/gpt2_sentiment_baseline.yaml` |
| SAP-PEFT ablation | `configs/baselines/tiroberta_tidg_sap_peft.yaml` |

To run every reported baseline in one pass (Windows PowerShell):

```powershell
.\scripts\reproduce_all.ps1
```

## Baseline methods

Each baseline YAML defines a set of PEFT methods compared on the same
backbone and task. The standard set includes:

- `full` - full fine-tuning upper bound
- `vanilla` - frozen base plus trainable classifier (negative control)
- `lora` - LoRA on attention projections
- `dora` - DoRA
- `adalora` - AdaLoRA with attention and FFN targets
- `progressive` with `allocation_strategy: budget` and
  `rank_schedule: adaptive | linear | fixed`
- `progressive` with `allocation_strategy: structure` (SAP-PEFT)

Nine methods in total. Per-seed accuracy is compared pairwise across
these methods with Wilcoxon signed-rank tests; paired predictions
are compared with McNemar's test; all p-values are Holm-corrected.

## Extracting kappa and MCC

The supplementary table of Cohen's kappa and MCC can be regenerated
from saved experiment outputs:

```bash
python scripts/collect_kappa_mcc.py \
    --experiments-root experiments \
    --output kappa_mcc.csv \
    --tex kappa_mcc.tex
```

Windows PowerShell:

```powershell
python scripts\collect_kappa_mcc.py `
    --experiments-root experiments `
    --output kappa_mcc.csv `
    --tex kappa_mcc.tex
```

The script walks `<experiments-root>/**/experiment_results.json`,
extracts `results.model_metrics.cohen_kappa` and
`results.model_metrics.mcc` for each seed, and aggregates per
experiment and method. It writes a CSV plus a LaTeX snippet.

## Hyperparameter sweeps

The paper's sensitivity table for the two EMA smoothing factors is
produced by `scripts/run_beta_sweep.py`, which sweeps `peft.ema_beta`
(`beta_1`, per-parameter EMA) and `peft.importance_smoothing`
(`beta_2`, per-module EMA) for SAP-PEFT on both TiDG backbones.

Run the full sweep:

```bash
python scripts/run_beta_sweep.py
```

Or restrict it to one axis and one backbone:

```bash
python scripts/run_beta_sweep.py --sweeps beta1 --backbones tiroberta
```

Arguments:
- `--sweeps` (default: `beta1 beta2`): which axes to sweep
- `--backbones` (default: `tiroberta afro_xlmr`): which backbone/task pairs
- `--seeds` (default: `42 43 44 45 46`)
- `--output-root` (default: `results/sweeps`)

The sweep runs 5 methods × 5 seeds per axis-value on the selected
backbones — roughly 60–90 minutes on an RTX 4090 per (backbone, axis).

Once complete, aggregate into a CSV and a LaTeX snippet:

```bash
python scripts/extract_sweep_results.py
```

This writes `beta_sweep_results.csv` and prints LaTeX table rows to
stdout. The script assumes the `results/sweeps/<backbone>/<axis>/<value>/`
layout produced by `run_beta_sweep.py`.

## Analysis scripts

Post-hoc analysis scripts used to generate the paper's tables and
figures from saved experiment outputs are in `scripts/analysis/`.
See `scripts/analysis/README.md` for the full list and usage
examples.

## Notes on MLflow

Running any command with MLflow enabled (the default in the shipped
configs) creates an `mlruns/` directory in the current working
directory. This directory is not required to reproduce the paper's
results and can be deleted after the run.

## Hardware

Numbers reported in the paper were produced on a single NVIDIA
RTX 4090 Laptop (16 GB). CPU-only execution is supported by setting
`device: "cpu"` in the base config; it is substantially slower.

Approximate wall time per full baseline (9 methods × 5 seeds) on a
single RTX 4090 Laptop, computed by summing per-seed
`training_time_hours` from the raw results:

| Backbone / Task | Wall time |
|---|---|
| TiroBERTa / news | 10 h |
| TiroBERTa / sentiment | 81 h |
| Afro-XLM-R / news | 12 h |
| Afro-XLM-R / sentiment | 82 h |
| GPT-2 / news | 18 h |
| GPT-2 / sentiment | 99 h |

Total compute for the paper's full experiment set is approximately
300 GPU-hours. These figures assume `--sequential` execution on a
single GPU; parallel execution on multi-GPU hardware reduces wall
time proportionally.

## Reproducibility

See `REPRODUCIBILITY.md` for exact seeds, environment versions, and
a mapping from paper tables and figures to reproduction commands.

## License

Released for the sole purpose of peer review. All rights reserved
until the anonymity period ends.