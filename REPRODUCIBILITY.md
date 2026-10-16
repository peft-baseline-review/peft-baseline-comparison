# Reproducibility Guide

This document maps every table and figure in the paper to the
commands, random seeds, and configuration files that produced it.

All commands assume the repository root as the working directory.

## Environment

- Python 3.12.8
- CUDA 12.6, cuDNN 8.9
- PyTorch 2.9.1, transformers 4.57.3, peft 0.18.0
- Single NVIDIA RTX 4090 Laptop (16 GB)
- Verified on Windows 11 and Ubuntu 22.04

See `requirements.txt` for the full pinned dependency set.

## Random seeds

All reported experiments use seeds `42, 43, 44, 45, 46`. Seeds are
set throughout the codebase via Python `random`, `numpy.random`, and
`torch.manual_seed`, with deterministic CUDA algorithms enabled:

- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `PYTHONHASHSEED=42`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Data loader workers set to 0 on Windows to avoid pickling issues

## Table and figure to command map

| Paper item | Config | Output directory |
|---|---|---|
| TiroBERTa news | `configs/baselines/tiroberta_tidg_baseline.yaml` | `results/baselines/tiroberta_tidg` |
| TiroBERTa sentiment | `configs/baselines/tiroberta_sentiment_baseline.yaml` | `results/baselines/tiroberta_sentiment` |
| Afro-XLM-R news | `configs/baselines/afro_xlmr_tidg_baseline.yaml` | `results/baselines/afro_xlmr_tidg` |
| Afro-XLM-R sentiment | `configs/baselines/afro_xlmr_sentiment_baseline.yaml` | `results/baselines/afro_xlmr_sentiment` |
| GPT-2 news | `configs/baselines/gpt2_tidg_baseline.yaml` | `results/baselines/gpt2_tidg` |
| GPT-2 sentiment | `configs/baselines/gpt2_sentiment_baseline.yaml` | `results/baselines/gpt2_sentiment` |
| SAP-PEFT ablation | `configs/baselines/tiroberta_tidg_sap_peft.yaml` | `results/baselines/tiroberta_tidg_ablation` |

Universal command:

```bash
python scripts/run_baseline_comparison.py \
    --config <CONFIG_PATH> \
    --sequential \
    --seeds 42,43,44,45,46 \
    --output-dir <OUTPUT_DIR>
```

Windows PowerShell:

```powershell
python scripts\run_baseline_comparison.py `
    --config <CONFIG_PATH> `
    --sequential `
    --seeds 42,43,44,45,46 `
    --output-dir <OUTPUT_DIR>
```

## Smoke test

Before running the full set, verify the pipeline end-to-end on the
fastest backbone/task combination:

```bash
python scripts/run_baseline_comparison.py \
    --config configs/baselines/tiroberta_tidg_vanilla_only.yaml \
    --sequential \
    --seeds 42,43 \
    --output-dir results/smoke
```

On an RTX 4090 Laptop this takes roughly 8-10 minutes and produces
a timestamped directory under `results/smoke/`. See `README.md` for
details.


## Output structure

Each run produces a timestamped directory:

```
results/baselines/<name>/<timestamp>/
    baseline_report.md
    baseline_summary.csv
    statistical_analysis.json
    raw_results.json
    statistical_power_analysis.json
    runner_configuration.json
    <method>/seed_<n>/
        config.yaml
        predictions.json
```
## Statistical analysis

The framework computes, for every method on every experiment:

- Wilcoxon signed-rank on per-seed accuracy (seed-aligned)
- McNemar on paired predictions, aggregated per seed by median
- Holm-Bonferroni correction across all pairwise comparisons
- Pareto frontier in a four-objective space (accuracy, parameter
  efficiency, speed, memory efficiency); hypervolume via Monte Carlo
- Cohen's kappa and MCC, saved as `cohen_kappa` and `mcc` in each
  run's `experiment_results.json`
- Bootstrap confidence intervals for all classification metrics
- Statistical power via Cohen's h, using a normal approximation

These are logged in `statistical_analysis.json` and mirrored into
`baseline_report.md`.


## Extracting kappa and MCC across all runs

The supplementary table of chance-corrected agreement metrics can be
regenerated from the saved experiment outputs:

```bash
python scripts/collect_kappa_mcc.py \
    --experiments-root experiments \
    --output kappa_mcc.csv \
    --tex kappa_mcc.tex
```

The script walks `<experiments-root>/**/experiment_results.json`,
extracts `results.model_metrics.cohen_kappa` and
`results.model_metrics.mcc` for each seed, and aggregates per
experiment and method.

## Analysis scripts

Post-hoc analysis and plotting scripts are in `scripts/analysis/`.
They read from `experiments/` and `results/baselines/`. See
`scripts/analysis/README.md` for usage.

## Hardware and runtime

Reported numbers were produced on a single RTX 4090 Laptop (16 GB).
CPU execution is possible by setting `device: "cpu"` in the base
config, but is substantially slower.

Approximate wall time per full baseline (9 methods × 5 seeds) on
the RTX 4090 Laptop, computed by summing per-seed
`training_time_hours`:

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

## Determinism notes

- On CPU, `torch.use_deterministic_algorithms` is enforced strictly.
- On CUDA, a small number of operations fall back to non-deterministic
  kernels; the runner logs the affected op names at DEBUG level.
  Empirical variance across seeds was below 0.3 percentage points
  absolute for all reported metrics.
- The data cache key includes the file SHA256, seed, split ratio, and
  label map. Changing any of these invalidates the cache and forces
  re-tokenization.

## Contact

During the anonymity period, please use the OpenReview discussion
forum.