# Analysis Scripts

This directory contains the post-hoc analysis scripts used to produce
the tables and figures in the paper from the raw outputs of
`run_baseline_comparison.py`.

All scripts assume they are run **from the repository root** and read
from `experiments/` or `results/baselines/` (produced by the training
pipeline). None of them require GPUs, though `calibration.py` benefits
from one.

## Prerequisites

Before running any script here:

1. **Training output must exist.** Most scripts read from
   `results/baselines/<experiment>/<timestamp>/raw_results.json` or
   `experiments/**/experiment_results.json`. If you haven't run the
   pipeline, start with the smoke test described in the root
   `README.md`.

2. **Path conventions matter.** Two scripts
   (`extract_peak_memory_to_csv.py` and
   `extract_training_time_carbon.py`) infer the model and task from
   the directory path. They expect:

   ```
   results/baselines/<backbone>_<task>/<timestamp>/raw_results.json
   ```

      For example, `results/baselines/tiroberta_tidg/<timestamp>/raw_results.json`.
   A run stored under `results/smoke_multi/` will be skipped because
   neither "tiroberta" nor "tidg" appears in the path.

3. **Minimum seeds for statistical tests.** Scripts that perform
   pairwise comparisons need at least 2 seeds per method. Three or
   more is better. The paper uses 5.

4. **Progressive runs for the rank heatmap.** Only runs of
   `progressive` PEFT methods produce a `rank_history` field, and only
   `*_report.json` files under `experiments/**/reports/` contain it.
   SAP-PEFT and the three Progressive variants qualify; LoRA, DoRA,
   AdaLoRA, Vanilla, and Full Fine-Tuning do not.

## Scripts

### Dataset statistics

Read the CSVs directly from `data/processed/`. No arguments.

- `compute_data_stats.py` - class counts, average word length,
  vocabulary size. Prints two LaTeX rows to stdout.
- `compute_raw_stats.py` - variant that reports max and average word
  and character lengths per split.

```bash
python scripts/analysis/compute_data_stats.py
python scripts/analysis/compute_raw_stats.py
```

### Calibration and energy

- `calibration_measures.py` - extracts ECE, Adaptive Calibration
  Error, Brier score, and classwise ECE from every
  `experiments/**/experiment_results.json`. No arguments; writes
  `calibration_metrics.csv` in the working directory.

  ```bash
  python scripts/analysis/calibration_measures.py
  ```

- `calibration.py` - validates the NVML-based energy meter against
  its theoretical FLOP-based estimate. Requires a **base** experiment
  config (`configs/experiments/*.yaml`), not a baseline config.
  Falls back to theoretical-only output if NVML is unavailable.

  ```bash
  python scripts/analysis/calibration.py \
      --config configs/experiments/tiroberta_tidg_base.yaml
  ```

  Arguments:
  - `--config` (default `configs/experiments/tiroberta_tidg_base.yaml`):
    path to a base experiment config under `configs/experiments/`
  - `--seed` (default `42`)
  - `--num-runs` (default `100`)

  Expected output: a calibration factor plus the theoretical and
  measured joules, or `Real measurement failed. Check NVML permissions
  or hardware.` on systems without NVML. Both are valid outcomes.

- `extract_peak_memory_to_csv.py` - extracts peak GPU memory per
  method from `raw_results.json` files.

  ```bash
  python scripts/analysis/extract_peak_memory_to_csv.py \
      --results-root results/baselines \
      --output peak_memory.csv
  ```

  Arguments:
  - `--results-root` (optional): directory to recursively search for
    `raw_results.json`
  - `--json` (optional, repeatable): explicit file paths, for cases
    where the results directory is unusual
  - `--output` (default `peak_memory.csv`)

  You must supply either `--results-root` or at least one `--json`.

- `extract_training_time_carbon.py` - training-time and estimated
  CO2 emissions, with bar plots.

  ```bash
  python scripts/analysis/extract_training_time_carbon.py \
      --raw_results_dir results/baselines \
      --output_dir figures/training_time
  ```

  Arguments:
  - `--raw_results_dir` (required): root directory to search
  - `--output_dir` (default `figures/training_time`)

  Same path-inference caveat as `extract_peak_memory_to_csv.py`:
  the run directory must contain both a model token and a task token
  (e.g. `results/baselines/tiroberta_tidg/...`).

- `efficiency_plot.py` - single-panel efficiency scatter
  (trainable parameters vs. accuracy).

  ```bash
  python scripts/analysis/efficiency_plot.py \
      --json results/baselines/tiroberta_tidg/<timestamp>/statistical_analysis.json \
      --output figures/efficiency_scatter
  ```

  Arguments:
  - `--json` (required): path to a `statistical_analysis.json`
  - `--output` (default `efficiency_scatter`): basename for the output
    `.pdf` and `.png`
  - `--title` (optional): plot title


### Statistical analysis

- `compute_cohens_d.py` - paired Cohen's d between two methods using
  per-seed accuracies from `raw_results.json`. No arguments; walks
  `results/baselines/` recursively and writes `cohens_d_results.csv`.

  ```bash
  python scripts/analysis/compute_cohens_d.py
  ```

  **The two methods being compared are hard-coded at the top of the
  file:**

  ```python
  METHOD1 = "SAP-PEFT"
  METHOD2 = "LoRA"
  ```

  To compare other pairs, edit those constants. The script will skip
  any (model, task) combination that doesn't contain both methods
  with at least 2 common seeds.

- `generate_figures_from_json.py` - the main figure suite: efficiency
  scatter, energy scatter, latency bar plot, Pareto parallel
  coordinates, p-value heatmap, memory bar plot.

  ```bash
  python scripts/analysis/generate_figures_from_json.py \
      --json results/baselines/<experiment>/<timestamp>/statistical_analysis.json \
      --output-dir figures/<experiment>
  ```

  Arguments:
  - `--json` (required): path to a `statistical_analysis.json`
  - `--output-dir` (required): directory for output figures

  Produces six figure pairs (`.pdf` + `.png`) unless the source
  file lacks a Pareto frontier, in which case the corresponding
  figure is skipped.

### Ablation

- `make_ablation_config.py` - single-config generator. Used by
  `run_ablation.py` and available as a standalone utility.

- `run_ablation.py` - orchestrates ablation sweeps. Generates one
  variant YAML per value, then runs the standard baseline runner.

  ```bash
  # Dry run: generate configs, do not train
  python scripts/analysis/run_ablation.py \
      --config configs/baselines/tiroberta_tidg_sap_peft.yaml \
      --dry-run

  # Run a single axis
  python scripts/analysis/run_ablation.py \
      --config configs/baselines/tiroberta_tidg_sap_peft.yaml \
      --param temperature
  ```

  Arguments:
  - `--config` (default: `configs/baselines/tiroberta_tidg_sap_peft.yaml`)
  - `--param` (optional): restrict to one axis; omit for all axes
  - `--seeds` (default `42,43,44,45,46`)
  - `--out-dir` (default `ablation_tmp_configs`)
  - `--dry-run`: generate YAMLs but do not launch training

- `run_all_ablations.ps1` - PowerShell driver for the full sweep on a
  chosen backbone.

  ```powershell
  .\scripts\analysis\run_all_ablations.ps1 -Backbone tiroberta_tidg -DryRun
  ```

  Arguments:
  - `-Backbone` (required): one of `tiroberta_tidg`, `afro_xlmr_tidg`,
    `gpt2_tidg`
  - `-Axes` (optional): subset of axes; omit for all
  - `-DryRun`: print commands, do not execute

- `extract_sensitivity.py` - reads the ablation outputs and produces
  a CSV plus LaTeX rows for the sensitivity table.

  ```bash
  python scripts/analysis/extract_sensitivity.py \
      --results-dir results/baselines \
      --pattern sap_peft_temperature \
      --output sensitivity_temperature.csv
  ```

  Arguments:
  - `--results-dir` (default `results/baselines`)
  - `--pattern` (default `sap_peft_temperature`)
  - `--output` (default `sensitivity_temperature.csv`)

  The `--pattern` value is matched against the ablation method names
  generated by `run_ablation.py`, e.g. `sap_peft_temperature`,
  `sap_peft_update_freq`, `sap_peft_min_rank`, `sap_peft_max_rank`,
  `sap_peft_beta2`.

#### Running a single ablation manually

For a one-off - for example, to fill in the temperature = 3.0 cell
of the sensitivity table - use the generator directly:

```bash
python scripts/analysis/make_ablation_config.py \
    --base configs/baselines/tiroberta_tidg_sap_peft.yaml \
    --override peft.temperature=3.0 \
    --name sap_peft_temperature_3_0 \
    --out ablation_tmp_configs/tiroberta_tidg__temperature_3_0.yaml

python scripts/run_baseline_comparison.py \
    --config ablation_tmp_configs/tiroberta_tidg__temperature_3_0.yaml \
    --sequential \
    --seeds 42,43,44,45,46
```

This produces a single output directory under `results/baselines/`,
which the sensitivity-table extractor can then read:

```bash
python scripts/analysis/extract_sensitivity.py \
    --pattern sap_peft_temperature \
    --output sensitivity_temperature.csv
```

  #### Complete ablation workflow

From the repository root:

```bash
# 1. Dry run: generate all variant YAMLs and verify
./scripts/analysis/run_all_ablations.ps1 -Backbone tiroberta_tidg -DryRun

# 2. Real run: sweep all axes
./scripts/analysis/run_all_ablations.ps1 -Backbone tiroberta_tidg

# 3. Same for Afro-XLM-R
./scripts/analysis/run_all_ablations.ps1 -Backbone afro_xlmr_tidg

# 4. Extract the sensitivity tables
python scripts/analysis/extract_sensitivity.py --pattern sap_peft_temperature
python scripts/analysis/extract_sensitivity.py --pattern sap_peft_update_freq
python scripts/analysis/extract_sensitivity.py --pattern sap_peft_min_rank
python scripts/analysis/extract_sensitivity.py --pattern sap_peft_max_rank
python scripts/analysis/extract_sensitivity.py --pattern sap_peft_beta2

# 5. Paste the printed LaTeX rows into the sensitivity table
```

### Figures

- `generate_figures_from_json.py` - see Statistical analysis above.

- `plot_overfitting_curves.py` - mean ± std train/validation curves
  aggregated over seeds.

  ```bash
  python scripts/analysis/plot_overfitting_curves.py \
      --experiments_root experiments \
      --output_dir figures/overfitting
  ```

  Arguments:
  - `--experiments_root` (required)
  - `--output_dir` (default `figures/overfitting`)
  - `--min_seeds` (default `2`)

  Requires at least `--min_seeds` runs per group. Groups with fewer
  seeds are skipped with a warning.

- `plot_rank_heatmap.py` - per-layer rank evolution heatmaps for the
  progressive PEFT variants.

  ```bash
  python scripts/analysis/plot_rank_heatmap.py \
      --experiments-root experiments \
      --output-dir figures/rank_heatmaps
  ```

  Arguments:
  - `--experiments-root` (default `experiments`)
  - `--output-dir` (default `figures/rank_heatmaps`)

  Discovers every `*_report.json` under the experiments tree that
  contains a non-empty `rank_history`. Runs that don't use a
  progressive method are silently skipped. Output filenames are
  derived from the report path.


## Verification

To confirm all analysis scripts are working end-to-end, run the
following sequence. It requires a training run stored under
`results/baselines/<backbone>_<task>/<timestamp>/` with at least
two methods and two seeds.

Set the run directory once:

```powershell
$RUN = (Get-ChildItem results\baselines\<backbone>_<task> -Directory |
        Sort-Object Name -Descending | Select-Object -First 1).FullName
```

Then verify each script produces its expected outputs:

| Script | Command | Expected artefact |
|---|---|---|
| `compute_data_stats.py` | `python scripts/analysis/compute_data_stats.py` | Two LaTeX rows on stdout |
| `compute_raw_stats.py` | `python scripts/analysis/compute_raw_stats.py` | Two LaTeX rows on stdout |
| `calibration_measures.py` | `python scripts/analysis/calibration_measures.py` | `calibration_metrics.csv` |
| `calibration.py` | `python scripts/analysis/calibration.py --config configs/experiments/tiroberta_tidg_base.yaml` | Calibration factor on stdout |
| `extract_peak_memory_to_csv.py` | `python scripts/analysis/extract_peak_memory_to_csv.py --results-root results/baselines --output $RUN\peak_memory.csv` | `peak_memory.csv` |
| `extract_training_time_carbon.py` | `python scripts/analysis/extract_training_time_carbon.py --raw_results_dir results/baselines --output_dir $RUN\figures\training_time` | PDFs in `$RUN\figures\training_time` |
| `efficiency_plot.py` | `python scripts/analysis/efficiency_plot.py --json $RUN\statistical_analysis.json --output $RUN\figures\efficiency_single` | `efficiency_single.pdf` and `.png` |
| `compute_cohens_d.py` | `python scripts/analysis/compute_cohens_d.py` | `cohens_d_results.csv` |
| `generate_figures_from_json.py` | `python scripts/analysis/generate_figures_from_json.py --json $RUN\statistical_analysis.json --output-dir $RUN\figures` | Six figure pairs in `$RUN\figures` |
| `plot_overfitting_curves.py` | `python scripts/analysis/plot_overfitting_curves.py --experiments_root experiments --output_dir $RUN\figures\overfitting` | One PDF per run group |
| `plot_rank_heatmap.py` | `python scripts/analysis/plot_rank_heatmap.py --experiments-root experiments --output-dir $RUN\figures\rank_heatmaps` | One PDF per progressive run |
| `collect_kappa_mcc.py` | `python scripts/collect_kappa_mcc.py --experiments-root experiments --output $RUN\kappa_mcc.csv --tex $RUN\kappa_mcc.tex` | `kappa_mcc.csv` and `kappa_mcc.tex` |

For the ablation scripts, dry-run mode is sufficient:

```powershell
python scripts\analysis\run_ablation.py `
    --config configs\baselines\tiroberta_tidg_sap_peft.yaml `
    --dry-run

.\scripts\analysis\run_all_ablations.ps1 -Backbone tiroberta_tidg -DryRun
```

Both should print the sequence of commands they would execute,
without launching any training.

## Notes

- Several scripts write their output to the **current working
  directory** (`calibration_metrics.csv`, `cohens_d_results.csv`,
  `kappa_mcc.csv`, `kappa_mcc.tex`). Running from the repository root
  keeps them from scattering into subdirectories.
- The `scripts/scrub.ps1` script removes generated analysis outputs
  before anonymisation. Re-run the affected scripts after cloning the
  anonymised repo if you need the CSV and LaTeX artefacts.
- Matplotlib figures are written to `figures/` by default unless an
  explicit output directory is provided.
    