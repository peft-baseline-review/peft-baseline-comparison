<#
.SYNOPSIS
    Selectively copy only the required project files into a clean
    anonymized directory.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$AnonRepo,
    [switch]$DryRun,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SourceRepo = (Resolve-Path $SourceRepo).Path
if (-not [System.IO.Path]::IsPathRooted($AnonRepo)) {
    $AnonRepo = Join-Path (Get-Location).Path $AnonRepo
}
$AnonRepo = [System.IO.Path]::GetFullPath($AnonRepo)

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  BOPET: selective copy for anonymized release" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ("  source : {0}" -f $SourceRepo)
Write-Host ("  target : {0}" -f $AnonRepo)
Write-Host ("  dry run: {0}" -f $DryRun)
Write-Host ""

if (Test-Path $AnonRepo) {
    $existing = Get-ChildItem -Force $AnonRepo -ErrorAction SilentlyContinue
    if ($existing -and -not $Force) {
        throw "Target exists and is not empty: $AnonRepo. Use -Force."
    }
    if (-not $DryRun) {
        Remove-Item -Recurse -Force $AnonRepo
    }
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $AnonRepo | Out-Null
}

# --- Python modules ---
$pythonFiles = @(
    "scripts\run_baseline_comparison.py",
    "scripts\collect_kappa_mcc.py",
    "scripts\run_beta_sweep.py",
    "scripts\extract_sweep_results.py",
    "experiment_runner.py",
    "config\__init__.py",
    "config\base.py",
    "config\baseline_config.py",
    "config\config_loader.py",
    "config\evaluation.py",
    "config\experiment.py",
    "config\model.py",
    "config\peft.py",
    "config\task.py",
    "config\training.py",
    "src\data\dataloader.py",
    "src\models\peft_factory.py",
    "src\models\progressive_peft.py",
    "src\training\trainer.py",
    "src\evaluation\evaluator.py",
    "src\evaluation\energy_meter.py",
    "src\utils\hyperparameter_adapter.py",
    "src\utils\mlflow_enhancer.py",
    "src\utils\risk_mitigation.py"
)

# --- PowerShell reproduction scripts ---
$scriptFiles = @(
    "scripts\reproduce_all.ps1"
)

# --- PowerShell reproduction scripts ---
$analysisFiles = @(
    "scripts\analysis\README.md",
    "scripts\analysis\calibration.py",
    "scripts\analysis\calibration_measures.py",
    "scripts\analysis\compute_cohens_d.py",
    "scripts\analysis\compute_data_stats.py",
    "scripts\analysis\compute_raw_stats.py",
    "scripts\analysis\efficiency_plot.py",
    "scripts\analysis\extract_peak_memory_to_csv.py",
    "scripts\analysis\extract_sensitivity.py",          # NEW
    "scripts\analysis\extract_training_time_carbon.py",
    "scripts\analysis\generate_figures_from_json.py",
    "scripts\analysis\make_ablation_config.py",          # NEW
    "scripts\analysis\plot_overfitting_curves.py",
    "scripts\analysis\plot_rank_heatmap.py",
    "scripts\analysis\run_ablation.py",
    "scripts\analysis\run_all_ablations.ps1"             # NEW
)

# --- YAML configs ---
$yamlFiles = @(
    "configs\baselines\tiroberta_tidg_baseline.yaml",
    "configs\baselines\tiroberta_sentiment_baseline.yaml",
    "configs\baselines\tiroberta_tidg_sap_peft.yaml",
    "configs\baselines\afro_xlmr_tidg_baseline.yaml",
    "configs\baselines\afro_xlmr_sentiment_baseline.yaml",
    "configs\baselines\afro_xlmr_tidg_sap_peft.yaml",
    "configs\baselines\gpt2_tidg_baseline.yaml",
    "configs\baselines\gpt2_sentiment_baseline.yaml",
    "configs\baselines\gpt2_sentiment_vanilla_only.yaml",
    "configs\experiments\tiroberta_tidg_base.yaml",
    "configs\experiments\tiroberta_sentiment_base.yaml",
    "configs\experiments\afro_xlmr_tidg_base.yaml",
    "configs\experiments\afro_xlmr_sentiment_base.yaml",
    "configs\experiments\gpt2_tidg_base.yaml",
    "configs\experiments\gpt2_sentiment_base.yaml",
    "configs\baselines\tiroberta_tidg_vanilla_only.yaml",
    "configs\model\tiroberta.yaml",
    "configs\model\afro_xlmr.yaml",
    "configs\model\gpt2.yaml",
    "configs\task\sentiment.yaml",
    "configs\task\tidg.yaml",
    "configs\common\evaluation.yaml",
    "configs\common\training_defaults.yaml"
)

# --- Datasets ---
$datasetFiles = @(
    "data\processed\sentiment\train.csv",
    "data\processed\sentiment\dev.csv",
    "data\processed\sentiment\test.csv",
    "data\processed\tidg\train.csv",
    "data\processed\tidg\val.csv",
    "data\processed\tidg\test.csv"
)

# --- Package __init__ files ---
$initFiles = @(
    "src\__init__.py",
    "src\data\__init__.py",
    "src\models\__init__.py",
    "src\training\__init__.py",
    "src\evaluation\__init__.py",
    "src\utils\__init__.py"
)

$copied  = [System.Collections.Generic.List[string]]::new()
$missing = [System.Collections.Generic.List[string]]::new()
$created = [System.Collections.Generic.List[string]]::new()

function Copy-One {
    param([string]$RelPath)
    $src = Join-Path $SourceRepo $RelPath
    $dst = Join-Path $AnonRepo   $RelPath
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        $missing.Add($RelPath)
        Write-Host ("    MISSING: {0}" -f $RelPath) -ForegroundColor Red
        return
    }
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
    $copied.Add($RelPath)
    Write-Host ("    copied: {0}" -f $RelPath) -ForegroundColor Gray
}

function Ensure-Init {
    param([string]$RelPath)
    $src = Join-Path $SourceRepo $RelPath
    $dst = Join-Path $AnonRepo   $RelPath
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    }
    if (Test-Path -LiteralPath $src -PathType Leaf) {
        if (-not $DryRun) { Copy-Item -LiteralPath $src -Destination $dst -Force }
        $copied.Add($RelPath)
        Write-Host ("    copied: {0}" -f $RelPath) -ForegroundColor Gray
    } else {
        if (-not (Test-Path -LiteralPath $dst) -and -not $DryRun) {
            New-Item -ItemType File -Path $dst | Out-Null
        }
        $created.Add($RelPath)
        Write-Host ("    created: {0}" -f $RelPath) -ForegroundColor Gray
    }
}

Write-Host "[1/6] Python modules" -ForegroundColor Cyan
foreach ($f in $pythonFiles) { Copy-One $f }

Write-Host "[2/6] Reproduction scripts" -ForegroundColor Cyan
foreach ($f in $scriptFiles) { Copy-One $f }

Write-Host "[3/6] Analysis scripts" -ForegroundColor Cyan
foreach ($f in $analysisFiles) { Copy-One $f }

Write-Host "[4/6] YAML configs" -ForegroundColor Cyan
foreach ($f in $yamlFiles) { Copy-One $f }

Write-Host "[5/6] Datasets" -ForegroundColor Cyan
foreach ($f in $datasetFiles) { Copy-One $f }

Write-Host "[6/6] Package __init__.py" -ForegroundColor Cyan
foreach ($f in $initFiles) { Ensure-Init $f }

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ("  copied  : {0}" -f $copied.Count)
Write-Host ("  created : {0}" -f $created.Count)
Write-Host ("  missing : {0}" -f $missing.Count)

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "MISSING FILES:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host ("  - {0}" -f $m) -ForegroundColor Red }
    exit 1
}

Write-Host ""
Write-Host "DONE." -ForegroundColor Green