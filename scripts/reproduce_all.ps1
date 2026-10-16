<#
.SYNOPSIS
    Reproduce all reported experiments sequentially.
.DESCRIPTION
    Runs every baseline configuration in the paper with the standard
    five seeds. Estimated wall time on an RTX 4090: 24-30 h for the
    full set. On CPU it will be substantially slower.
.EXAMPLE
    .\scripts\reproduce_all.ps1
#>

[CmdletBinding()]
param(
    [string]$Seeds = "42,43,44,45,46"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tables = @(
    @{ Name = "TiroBERTa news";
       Config = "configs\baselines\tiroberta_tidg_baseline.yaml";
       Out = "results\baselines\tiroberta_tidg" },
    @{ Name = "TiroBERTa sentiment";
       Config = "configs\baselines\tiroberta_sentiment_baseline.yaml";
       Out = "results\baselines\tiroberta_sentiment" },
    @{ Name = "Afro-XLM-R news";
       Config = "configs\baselines\afro_xlmr_tidg_baseline.yaml";
       Out = "results\baselines\afro_xlmr_tidg" },
    @{ Name = "Afro-XLM-R sentiment";
       Config = "configs\baselines\afro_xlmr_sentiment_baseline.yaml";
       Out = "results\baselines\afro_xlmr_sentiment" },
    @{ Name = "GPT-2 news";
       Config = "configs\baselines\gpt2_tidg_baseline.yaml";
       Out = "results\baselines\gpt2_tidg" },
    @{ Name = "GPT-2 sentiment";
       Config = "configs\baselines\gpt2_sentiment_baseline.yaml";
       Out = "results\baselines\gpt2_sentiment" },
    @{ Name = "SAP-PEFT ablation";
       Config = "configs\baselines\tiroberta_tidg_sap_peft.yaml";
       Out = "results\baselines\tiroberta_tidg_ablation" }
)

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Reproducing all reported experiments" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ("  seeds  : {0}" -f $Seeds)
Write-Host ("  tables : {0}" -f $tables.Count)
Write-Host ""

$totalSw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($t in $tables) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ("Running {0}" -f $t.Name) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan

    if (-not (Test-Path -LiteralPath $t.Config)) {
        Write-Host ("  SKIP - config missing: {0}" -f $t.Config) -ForegroundColor Yellow
        continue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    python scripts\run_baseline_comparison.py `
        --config $t.Config `
        --sequential `
        --seeds $Seeds `
        --output-dir $t.Out

    $sw.Stop()
    Write-Host ("  {0} finished in {1:hh\:mm\:ss}" -f $t.Name, $sw.Elapsed) -ForegroundColor Green
}

$totalSw.Stop()

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ("All experiments complete in {0:hh\:mm\:ss}" -f $totalSw.Elapsed) -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Outputs written under results\baselines\"
Write-Host "To regenerate the kappa and MCC supplementary table, run:"
Write-Host "  python scripts\collect_kappa_mcc.py --experiments-root experiments --output kappa_mcc.csv --tex kappa_mcc.tex"