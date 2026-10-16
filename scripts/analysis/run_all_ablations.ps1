<#
.SYNOPSIS
    Run ablation sweeps for SAP-PEFT on one or more backbones.

.EXAMPLE
    .\run_all_ablations.ps1 -Backbone tiroberta_tidg

.EXAMPLE
    .\run_all_ablations.ps1 -Backbone tiroberta_tidg -DryRun
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("tiroberta_tidg","afro_xlmr_tidg","gpt2_tidg")]
    [string]$Backbone,

    [string[]]$Axes = @(),         # empty = all axes

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$configMap = @{
    "tiroberta_tidg"   = "configs\baselines\tiroberta_tidg_sap_peft.yaml"
    "afro_xlmr_tidg"   = "configs\baselines\afro_xlmr_tidg_sap_peft.yaml"
    "gpt2_tidg"        = "configs\baselines\gpt2_tidg_baseline.yaml"
}

$config = $configMap[$Backbone]
if (-not (Test-Path $config)) {
    throw "Config not found: $config"
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ("  SAP-PEFT ablation sweep: {0}" -f $Backbone) -ForegroundColor Cyan
Write-Host ("  Base config: {0}" -f $config)
Write-Host ("  Dry run:     {0}" -f $DryRun)
Write-Host "=================================================================" -ForegroundColor Cyan

if ($Axes.Count -eq 0) {
    python scripts\analysis\run_ablation.py `
        --config $config `
        $(if ($DryRun) { "--dry-run" })
} else {
    foreach ($axis in $Axes) {
        python scripts\analysis\run_ablation.py `
            --config $config `
            --param $axis `
            $(if ($DryRun) { "--dry-run" })
    }
}

Write-Host ""
Write-Host "Done. See ablation_tmp_configs\ for generated YAMLs." -ForegroundColor Green