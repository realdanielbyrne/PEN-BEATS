<#
.SYNOPSIS
    Sequential single-GPU launcher for smol_replacement_paper variants on Windows.

.DESCRIPTION
    Reads the experiment YAML, sets D-drive artifact paths via PELLM_DATA_ROOT and
    related environment variables, then runs each pending variant one at a time using
    `accelerate launch --num_processes 1`.

    Completion is detected by the presence of
        D:\pellm\trainedmodels\<exp>\<variant>\training_manifest.json

    Per-variant stdout/stderr is tee'd to
        D:\pellm\runs\<exp>\<variant>\windows_launcher.log

.PARAMETER Config
    Path to the YAML experiment config.
    Default: scripts\experiments\smol_replacement_paper_windows.yaml

.PARAMETER Only
    Run only these variant names (space-separated). Overrides completion-skip.

.PARAMETER Skip
    Variant names to skip even if not yet complete.

.PARAMETER RestartVariant
    Clear all outputs for these variants and rerun them.

.PARAMETER ForceVariant
    Rerun these variants even if already complete (no clear).

.PARAMETER TokenBudget
    Override training.token_budget (passed through to pretrain_smol_replacement.py).

.PARAMETER DryRun
    Print the planned schedule and commands without launching anything.

.EXAMPLE
    # Run all pending variants
    .\scripts\run_smol_replacement_windows.ps1

    # Dry-run preview
    .\scripts\run_smol_replacement_windows.ps1 -DryRun

    # Run only specific variants
    .\scripts\run_smol_replacement_windows.ps1 -Only baseline,ae_mlp_512

    # Restart a variant from scratch
    .\scripts\run_smol_replacement_windows.ps1 -RestartVariant baseline
#>

[CmdletBinding()]
param(
    [string]$Config = "scripts\experiments\smol_replacement_paper_windows.yaml",
    [string[]]$Only      = @(),
    [string[]]$Skip      = @(),
    [string[]]$RestartVariant = @(),
    [string[]]$ForceVariant   = @(),
    [long]$TokenBudget   = 0,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$RepoRoot   = Split-Path -Parent $PSScriptRoot
$PythonBin  = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Accelerate = Join-Path $RepoRoot ".venv\Scripts\accelerate.exe"
$PretrainScript = Join-Path $RepoRoot "scripts\pretrain_smol_replacement.py"
$ConfigPath = Join-Path $RepoRoot $Config

foreach ($required in $PythonBin, $PretrainScript, $ConfigPath) {
    if (-not (Test-Path $required)) {
        Write-Error "Required path not found: $required"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# D-drive artifact environment (mirrors artifact_paths.py defaults)
# ---------------------------------------------------------------------------
$DataRoot = "D:\pellm"

$env:PELLM_DATA_ROOT    = $DataRoot
$env:HF_HOME            = "$DataRoot\hf_home"
$env:HF_DATASETS_CACHE  = "$DataRoot\datasets\hf_cache"
$env:TRANSFORMERS_CACHE = "$DataRoot\datasets\transformers_cache"
$env:TMPDIR             = "$DataRoot\tmp"

$TrainedModelsDir = "$DataRoot\trainedmodels"
$RunsDir          = "$DataRoot\runs"

# Override env from the YAML experiment.env block if present (parsed below).

# ---------------------------------------------------------------------------
# Parse YAML to get experiment name and variant list
# Uses the .venv Python (which has pyyaml from the experiments extra).
# ---------------------------------------------------------------------------
$YamlParseScript = @"
import sys, json
import yaml
with open(sys.argv[1], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(json.dumps({
    'name': cfg['experiment']['name'],
    'variants': list(cfg.get('variants', {}).keys()),
    'env': cfg['experiment'].get('env', {}),
}))
"@

$parsed = & $PythonBin -c $YamlParseScript $ConfigPath | ConvertFrom-Json
$ExpName  = $parsed.name
$AllVariants = $parsed.variants

# Apply any env overrides declared in the YAML experiment.env block.
foreach ($kv in $parsed.env.PSObject.Properties) {
    [System.Environment]::SetEnvironmentVariable($kv.Name, $kv.Value, "Process")
    Write-Verbose "env $($kv.Name)=$($kv.Value)"
}

Write-Host "Experiment : $ExpName"
Write-Host "Config     : $ConfigPath"
Write-Host "Data root  : $($env:PELLM_DATA_ROOT)"
Write-Host ""

# ---------------------------------------------------------------------------
# Helper: check if a variant has completed training
# ---------------------------------------------------------------------------
function Is-Complete([string]$variant) {
    $manifest = "$TrainedModelsDir\$ExpName\$variant\training_manifest.json"
    return Test-Path $manifest
}

# ---------------------------------------------------------------------------
# Helper: clear all output dirs for a variant
# ---------------------------------------------------------------------------
function Clear-Variant([string]$variant) {
    $dirs = @(
        "$TrainedModelsDir\$ExpName\$variant",
        "$DataRoot\checkpoints\$ExpName\$variant",
        "$RunsDir\$ExpName\$variant",
        "$DataRoot\evals\$ExpName\$variant"
    )
    $cleared = 0
    foreach ($d in $dirs) {
        if (Test-Path $d) {
            Remove-Item -Recurse -Force $d
            $cleared++
        }
    }
    Write-Host "  cleared $variant: $cleared dirs removed"
}

# ---------------------------------------------------------------------------
# Select variants to run
# ---------------------------------------------------------------------------
if ($Only.Count -gt 0) {
    $unknown = $Only | Where-Object { $AllVariants -notcontains $_ }
    if ($unknown) { Write-Error "Unknown variant(s) in -Only: $($unknown -join ', ')"; exit 1 }
    $candidates = $Only
} else {
    $candidates = $AllVariants
}

$restartSet = [System.Collections.Generic.HashSet[string]]$RestartVariant
$forceSet   = [System.Collections.Generic.HashSet[string]]($ForceVariant + $RestartVariant)
$skipSet    = [System.Collections.Generic.HashSet[string]]$Skip

if ($RestartVariant.Count -gt 0 -and -not $DryRun) {
    foreach ($v in $RestartVariant) {
        Clear-Variant $v
    }
}

$selected = [System.Collections.Generic.List[string]]@()
foreach ($v in $candidates) {
    if ($skipSet.Contains($v)) {
        Write-Host "  skip  $v  (--Skip)"
        continue
    }
    if ($forceSet.Contains($v) -or $Only.Count -gt 0) {
        $selected.Add($v)
        continue
    }
    if (Is-Complete $v) {
        Write-Host "  skip  $v  (already complete)"
        continue
    }
    $selected.Add($v)
}

if ($selected.Count -eq 0) {
    Write-Host "Nothing to run."
    exit 0
}

Write-Host "Pending variants ($($selected.Count)): $($selected -join ', ')"
Write-Host ""

# ---------------------------------------------------------------------------
# Build the accelerate command for a variant
# ---------------------------------------------------------------------------
function Build-Command([string]$variant) {
    $cmd = @(
        $Accelerate,
        "launch",
        "--num_processes", "1",
        $PretrainScript,
        $ConfigPath,
        "--variant", $variant
    )
    if ($TokenBudget -gt 0) {
        $cmd += @("--token-budget", "$TokenBudget")
    }
    return $cmd
}

# ---------------------------------------------------------------------------
# Dry-run: just print what would be run
# ---------------------------------------------------------------------------
if ($DryRun) {
    foreach ($v in $selected) {
        $cmd = Build-Command $v
        $logPath = "$RunsDir\$ExpName\$v\windows_launcher.log"
        Write-Host "--- $v ---"
        Write-Host "  $($cmd -join ' ')"
        Write-Host "  # log: $logPath"
        Write-Host ""
    }
    exit 0
}

# ---------------------------------------------------------------------------
# Sequential run
# ---------------------------------------------------------------------------
$overallStart = Get-Date
$failures = [System.Collections.Generic.List[string]]@()

foreach ($v in $selected) {
    $logDir  = "$RunsDir\$ExpName\$v"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logPath = "$logDir\windows_launcher.log"

    $cmd = Build-Command $v
    $exe  = $cmd[0]
    $args = $cmd[1..($cmd.Length - 1)]

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $header = "`n=== launcher $timestamp variant=$v ===`n"
    Add-Content -Path $logPath -Value $header

    Write-Host "=== variant: $v ===" -ForegroundColor Cyan
    Write-Host "  command : $($cmd -join ' ')"
    Write-Host "  log     : $logPath"
    Write-Host ""

    $variantStart = Get-Date

    # Run the variant, tee-ing output to the log file and the console.
    & $exe @args 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $variantStart
    $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed

    if ($exitCode -ne 0) {
        Write-Host "`n  FAILED (exit $exitCode) in $elapsedStr" -ForegroundColor Red
        $failures.Add($v)
        # Stop on failure — later variants may depend on earlier ones.
        break
    } else {
        Write-Host "`n  OK in $elapsedStr" -ForegroundColor Green
    }
    Write-Host ""
}

$totalElapsed = (Get-Date) - $overallStart
$totalStr = "{0:hh\:mm\:ss}" -f $totalElapsed

Write-Host "=== done in $totalStr ===" -ForegroundColor Cyan
if ($failures.Count -gt 0) {
    Write-Host "FAILURES: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
exit 0
