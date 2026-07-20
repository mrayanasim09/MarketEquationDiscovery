<#
.SYNOPSIS
Reproduction script for "Graph Neural Networks for Macroeconomic Forecasting"

.DESCRIPTION
This script sequentially executes the full V2.1 benchmark pipeline on Windows.
Expected runtime: ~12 hours on Apple Silicon, longer on other hardware.
Author: Rayyan Asim (mrayanasim09@gmail.com)
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================================="
Write-Host " Reprodution Pipeline: GNNs for Macroeconomic Forecasting"
Write-Host "==========================================================="
Write-Host ""

# Verify Python
$PythonCmd = "python"
if (-Not (Get-Command $PythonCmd -ErrorAction SilentlyContinue)) {
    Write-Error "Error: python is not installed or not in PATH."
    exit 1
}

$PyVersionStr = & $PythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$PyVersion = [version]"$PyVersionStr.0"
if ($PyVersion -lt [version]"3.10.0") {
    Write-Error "Error: Python 3.10 or higher is required (found $PyVersionStr)."
    exit 1
}

# Set up virtual environment
$VenvDir = "..\.venv"
if (-Not (Test-Path $VenvDir)) {
    Write-Host "[1/2] Creating virtual environment at $VenvDir..."
    & $PythonCmd -m venv $VenvDir
}

Write-Host "[2/2] Installing requirements..."
$PythonExec = Join-Path $VenvDir "Scripts\python.exe"
# Fallback to Unix-style if running PWSH on macOS/Linux
if (-Not (Test-Path $PythonExec)) {
    $PythonExec = Join-Path $VenvDir "bin\python"
}

& $PythonExec -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
    & $PythonExec -m pip install -r requirements.txt
} else {
    Write-Host "Warning: requirements.txt not found." -ForegroundColor Yellow
}

Write-Host "`nStarting benchmark pipeline..."
Write-Host "Commit: a7fef297346868ddf2ae178e45d47ab27cc55317`n"

# Ensure results directory exists
$ResultsDir = "experiments\results\v2_1"
if (-Not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
}

# Step 1: Validate contract
Write-Host ">>> [1/6] Validating v2.1 contract..." -ForegroundColor Cyan
& $PythonExec -m src.models.validate_v2_1_contract

# Step 2: Run benchmark engine
Write-Host ">>> [2/6] Running benchmark engine (this will take several hours)..." -ForegroundColor Cyan
& $PythonExec -m src.models.run_benchmark_engine_v2_1 --execute 2>&1 | Tee-Object -FilePath "$ResultsDir\execution_log.txt"

# Step 3: Validate results
Write-Host ">>> [3/6] Validating benchmark results..." -ForegroundColor Cyan
& $PythonExec -m src.models.validate_v2_1_results

# Step 4: Analyze results
Write-Host ">>> [4/6] Analyzing results..." -ForegroundColor Cyan
& $PythonExec -m src.models.analyze_v2_1_results

# Step 5: Generate manuscript
Write-Host ">>> [5/6] Generating manuscript..." -ForegroundColor Cyan
& $PythonExec -m src.models.generate_v2_1_manuscript

# Step 6: Generate report
Write-Host ">>> [6/6] Generating summary report..." -ForegroundColor Cyan
& $PythonExec -m src.models.generate_v2_1_report

Write-Host "`n==========================================================="
Write-Host " Pipeline Execution Complete"
Write-Host "===========================================================`n"
Write-Host "Verifying output hashes..."

function Verify-Hash {
    param (
        [string]$FilePath,
        [string]$ExpectedHash
    )
    
    if (-Not (Test-Path $FilePath)) {
        Write-Host "FAIL: $FilePath not found." -ForegroundColor Red
        return
    }
    
    $ActualHash = (Get-FileHash -Algorithm SHA256 -Path $FilePath).Hash.ToLower()
    if ($ActualHash -eq $ExpectedHash.ToLower()) {
        Write-Host "PASS: $FilePath matches expected hash." -ForegroundColor Green
    } else {
        Write-Host "FAIL: Hash mismatch for $FilePath" -ForegroundColor Red
        Write-Host "  Expected: $ExpectedHash"
        Write-Host "  Actual:   $ActualHash"
    }
}

Verify-Hash "experiments\results\v2_1\forecasts.parquet" "f988dfb1249fd77739ffdecae6323073c8cbd363c48412d4e3248454f98b3798"
Verify-Hash "experiments\results\v2_1\metrics.parquet" "231b349f6ab2d5bba4cc42cf047c2cedd88300f5836302b5e79e4ff9071abcaa"
Verify-Hash "experiments\results\v2_1\dm_tests.parquet" "a57490a079580c48458143961f9e8dde6f7cf72e77818a71246d73496c397048"

Write-Host "`nReproduction finished successfully." -ForegroundColor Green
