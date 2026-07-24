#!/usr/bin/env bash
# Reproduction script for "Graph Neural Networks for Macroeconomic Forecasting"
# Author: Rayyan Asim (mrayanasim09@gmail.com)
#
# This script sequentially executes the full V2.1 benchmark pipeline.
# Expected runtime: ~12 hours on Apple Silicon, longer on other hardware.

set -euo pipefail

echo "==========================================================="
echo " Reproduction Pipeline: GNNs for Macroeconomic Forecasting"
echo "==========================================================="
echo ""

# Verify Python version (requires Python 3.10+)
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$PY_VERSION < 3.10" | bc -l) )); then
    echo "Error: Python 3.10 or higher is required (found $PY_VERSION)."
    exit 1
fi

# Set up virtual environment
VENV_DIR="./.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/2] Creating virtual environment at $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

echo "[2/2] Installing requirements..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo ""
echo "Starting benchmark pipeline..."
echo "Commit: a7fef297346868ddf2ae178e45d47ab27cc55317"
echo ""

# Ensure results directory exists
mkdir -p experiments/results/v2_1

# Step 1: Validate contract
echo ">>> [1/6] Validating v2.1 contract..."
"$VENV_DIR/bin/python" -m src.models.validate_v2_1_contract

# Step 2: Run benchmark engine
echo ">>> [2/6] Running benchmark engine (this will take several hours)..."
"$VENV_DIR/bin/python" -m src.models.run_benchmark_engine_v2_1 --execute 2>&1 | tee experiments/results/v2_1/execution_log.txt

# Step 3: Validate results
echo ">>> [3/6] Validating benchmark results..."
"$VENV_DIR/bin/python" -m src.models.validate_v2_1_results

# Step 4: Analyze results
echo ">>> [4/6] Analyzing results..."
"$VENV_DIR/bin/python" -m src.models.analyze_v2_1_results

# Step 5: Generate manuscript
echo ">>> [5/6] Generating manuscript..."
"$VENV_DIR/bin/python" -m src.models.generate_v2_1_manuscript

# Step 6: Generate report
echo ">>> [6/6] Generating summary report..."
"$VENV_DIR/bin/python" -m src.models.generate_v2_1_report

echo ""
echo "==========================================================="
echo " Pipeline Execution Complete"
echo "==========================================================="
echo ""
echo "Verifying output hashes..."

# Verification function
verify_hash() {
    local file=$1
    local expected=$2
    if [ ! -f "$file" ]; then
        echo "FAIL: $file not found."
        return 1
    fi
    
    local actual
    if command -v shasum &> /dev/null; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    elif command -v sha256sum &> /dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    else
        echo "SKIP: shasum/sha256sum not found."
        return 0
    fi
    
    if [ "$actual" == "$expected" ]; then
        echo "PASS: $file matches expected hash."
    else
        echo "FAIL: Hash mismatch for $file"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
    fi
}

verify_hash "experiments/results/v2_1/forecasts.parquet" "f988dfb1249fd77739ffdecae6323073c8cbd363c48412d4e3248454f98b3798"
verify_hash "experiments/results/v2_1/metrics.parquet" "231b349f6ab2d5bba4cc42cf047c2cedd88300f5836302b5e79e4ff9071abcaa"
verify_hash "experiments/results/v2_1/dm_tests.parquet" "a57490a079580c48458143961f9e8dde6f7cf72e77818a71246d73496c397048"

echo ""
echo "Reproduction finished successfully."
