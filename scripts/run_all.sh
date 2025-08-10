#!/usr/bin/env bash

# Run the full MICC simulation over the parameter grid and aggregate
# results into summary tables.  This script assumes that the
# micc-fphs package has been installed into the current Python
# environment or is available on the PYTHONPATH.

set -euo pipefail

# Navigate to the repository root (directory containing this script)
REPO_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )
cd "$REPO_ROOT"

CONFIG_PATH="configs/fphs_anchors.yaml"
OUTPUT_DIR="runs"
CSV_PATH="results/summary.csv"
PARQUET_PATH="results/summary.parquet"

# Set PYTHONPATH so that the micc package under src/ is discoverable
export PYTHONPATH="$REPO_ROOT/src"

# Suppress the per‑condition "Wrote" messages by redirecting
# stdout to /dev/null.  Without this redirection the volume of
# printed lines (hundreds of conditions) can overwhelm the
# container's output buffer.
python3 -m micc.run_micc --config "$CONFIG_PATH" --output "$OUTPUT_DIR" > /dev/null
python3 scripts/aggregate.py --input "$OUTPUT_DIR" --csv "$CSV_PATH" --parquet "$PARQUET_PATH" > /dev/null

echo "Simulation and aggregation complete. Results written to $CSV_PATH and $PARQUET_PATH."