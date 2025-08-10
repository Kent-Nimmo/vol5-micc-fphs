#!/usr/bin/env python3
"""Aggregate MICC simulation outputs into summary tables.

This script scans a directory of per‑condition JSON files (as
produced by ``run_micc.py``), combines them into a single pandas
DataFrame, and writes the result to CSV and Parquet formats.  The
resulting table includes one row per JSON file with nested structures
flattened into top‑level columns.  For example, the coherence proxy
dictionary ``{'C': c, 'P_entropy': p}`` becomes columns ``coherence_C``
and ``coherence_P_entropy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def flatten_record(data: dict) -> dict:
    """Flatten nested dictionaries for tabular output."""
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = flatten_record(value)
            for nk, nv in nested.items():
                flat[f"{key}_{nk}"] = nv
        else:
            flat[key] = value
    return flat


def main(input_dir: str = 'runs', output_csv: str = 'results/summary.csv', output_parquet: str = 'results/summary.parquet') -> None:
    input_dir = Path(input_dir)
    # Collect JSON files
    records = []
    for json_file in input_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        flat = flatten_record(data)
        records.append(flat)
    if not records:
        print(f"No JSON files found in {input_dir}")
        return
    df = pd.DataFrame(records)
    # Ensure results directory exists
    out_csv = Path(output_csv)
    out_parquet = Path(output_parquet)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV and Parquet
    df.to_csv(out_csv, index=False)
    # Attempt to write Parquet if pyarrow or fastparquet is available
    try:
        df.to_parquet(out_parquet, index=False)
        wrote_parquet = True
    except Exception as e:
        wrote_parquet = False
        print(f"Warning: unable to write Parquet file ({e}). Only CSV will be generated.")
    if wrote_parquet:
        print(f"Wrote summary to {out_csv} and {out_parquet}")
    else:
        print(f"Wrote summary to {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate MICC JSON outputs into CSV and Parquet')
    parser.add_argument('--input', default='runs', help='Directory containing JSON files')
    parser.add_argument('--csv', default='results/summary.csv', help='Output CSV path')
    parser.add_argument('--parquet', default='results/summary.parquet', help='Output Parquet path')
    args = parser.parse_args()
    main(args.input, args.csv, args.parquet)