#!/usr/bin/env python3
"""Aggregate MICC simulation outputs into summary tables (expanded).

What’s new:
- Robust flattening of nested dicts (unchanged core).
- Convenience columns for visibility (Fourier, fit, R2) and sigma.
- Graceful handling of older JSONs (missing keys).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def flatten_record(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries for tabular output (depth-first)."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        k = f"{prefix}{key}" if prefix == "" else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(flatten_record(value, prefix=k))
        else:
            flat[k] = value
    return flat


def _derive_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add convenience/compat columns from the flattened dict."""
    out: Dict[str, Any] = {}

    # Visibility conveniences
    vis = row.get("visibility", None)
    out["visibility"] = vis

    # If twist_scan.fit exists, pull details
    vfit = row.get("twist_scan_fit_visibility", None)
    vfit_four = row.get("twist_scan_fit_visibility_fourier", None)
    vfit_cos = row.get("twist_scan_fit_visibility_fit", None)
    vfit_R2 = row.get("twist_scan_fit_R2", None)
    vfit_A = row.get("twist_scan_fit_A", None)
    vfit_I0 = row.get("twist_scan_fit_I0", None)

    # Prefer Fourier visibility as the canonical one if present
    out["visibility_fourier"] = vfit_four if vfit_four is not None else vis
    out["visibility_fit"] = vfit_cos
    out["visibility_R2"] = vfit_R2
    out["visibility_A"] = vfit_A
    out["visibility_I0"] = vfit_I0

    # Sigma conveniences (backward compatible)
    # Current schema has top-level 'sigma' float; future may nest.
    sigma = row.get("sigma", None)
    if isinstance(sigma, dict):
        out["sigma_creutz"] = sigma.get("creutz_22", None)
        out["sigma_global"] = sigma.get("global", None)
        out["sigma_err"] = sigma.get("err", None)
    else:
        out["sigma_creutz"] = sigma
        out["sigma_global"] = row.get("sigma_global", None)
        out["sigma_err"] = row.get("sigma_err", None)

    # Coherence proxies (names are stable)
    out["P_entropy"] = row.get("coherence_P_entropy", None)
    out["C_factor"] = row.get("coherence_C", None)

    # Diagnostics bits (if present)
    out["valid_anchor_fraction"] = row.get("diagnostics_valid_anchor_fraction", None)
    out["measured_link_count"] = row.get("diagnostics_measured_link_count", None)

    return out


def main(input_dir: str = 'runs', output_csv: str = 'results/summary.csv', output_parquet: str = 'results/summary.parquet') -> None:
    input_dir = Path(input_dir)
    records = []

    for json_file in input_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        flat = flatten_record(data)
        # Derive convenience columns
        flat.update(_derive_columns(flat))
        records.append(flat)

    if not records:
        print(f"No JSON files found in {input_dir}")
        return

    df = pd.DataFrame(records)

    # Ensure results directory exists
    out_csv = Path(output_csv); out_parquet = Path(output_parquet)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV and Parquet
    df.to_csv(out_csv, index=False)
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
