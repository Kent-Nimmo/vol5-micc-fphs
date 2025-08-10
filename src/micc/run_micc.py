"""MICC simulation orchestrator.

This script drives the measurement‑induced context collapse (MICC)
simulation across a grid of parameter combinations specified in a
YAML configuration file.  For each condition (gauge group, lattice
size, inverse coupling ``b``, pivot slope ``kappa``, measurement
schedule, readout fraction ``f`` and random seed) it performs the
following steps:

1. Generate or load flip counts for the given lattice and seed.
2. Compute the logistic fractal dimension ``D(n)`` with slope
   ``kappa``, then the pivot weight ``g(D)`` and multiply by ``b``.
3. Build a constant diagonal kernel for the gauge group and
   exponentiate the gauge potentials to obtain link matrices ``U``.
4. Apply the dephasing channel according to the schedule and
   readout fraction.
5. Measure Wilson loops and estimate the string tension ``σ``.
6. Measure the two‑path interferometer and extract the visibility ``V``.
7. Compute coherence proxies ``C`` and ``P_entropy``.
8. Serialize the results to a JSON file and record hashes of
   intermediate artifacts for provenance.

Aggregate statistics and plots are produced by separate scripts (see
``scripts/aggregate.py``).  This script is intentionally lightweight
and does not perform any statistical analysis; its sole responsibility
is to generate per‑condition data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

from .utils_fphs import (
    load_flip_counts,
    logistic_dimension,
    pivot_weight,
    build_constant_kernel,
    build_kernel_from_rhos,
    compute_kernel_eigs,
    exponentiate_potential,
    compute_sha256_of_array,
    compute_sha256_of_obj,
)
from .measurement import apply_dephasing
from .loops import measure_wilson_loops, estimate_string_tension
from .interference import build_twist_scan, fit_visibility
from .purity import compute_coherence_factor, compute_spectral_entropy


def run_condition(cfg: Dict[str, Any], gauge: str, L: int, b: float, kappa: float, schedule: str, f: float, seed: int, out_dir: Path, rhos: np.ndarray) -> None:
    """Run a single MICC simulation condition and write JSON output.

    Parameters
    ----------
    cfg : dict
        Configuration mapping loaded from YAML.  Only ``pivot`` keys
        are used here.
    gauge : str
        Gauge group name (e.g. 'SU2', 'SU3').
    L : int
        Lattice size.
    b : float
        Inverse coupling (scales the pivot weight).
    kappa : float
        Pivot slope parameter (logistic slope).
    schedule : str
        Measurement schedule ('pre', 'during', 'post').
    f : float
        Measurement fraction in [0,1].
    seed : int
        Random seed.
    out_dir : Path
        Directory in which to write the JSON output.
    """
    # Create RNG seeded deterministically per condition
    base_rng = np.random.default_rng(seed)
    # 1. Flip counts
    flip_counts = load_flip_counts(L, seed)
    flip_counts_hash = compute_sha256_of_array(flip_counts)
    num_links = flip_counts.size
    # 2. Fractal dimension and pivot weight
    D = logistic_dimension(flip_counts, kappa, n0=cfg.get('pivot', {}).get('logistic_n0', 0.0))
    gD = pivot_weight(D) * b
    pivot_params = {
        'kappa': kappa,
        'n0': cfg.get('pivot', {}).get('logistic_n0', 0.0),
        'a': 1/3.0,
        'b_coupling': b,
    }
    pivot_hash = compute_sha256_of_obj(pivot_params)
    # 3. Kernel and U
    # Build a per‑link kernel using the spectral radii.  For each gauge
    # group the generator is scaled by the tiled ρ values.  The
    # resulting kernel has shape (num_links, N, N).
    kernel = build_kernel_from_rhos(rhos, L, gauge)
    kernel_hash = compute_sha256_of_array(kernel)
    U = exponentiate_potential(gD, kernel)
    # For 'pre' and 'during' schedules we apply dephasing before measurement
    if schedule in ('pre', 'during') and f > 0.0:
        U_meas, measured_count = apply_dephasing(U, f, gauge, base_rng)
    else:
        U_meas = U.copy()
        measured_count = 0
    # 4. Wilson loops and string tension (for 'post' schedule measurement does not affect loops)
    wloops = measure_wilson_loops(U_meas if schedule != 'post' else U, L)
    sigma, sigma_err = estimate_string_tension(wloops)
    # 5. Interference and visibility
    phi_values, I_values = build_twist_scan(U_meas if schedule != 'post' else U, L, gauge)
    vis_fit = fit_visibility(phi_values, I_values)
    I0 = vis_fit['I0']
    A = vis_fit['A']
    # The visibility is defined as A / I0.  Use the absolute value
    # of the baseline to avoid negative visibilities when I0 < 0.
    visibility = A / abs(I0) if I0 != 0 else 0.0
    # 6. Coherence proxies (compute on measured state for all schedules except 'post')
    U_for_purity = U_meas if schedule != 'post' else U
    C_factor = compute_coherence_factor(U_for_purity)
    P_entropy = compute_spectral_entropy(U_for_purity)
    # 7. Build record
    record = {
        'sim': 'MICC',
        'gauge': gauge,
        'rep': 'fundamental',
        'L': L,
        'b': b,
        'kappa': kappa,
        'seed': seed,
        'measurement_model': 'dephasing',
        'f': f,
        'schedule': schedule,
        'sigma': sigma,
        'sigma_err': sigma_err,
        'visibility': visibility,
        'visibility_err': 0.0,  # visibility fit error not estimated
        'coherence': {
            'C': C_factor,
            'P_entropy': P_entropy,
        },
        'twist_scan': {
            'phi_values': phi_values.tolist(),
            'fit': vis_fit,
        },
        'diagnostics': {
            'measured_link_count': measured_count,
            'selection_policy': 'uniform',
        },
        'artifacts': {
            'kernel_hash': kernel_hash,
            'flip_counts_hash': flip_counts_hash,
            'pivot_fit_hash': pivot_hash,
            'config_hash': '',  # filled later
        },
    }
    # Write JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{gauge}_L{L}_b{b}_k{kappa}_seed{seed}_{schedule}_f{f}.json"
    out_path = out_dir / fname
    with out_path.open('w') as f_json:
        json.dump(record, f_json, indent=2)
    print(f"Wrote {out_path}")


def main(config_path: str = 'configs/fphs_anchors.yaml', output_dir: str = 'runs') -> None:
    """Main entry point for the MICC simulation.

    Reads the YAML configuration at ``config_path``, iterates over all
    parameter combinations, and writes one JSON file per condition to
    ``output_dir``.
    """
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(output_dir)
    # Load the D_values table and compute spectral radii once.  The
    # D_values_path in the config is relative to the repository root.
    dval_path = cfg.get('D_values_path')
    if dval_path is None:
        raise KeyError('D_values_path missing from configuration')
    # Resolve path relative to the config file location
    # If the D_values path is absolute, use it directly.  Otherwise
    # resolve relative to the project root (one directory above the
    # config file).  This allows ``D_values_path: data/D_values.csv``
    # in the YAML to locate ``micc-fphs/data/D_values.csv`` correctly.
    dval_path_obj = Path(dval_path)
    if dval_path_obj.is_absolute():
        dval_file = dval_path_obj
    else:
        # repository root is one level above the configs directory
        repo_root = config_path.parent.parent
        dval_file = (repo_root / dval_path_obj).resolve()
    import csv
    D_vals = []
    with open(dval_file, 'r') as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            D_vals.append(float(row['D']))
    D_vals = np.array(D_vals, dtype=float)
    rhos = compute_kernel_eigs(D_vals)
    # Derive parameter lists
    lattices = cfg.get('lattice_sizes', [])
    couplings = cfg.get('inverse_couplings', [])
    kappas = cfg.get('pivot_slopes', [])
    gauges = cfg.get('gauge_groups', [])
    f_grid = cfg.get('f_grid', {})
    seeds = cfg.get('seeds', [])
    # Precompute config hash (for provenance)
    config_hash = compute_sha256_of_obj(cfg)
    # Iterate over all parameter combinations
    for gauge in gauges:
        for L in lattices:
            for b in couplings:
                for kappa in kappas:
                    for seed in seeds:
                        for schedule, f_list in f_grid.items():
                            for f_val in f_list:
                                run_condition(cfg, gauge, L, float(b), float(kappa), schedule, float(f_val), int(seed), out_dir, rhos)
    # After generating all JSON files, insert config_hash into each artifact
    for json_file in out_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        data['artifacts']['config_hash'] = config_hash
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run MICC simulation over parameter grid')
    parser.add_argument('--config', default='configs/fphs_anchors.yaml', help='Path to configuration YAML')
    parser.add_argument('--output', default='runs', help='Output directory for per‑condition JSON files')
    args = parser.parse_args()
    main(args.config, args.output)