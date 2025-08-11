# src/micc/run_micc.py
"""MICC simulation orchestrator (HYBRID update: during-measure loops/coherence + kick_passes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml

from .utils_fphs import (
    load_flip_counts,
    logistic_dimension,
    pivot_weight,
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


def _jackknife_err(values: List[float]) -> float:
    """Jackknife standard error of the mean for a 1D list."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean_all = float(np.mean(values))
    loo_means = []
    for i in range(n):
        loo = [values[j] for j in range(n) if j != i]
        loo_means.append(float(np.mean(loo)))
    var_jk = (n - 1) / n * float(np.sum((np.asarray(loo_means) - mean_all) ** 2))
    return float(np.sqrt(max(var_jk, 0.0)))


def run_condition(
    cfg: Dict[str, Any],
    gauge: str,
    L: int,
    b: float,
    kappa: float,
    schedule: str,
    f: float,
    seed: int,
    out_dir: Path,
    rhos: np.ndarray,
) -> None:
    """Run a single MICC simulation condition and write JSON output."""
    rng = np.random.default_rng(seed)

    # 1) Flip counts (hash for provenance)
    flip_counts = load_flip_counts(L, seed)
    flip_counts_hash = compute_sha256_of_array(flip_counts)
    num_links = flip_counts.size

    # 2) FPHS pivot → potential scale
    D = logistic_dimension(
        flip_counts,
        kappa,
        n0=cfg.get("pivot", {}).get("logistic_n0", 0.0),
    )
    gD = pivot_weight(D) * b
    pivot_params = {
        "kappa": kappa,
        "n0": cfg.get("pivot", {}).get("logistic_n0", 0.0),
        "a": 1 / 3.0,
        "b_coupling": b,
    }
    pivot_hash = compute_sha256_of_obj(pivot_params)

    # 3) Kernel from spectral radii → link field U
    kernel = build_kernel_from_rhos(rhos, L, gauge)
    kernel_hash = compute_sha256_of_array(kernel)
    U = exponentiate_potential(gD, kernel)

    # --- HYBRID knobs ---
    vis_cfg = cfg.get("visibility", {}) or {}
    num_phi = int(vis_cfg.get("num_phi", 64))
    cycles = int(vis_cfg.get("cycles", 8))
    eps_min = float(vis_cfg.get("eps", 1e-12))
    eps_scale = float(vis_cfg.get("eps_scale", 1e-6))

    meas_cfg = cfg.get("measurement", {}) or {}
    kick_passes = int(meas_cfg.get("kick_passes", 1))  # amplify dephasing impact

    loops_cfg = cfg.get("loops", {}) or {}
    loop_meas_cycles = int(loops_cfg.get("measurement_cycles", 1))  # avg σ/P over measured snapshots

    # Helper: apply dephasing multiple passes (used when schedule='during')
    def _apply_kicks(U_in: np.ndarray) -> np.ndarray:
        U_tmp = U_in
        for _ in range(max(kick_passes, 1)):
            U_tmp, _ = apply_dephasing(U_tmp, f, gauge, rng)
        return U_tmp

    # 4) Apply measurement channel according to schedule (pre/post baseline)
    if schedule == "pre" and f > 0.0:
        U_pre, measured_count = apply_dephasing(U, f, gauge, rng)
        base_for_obs = U_pre
    else:
        measured_count = 0
        base_for_obs = U

    # 5) Interference → visibility with multi‑scan averaging & WLS
    per_cycle_V: List[float] = []
    per_cycle_I0: List[float] = []
    scans: List[np.ndarray] = []
    phi_values = None

    for c in range(cycles):
        if schedule == "during" and f > 0.0:
            # fresh multi-pass kick snapshot each cycle
            U_cycle = _apply_kicks(U)
        elif schedule == "post":
            U_cycle = U  # measurement after observation
        else:
            U_cycle = base_for_obs  # pre or f=0

        phi_c, I_c = build_twist_scan(U_cycle, L, gauge, num_phi=num_phi, normalize=True)
        if phi_values is None:
            phi_values = phi_c
        scans.append(I_c)

        # Per-cycle quick visibility (unweighted) for jackknife
        fit_c = fit_visibility(phi_c, I_c, eps=eps_min, eps_scale=eps_scale, clip_to_unit=True)
        per_cycle_V.append(float(fit_c["visibility"]))
        per_cycle_I0.append(float(fit_c["I0"]))

    I_mat = np.vstack(scans)  # [cycles, num_phi]
    I_mean = I_mat.mean(axis=0)
    # Per-φ variance → weights for WLS (avoid zero)
    ddof = 1 if cycles > 1 else 0
    var_phi = I_mat.var(axis=0, ddof=ddof)
    weights = 1.0 / np.maximum(var_phi, 1e-12)

    vis_fit = fit_visibility(
        phi_values,
        I_mean,
        weights=weights,
        eps=eps_min,
        eps_scale=eps_scale,
        clip_to_unit=True,
    )
    visibility = float(vis_fit.get("visibility", 0.0))
    visibility_err = _jackknife_err(per_cycle_V)

    # 6) Wilson loops / σ and coherence proxies
    if schedule == "during" and f > 0.0:
        # Average over several *measured* snapshots to stabilize σ and P
        sig_list, c_list, p_list = [], [], []
        for _ in range(max(loop_meas_cycles, 1)):
            U_snap = _apply_kicks(U)
            wloops, samples = measure_wilson_loops(
                U_snap,
                L,
                sizes=((1, 1), (1, 2), (2, 1), (2, 2)),
                return_samples=True,
            )
            sig, _sig_err_i = estimate_string_tension(wloops, samples=samples)
            if np.isfinite(sig):
                sig_list.append(sig)
            c_list.append(compute_coherence_factor(U_snap))
            p_list.append(compute_spectral_entropy(U_snap))
        if len(sig_list) == 0:
            sigma, sigma_err = float("nan"), float("nan")
        else:
            sigma = float(np.mean(sig_list))
            sigma_err = float(np.std(sig_list, ddof=1) / max(len(sig_list) ** 0.5, 1.0))
        C_factor = float(np.mean(c_list)) if len(c_list) else float("nan")
        P_entropy = float(np.mean(p_list)) if len(p_list) else float("nan")
    else:
        # Old behavior for pre/post or f=0
        U_for_loops = base_for_obs if schedule != "post" else U
        wloops, samples = measure_wilson_loops(
            U_for_loops,
            L,
            sizes=((1, 1), (1, 2), (2, 1), (2, 2)),
            return_samples=True,
        )
        sigma, sigma_err = estimate_string_tension(wloops, samples=samples)
        C_factor = compute_coherence_factor(U_for_loops)
        P_entropy = compute_spectral_entropy(U_for_loops)

    # QC flags for visibility
    qc_visibility_ok = bool(vis_fit.get("R2", 0.0) >= 0.5)
    qc_visibility_unit = bool(0.0 <= visibility <= 1.0 + 1e-9)
    qc_visibility_i0_small = bool(abs(vis_fit.get("I0", 0.0)) < 1e-10)

    # 7) Serialize
    record = {
        "sim": "MICC",
        "gauge": gauge,
        "rep": "fundamental",
        "L": L,
        "b": b,
        "kappa": kappa,
        "seed": seed,
        "measurement_model": "dephasing",
        "f": f,
        "schedule": schedule,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "visibility": visibility,
        "visibility_err": visibility_err,
        "coherence": {"C": C_factor, "P_entropy": P_entropy},
        "twist_scan": {
            "phi_values": phi_values.tolist(),
            "mean_I": I_mean.tolist(),
            "fit": vis_fit,  # contains A, I0, phi0, R2, V_* and chosen 'visibility'
            "cycles": cycles,
            "per_cycle_V": per_cycle_V,
        },
        "diagnostics": {
            "measured_link_count": int(round(f * num_links)),
            "selection_policy": "uniform",
            "num_links": int(num_links),
            "visibility_qc": {
                "r2_ok": qc_visibility_ok,
                "v_in_unit_interval": qc_visibility_unit,
                "i0_very_small": qc_visibility_i0_small,
            },
            "hybrid_knobs": {
                "kick_passes": int(kick_passes),
                "loop_measurement_cycles": int(loop_meas_cycles),
            },
        },
        "artifacts": {
            "kernel_hash": kernel_hash,
            "flip_counts_hash": flip_counts_hash,
            "pivot_fit_hash": pivot_hash,
            "config_hash": "",  # filled after all files are written
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{gauge}_L{L}_b{b}_k{kappa}_seed{seed}_{schedule}_f{f}.json"
    with (out_dir / fname).open("w") as f_json:
        json.dump(record, f_json, indent=2)
    print(f"Wrote {out_dir / fname}")


def main(config_path: str = "configs/fphs_anchors.yaml", output_dir: str = "runs") -> None:
    """Iterate over the config grid and write one JSON per condition."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = Path(output_dir)

    # Load D-values and compute spectral radii once
    dval_path = cfg.get("D_values_path")
    if dval_path is None:
        raise KeyError("D_values_path missing from configuration")

    dval_path_obj = Path(dval_path)
    if dval_path_obj.is_absolute():
        dval_file = dval_path_obj
    else:
        repo_root = config_path.parent.parent  # …/micc-fphs/
        dval_file = (repo_root / dval_path_obj).resolve()

    import csv
    D_vals = []
    with dval_file.open("r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            D_vals.append(float(row["D"]))
    D_vals = np.array(D_vals, dtype=float)
    rhos = compute_kernel_eigs(D_vals)

    # Parameter lists
    lattices = cfg.get("lattice_sizes", [])
    couplings = cfg.get("inverse_couplings", [])
    kappas = cfg.get("pivot_slopes", [])
    gauges = cfg.get("gauge_groups", [])
    f_grid = cfg.get("f_grid", {})
    seeds = cfg.get("seeds", [])

    # Config hash for provenance
    config_hash = compute_sha256_of_obj(cfg)

    # Sweep grid
    for gauge in gauges:
        for L in lattices:
            for b in couplings:
                for kappa in kappas:
                    for seed in seeds:
                        for schedule, f_list in f_grid.items():
                            for f_val in f_list:
                                run_condition(
                                    cfg,
                                    gauge,
                                    int(L),
                                    float(b),
                                    float(kappa),
                                    schedule,
                                    float(f_val),
                                    int(seed),
                                    out_dir,
                                    rhos,
                                )

    # Post-populate config hash into each JSON
    for json_file in out_dir.glob("*.json"):
        data = json.loads(json_file.read_text())
        data["artifacts"]["config_hash"] = config_hash
        json_file.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MICC simulation over parameter grid")
    parser.add_argument("--config", default="configs/fphs_anchors.yaml", help="Path to configuration YAML")
    parser.add_argument("--output", default="runs", help="Output directory for per-condition JSON files")
    args = parser.parse_args()
    main(args.config, args.output)
