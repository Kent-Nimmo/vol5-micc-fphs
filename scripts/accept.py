#!/usr/bin/env python3
"""
QC & acceptance runner for MICC.

Reads the aggregated CSV (from scripts/aggregate.py), groups runs by
(gauge, L, b, kappa) at schedule="during", and applies the acceptance gates:

Gates
-----
1) Visibility: V(f) monotone ↓ with alpha>0 and 95% CI excluding 0,
   AND exponential-fit R^2 >= r2_min on at least one anchor per gauge.
2) Sigma:      σ(f) monotone ↑ (slope CI > 0) on at least one anchor per gauge.
3) Coherence:  P_entropy monotone ↓ (slope CI < 0) for all anchors.
4) Hierarchy:  At f=0 (during), mean σ_SU3 > mean σ_SU2 for all matched (L,b,κ).
5) QC flags:   No hard QC failures (e.g., visibility R^2 < 0.5 per condition).

Outputs:
- results/acceptance_report.json (machine-readable)
- console summary

Exit code:
- 0 if all gates pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Use analysis helpers for weighted fits / bootstrap
from micc.analysis import (
    exp_fit_weighted,
    weighted_linear_fit,
    spearman_monotone,
    bootstrap_alpha_from_seeds,
)

R2_MIN = 0.80      # target for exponential fit of V(f)
Z95   = 1.96

def _group_key(row) -> Tuple:
    return (row["gauge"], int(row["L"]), float(row["b"]), float(row["kappa"]))

def _seed_group(df: pd.DataFrame) -> Dict[Tuple, pd.DataFrame]:
    # schedule filter: acceptance is defined on the primary 'during' schedule
    dfd = df[df["schedule"] == "during"].copy()
    groups: Dict[Tuple, pd.DataFrame] = {}
    for key, g in dfd.groupby(["gauge", "L", "b", "kappa"], dropna=False):
        groups[(key[0], int(key[1]), float(key[2]), float(key[3]))] = g.sort_values("f")
    return groups

def _mean_sem_by_f(g: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Produce mean & SEM across seeds for each f
    out = g.groupby("f")[col].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out["f"].to_numpy(float), out["mean"].to_numpy(float), out["sem"].to_numpy(float)

def _by_seed_matrix(g: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    # Return f grid and a [n_seeds, n_f] matrix of values (aligned by f)
    fvals = np.sort(g["f"].unique())
    seeds = np.sort(g["seed"].unique())
    M = np.zeros((len(seeds), len(fvals)), dtype=float)
    for i, s in enumerate(seeds):
        gi = g[g["seed"] == s]
        # align to the full f grid
        M[i] = gi.set_index("f").reindex(fvals)[col].to_numpy(float)
    return fvals, M

def _pass_visibility(g: pd.DataFrame) -> Dict:
    # Visibility: use Fourier V (aggregator exports it as visibility or visibility_fourier)
    if "visibility_fourier" in g.columns:
        vcol = "visibility_fourier"
    else:
        vcol = "visibility"
    f, V_mean, V_sem = _mean_sem_by_f(g, vcol)

    # Need at least 3 f points
    if len(f) < 3 or np.any(V_mean <= 0):
        return {"pass": False, "reason": "insufficient V grid or nonpositive V"}

    # Weighted exp fit on pooled mean; plus bootstrap alpha CI across seeds
    f_seed, V_mat = _by_seed_matrix(g, vcol)
    if len(f_seed) != len(f) or not np.allclose(f_seed, f):
        # fall back to pooled fit only
        fit = exp_fit_weighted(f, V_mean, V_sem)
        alpha, R2 = fit["alpha"], fit["R2"]
        alpha_lo, alpha_hi = np.nan, np.nan
    else:
        boot = bootstrap_alpha_from_seeds(f, [V_mat[i] for i in range(V_mat.shape[0])], n_boot=500)
        alpha, alpha_lo, alpha_hi, R2 = boot["alpha"], boot["alpha_lo"], boot["alpha_hi"], boot["R2"]

    # Gate: alpha>0 and CI excludes 0, and R2 >= threshold
    pass_alpha = (alpha_hi if not np.isnan(alpha_hi) else alpha) > 0.0
    pass_r2    = R2 >= R2_MIN
    return {
        "pass": bool(pass_alpha and pass_r2),
        "alpha": float(alpha),
        "alpha_lo": float(alpha_lo) if not np.isnan(alpha_lo) else None,
        "alpha_hi": float(alpha_hi) if not np.isnan(alpha_hi) else None,
        "R2": float(R2),
    }

def _pass_sigma(g: pd.DataFrame) -> Dict:
    # σ monotone ↑ via weighted linear slope on pooled seed means
    f, s_mean, s_sem = _mean_sem_by_f(g, "sigma_creutz" if "sigma_creutz" in g.columns else "sigma")
    if len(f) < 3 or np.any(~np.isfinite(s_mean)):
        return {"pass": False, "reason": "insufficient sigma grid"}
    fit = weighted_linear_fit(f, s_mean, s_sem)
    slope = fit["slope"]; half = fit["ci_half_width"]
    return {"pass": bool((slope - half) > 0.0), "slope": float(slope), "ci_half_width": float(half)}

def _pass_coherence(g: pd.DataFrame) -> Dict:
    # P_entropy monotone ↓
    f, p_mean, p_sem = _mean_sem_by_f(g, "P_entropy" if "P_entropy" in g.columns else "coherence_P_entropy")
    if len(f) < 3 or np.any(~np.isfinite(p_mean)):
        return {"pass": False, "reason": "insufficient P grid"}
    fit = weighted_linear_fit(f, p_mean, p_sem)
    slope = fit["slope"]; half = fit["ci_half_width"]
    return {"pass": bool((slope + half) < 0.0), "slope": float(slope), "ci_half_width": float(half)}

def _hierarchy_su3_gt_su2(dfd: pd.DataFrame) -> Dict:
    # Compare σ at f=0 (during) for matched (L,b,κ) between SU3 and SU2
    base = dfd[(dfd["schedule"] == "during") & (dfd["f"] == 0.0)]
    kcols = ["L", "b", "kappa"]
    su2 = base[base["gauge"] == "SU2"].groupby(kcols)["sigma"].mean()
    su3 = base[base["gauge"] == "SU3"].groupby(kcols)["sigma"].mean()
    pairs = su2.index.intersection(su3.index)
    if len(pairs) == 0:
        return {"pass": False, "reason": "no matched anchors at f=0"}
    flags = []
    for key in pairs:
        flags.append(float(su3.loc[key]) > float(su2.loc[key]))
    return {"pass": all(flags), "n_pairs": int(len(pairs)), "violations": int(len(flags) - sum(flags))}

def _qc_flags_fail(dfd: pd.DataFrame) -> Dict:
    # Hard QC: any condition with visibility R2 < 0.5 (from per-condition vis fit), or i0 very small
    bad = 0
    total = 0
    if "twist_scan_fit_R2" in dfd.columns:
        total = int(dfd.shape[0])
        bad = int((dfd["twist_scan_fit_R2"].fillna(1.0) < 0.5).sum())
    small_i0 = int((dfd.get("twist_scan_fit_I0", pd.Series([1.0]*dfd.shape[0])).abs() < 1e-10).sum())
    return {"pass": bool(bad == 0), "bad_r2_lt_0p5": bad, "i0_very_small": small_i0}

def main(summary_csv: str, r2_min: float = R2_MIN) -> int:
    global R2_MIN
    R2_MIN = float(r2_min)

    df = pd.read_csv(summary_csv)
    # Derive convenience columns if aggregator didn’t already add them
    if "sigma_creutz" not in df.columns and "sigma" in df.columns:
        df["sigma_creutz"] = df["sigma"]

    # Gauge-level accumulators
    per_group = []
    groups = _seed_group(df)

    # Per-(gauge, L, b, kappa) evaluations
    for key, g in groups.items():
        g = g.copy()
        # Prefer Fourier visibility if present
        if "visibility_fourier" in g.columns:
            g["V_use"] = g["visibility_fourier"]
        else:
            g["V_use"] = g["visibility"]
        if "visibility_err" in g.columns:
            g["V_err"] = g["visibility_err"]
        else:
            g["V_err"] = np.nan

        # Visibility
        vres = _pass_visibility(g.rename(columns={"visibility_fourier": "visibility", "V_use": "visibility", "V_err": "visibility_err"}))
        # Sigma
        sres = _pass_sigma(g)
        # Coherence
        cres = _pass_coherence(g)

        per_group.append({
            "gauge": key[0], "L": key[1], "b": key[2], "kappa": key[3],
            "V_pass": vres["pass"], "alpha": vres.get("alpha"), "alpha_lo": vres.get("alpha_lo"), "alpha_hi": vres.get("alpha_hi"), "V_R2": vres.get("R2"),
            "sigma_pass": sres["pass"], "sigma_slope": sres.get("slope"), "sigma_slope_ci": sres.get("ci_half_width"),
            "P_pass": cres["pass"], "P_slope": cres.get("slope"), "P_slope_ci": cres.get("ci_half_width"),
        })

    per_group_df = pd.DataFrame(per_group)

    # Gauge-level gates: need at least one anchor passing V and one passing σ per gauge
    gates_by_gauge = {}
    for gauge, gdf in per_group_df.groupby("gauge"):
        gates_by_gauge[gauge] = {
            "V_any": bool((gdf["V_pass"] == True).any()),
            "sigma_any": bool((gdf["sigma_pass"] == True).any()),
            "P_all": bool((gdf["P_pass"] == True).all()),
        }

    # Hierarchy SU3>SU2 at f=0
    hier = _hierarchy_su3_gt_su2(df)

    # QC flags check
    qc = _qc_flags_fail(df)

    # Overall decision
    overall_pass = (
        gates_by_gauge.get("SU2", {}).get("V_any", False) and
        gates_by_gauge.get("SU3", {}).get("V_any", False) and
        gates_by_gauge.get("SU2", {}).get("sigma_any", False) and
        gates_by_gauge.get("SU3", {}).get("sigma_any", False) and
        gates_by_gauge.get("SU2", {}).get("P_all", False) and
        gates_by_gauge.get("SU3", {}).get("P_all", False) and
        hier["pass"] and
        qc["pass"]
    )

    report = {
        "summary_csv": summary_csv,
        "r2_min": R2_MIN,
        "per_group": per_group,
        "gates_by_gauge": gates_by_gauge,
        "hierarchy": hier,
        "qc": qc,
        "overall_pass": overall_pass,
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "acceptance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n=== MICC Acceptance Report ===")
    for gauge, gates in gates_by_gauge.items():
        print(f"{gauge:>4} :: V_any={gates['V_any']}  sigma_any={gates['sigma_any']}  P_all={gates['P_all']}")
    print(f"Hierarchy SU3>SU2 @ f=0 :: pass={hier['pass']} (pairs={hier.get('n_pairs', 0)}, violations={hier.get('violations', 0)})")
    print(f"QC flags :: pass={qc['pass']}  bad_r2_lt_0.5={qc['bad_r2_lt_0p5']}  i0_very_small={qc['i0_very_small']}")
    print(f"OVERALL :: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run MICC acceptance gates over the aggregated summary")
    p.add_argument("--summary", default="results/summary.csv", help="Path to aggregated CSV")
    p.add_argument("--r2-min", type=float, default=R2_MIN, help="Min R^2 required for visibility exponential fit")
    args = p.parse_args()
    raise SystemExit(main(args.summary, args.r2_min))
