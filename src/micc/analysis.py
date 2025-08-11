"""Statistical analysis utilities for MICC outputs (expanded).

What’s new:
- Weighted exponential fit for V(f) on log-scale with R^2.
- Bootstrap CI for alpha (resampling across seeds).
- Weighted slope + 95% CI for sigma(f) (and any y vs f).
- Spearman monotonicity test (rank-based, robust to mild nonlinearity).

Notes:
- For exp fits: we model ln V = ln V0 - alpha * f.
- Weights for ln V come from error propagation: var(ln V) ≈ (σV / V)^2.
"""

from __future__ import annotations
from typing import Iterable, Optional, Dict, Tuple, List
import numpy as np
import math


# ---------- Small helpers ----------

def _as_float_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def _r2_score(y: np.ndarray, yhat: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        sst = float(np.sum((y - y.mean())**2))
        sse = float(np.sum((y - yhat)**2))
    else:
        w = np.asarray(w, dtype=float)
        ybar = float((w * y).sum() / max(w.sum(), 1e-15))
        sst = float((w * (y - ybar)**2).sum())
        sse = float((w * (y - yhat)**2).sum())
    return float(1.0 - sse / max(sst, 1e-15))


def _weighted_fit(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (beta, cov_beta) for y ~ X beta using optional diagonal weights."""
    if w is None:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        # homoscedastic OLS covariance proxy
        resid = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        s2 = float((resid @ resid) / dof)
        cov = s2 * np.linalg.pinv(XtX)
        return beta, cov
    W = np.diag(np.asarray(w, dtype=float))
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.pinv(XtWX)
    return beta, cov


# ---------- Monotonicity & simple fits (kept/back‑compat) ----------

def test_monotonicity(f_values: Iterable[float], y_values: Iterable[float], expectation: str) -> Dict[str, float | bool]:
    f = _as_float_array(f_values); y = _as_float_array(y_values)
    X = np.column_stack([np.ones_like(f), f])
    beta, cov = _weighted_fit(X, y, None)
    intercept, slope = beta
    slope_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    z = 1.96
    if expectation == 'increase':
        is_monotonic = (slope - z * slope_err) > 0
    elif expectation == 'decrease':
        is_monotonic = (slope + z * slope_err) < 0
    else:
        raise ValueError("expectation must be 'increase' or 'decrease'")
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'slope_err': slope_err,
        'ci_half_width': float(z * slope_err),
        'is_monotonic': bool(is_monotonic),
    }


def fit_exponential_decay(f_values: Iterable[float], V_values: Iterable[float]) -> Dict[str, float]:
    f = _as_float_array(f_values); V = _as_float_array(V_values)
    mask = V > 0
    f = f[mask]; V = V[mask]
    logV = np.log(V)
    X = np.column_stack([np.ones_like(f), f])
    beta, cov = _weighted_fit(X, logV, None)
    intercept, slope = beta
    alpha = -slope
    alpha_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    V0 = float(np.exp(intercept))
    V0_err = float(np.exp(intercept) * np.sqrt(max(cov[0, 0], 0.0)))
    yhat = X @ beta
    R2 = _r2_score(logV, yhat, None)
    return {'alpha': float(alpha), 'alpha_err': alpha_err, 'V0': V0, 'V0_err': V0_err, 'R2': R2}


# ---------- New: weighted linear fit & Spearman ----------

def spearman_monotone(f_values: Iterable[float], y_values: Iterable[float], expectation: str) -> Dict[str, float | bool]:
    f = _as_float_array(f_values); y = _as_float_array(y_values)

    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind='mergesort')
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(a)+1, dtype=float)
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, r)
        avg = sums / counts
        return avg[inv]

    rf, ry = ranks(f), ranks(y)
    rf0, ry0 = rf - rf.mean(), ry - ry.mean()
    rho = float((rf0 @ ry0) / math.sqrt((rf0 @ rf0) * (ry0 @ ry0) + 1e-15))

    # exact permutation p-value for small n
    n = len(f)
    p_value = float('nan')
    if n <= 8:
        from itertools import permutations
        cnt = 0; ge = 0
        for perm in permutations(range(n)):
            ry_p = ry[list(perm)]
            ry0p = ry_p - ry_p.mean()
            rho_p = (rf0 @ ry0p) / math.sqrt((rf0 @ rf0) * (ry0p @ ry0p) + 1e-15)
            cnt += 1
            if abs(rho_p) >= abs(rho) - 1e-15:
                ge += 1
        p_value = ge / cnt

    ok = rho > 0 if expectation == 'increase' else rho < 0
    return {'rho': rho, 'p_value': p_value, 'is_monotonic': bool(ok)}


def weighted_linear_fit(
    f_values: Iterable[float],
    y_values: Iterable[float],
    y_errors: Optional[Iterable[float]] = None,
    z: float = 1.96,
) -> Dict[str, float | bool]:
    f = _as_float_array(f_values); y = _as_float_array(y_values)
    w = None
    if y_errors is not None:
        yerr = _as_float_array(y_errors)
        w = 1.0 / np.maximum(yerr**2, 1e-15)
    X = np.column_stack([np.ones_like(f), f])
    beta, cov = _weighted_fit(X, y, w)
    intercept, slope = beta
    slope_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {'slope': float(slope), 'intercept': float(intercept), 'slope_err': slope_err, 'ci_half_width': float(z * slope_err)}


# ---------- New: weighted exponential fit with errors & bootstrap alpha ----------

def exp_fit_weighted(
    f_values: Iterable[float],
    V_values: Iterable[float],
    V_errors: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Weighted fit of ln V = ln V0 - alpha f with propagated weights and R^2."""
    f = _as_float_array(f_values); V = _as_float_array(V_values)
    mask = V > 0
    f = f[mask]; V = V[mask]
    if V_errors is not None:
        Verr = _as_float_array(V_errors)[mask]
        Verr = np.maximum(Verr, 1e-12)
        w = 1.0 / np.maximum((Verr / V)**2, 1e-12)   # weights for ln V
    else:
        w = None
    logV = np.log(V)
    X = np.column_stack([np.ones_like(f), f])
    beta, cov = _weighted_fit(X, logV, w)
    intercept, slope = beta
    yhat = X @ beta
    R2 = _r2_score(logV, yhat, w)
    return {
        'alpha': float(-slope),
        'alpha_err': float(np.sqrt(max(cov[1, 1], 0.0))),
        'V0': float(np.exp(intercept)),
        'V0_err': float(np.exp(intercept) * np.sqrt(max(cov[0, 0], 0.0))),
        'R2': R2,
    }


def bootstrap_alpha_from_seeds(
    f_values: Iterable[float],
    V_by_seed: List[Iterable[float]],
    V_err_by_seed: Optional[List[Iterable[float]]] = None,
    n_boot: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Bootstrap alpha CI by resampling seeds with replacement.

    Parameters
    ----------
    f_values : array-like of f grid (shared across seeds)
    V_by_seed : list of sequences, each is V(f) for one seed
    V_err_by_seed : optional list of V_err(f) sequences per seed (same shapes)
    n_boot : number of bootstrap resamples

    Returns keys: alpha, alpha_lo, alpha_hi (95% CI), R2 (from full pooled fit)
    """
    rng = np.random.default_rng() if rng is None else rng
    f = _as_float_array(f_values)
    Vs = [ _as_float_array(v) for v in V_by_seed ]
    if V_err_by_seed is not None:
        VEs = [ _as_float_array(e) for e in V_err_by_seed ]
    else:
        VEs = None

    # Full pooled (across seeds) mean & SEM → reference fit & R2
    V_mat = np.vstack(Vs)              # [n_seeds, n_f]
    V_mean = V_mat.mean(axis=0)
    V_sem  = V_mat.std(axis=0, ddof=1) / max(V_mat.shape[0]**0.5, 1.0)
    ref = exp_fit_weighted(f, V_mean, V_sem)

    # Bootstrap over seeds
    n_seeds = V_mat.shape[0]
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)  # resample seeds
        Vb = V_mat[idx].mean(axis=0)
        Vb_sem = V_mat[idx].std(axis=0, ddof=1) / max(n_seeds**0.5, 1.0)
        fitb = exp_fit_weighted(f, Vb, Vb_sem)
        alphas.append(fitb['alpha'])
    alphas = np.asarray(alphas, dtype=float)
    lo, hi = np.quantile(alphas, [0.025, 0.975])

    return {
        'alpha': float(ref['alpha']),
        'alpha_lo': float(lo),
        'alpha_hi': float(hi),
        'R2': float(ref['R2']),
        'V0': float(ref['V0']),
    }
