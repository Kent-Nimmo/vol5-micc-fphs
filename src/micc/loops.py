# loops.py  —  patched
"""
Wilson loop measurement routines.

Changes:
- Per‑anchor Creutz: drop invalid anchors instead of clamping → less bias.
- σ forced non‑negative after log (area‑law effective tension).
- Adds optional global area‑law fit helper for cross‑checks.
- Adds a 'detailed' Creutz estimator that also returns valid_anchor_fraction.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Iterable, Optional, List


def build_lattice(L: int) -> np.ndarray:
    links = []
    for x in range(L):
        for y in range(L):
            for mu in (0, 1):  # 0:+x, 1:+y
                links.append(((x, y), mu))
    return np.array(links, dtype=object)


def _link_index(L: int, x: int, y: int, mu: int) -> int:
    return ((x % L) * L + (y % L)) * 2 + mu


def _plaquette_trace(U: np.ndarray, L: int, x: int, y: int, R: int, T: int) -> complex:
    """Trace of an (R×T) Wilson loop anchored at (x,y)."""
    N = U.shape[1]
    M = np.eye(N, dtype=complex)
    # bottom edge (+x)
    for i in range(R):
        M = M @ U[_link_index(L, x + i, y, 0)]
    # right edge (+y)
    for j in range(T):
        M = M @ U[_link_index(L, x + R, y + j, 1)]
    # top edge (−x)
    for i in range(R):
        M = M @ U[_link_index(L, x + R - 1 - i, y + T, 0)].conj().T
    # left edge (−y)
    for j in range(T):
        M = M @ U[_link_index(L, x, y + T - 1 - j, 1)].conj().T
    return np.trace(M)


def measure_wilson_loops(
    U: np.ndarray,
    L: int,
    sizes: Iterable[Tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
    return_samples: bool = False,
):
    """Average rectangular Wilson loops; optionally return per‑anchor samples."""
    num_links, N, _ = U.shape
    results: Dict[Tuple[int, int], float] = {}
    samples: Dict[Tuple[int, int], np.ndarray] = {}

    for (R, T) in sizes:
        vals = np.empty((L, L), dtype=float)
        for x in range(L):
            for y in range(L):
                tr = _plaquette_trace(U, L, x, y, R, T).real / N
                vals[x, y] = tr
        results[(R, T)] = float(np.mean(vals))
        if return_samples:
            samples[(R, T)] = vals

    if return_samples:
        return results, samples
    return results


def estimate_string_tension(
    wloops: Dict[Tuple[int, int], float],
    samples: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Estimate σ via (2,2) Creutz. Prefer per‑anchor when available.

    χ = -log( (W22 * W11) / (W21 * W12) )
    We DROP anchors with invalid loop products rather than clamping.
    """
    # Per‑anchor path (preferred)
    if samples is not None and all(k in samples for k in [(2, 2), (2, 1), (1, 2), (1, 1)]):
        W22 = samples[(2, 2)]
        W21 = samples[(2, 1)]
        W12 = samples[(1, 2)]
        W11 = samples[(1, 1)]

        denom = W21 * W12
        numer = W22 * W11
        valid = (denom > eps) & (numer > eps) & np.isfinite(denom) & np.isfinite(numer)

        if not np.any(valid):
            return float('nan'), float('nan')

        ratio = (numer[valid] / denom[valid]).astype(float)
        # Guard remaining non‑positives from numeric accidents
        ratio = ratio[ratio > eps]
        if ratio.size == 0:
            return float('nan'), float('nan')

        chi = -np.log(ratio)
        chi = np.clip(chi, 0.0, None)  # effective tension cannot be negative

        sigma = float(np.mean(chi))
        # SEM over anchors as a lower bound; proper CIs via blocking/jackknife upstream
        n = chi.size
        sigma_err = float(np.std(chi, ddof=1) / max(n**0.5, 1.0))
        return sigma, sigma_err

    # Fallback: global ratio from averaged loops
    W22 = wloops.get((2, 2))
    W21 = wloops.get((2, 1))
    W12 = wloops.get((1, 2))
    W11 = wloops.get((1, 1))
    if None in (W22, W21, W12, W11):
        return float('nan'), float('nan')

    denom = W21 * W12
    numer = W22 * W11
    if denom <= eps or numer <= eps:
        return float('nan'), float('nan')

    ratio = numer / denom
    if ratio <= eps:
        return float('nan'), float('nan')

    sigma = max(-np.log(ratio), 0.0)

    # Conservative propagated error assuming ~10% rel error per loop
    rel_err = 0.10
    var_ratio = ratio**2 * (4 * rel_err**2)
    sigma_err = 0.5 * (var_ratio**0.5) / ratio
    return float(sigma), float(sigma_err)


def estimate_string_tension_detailed(
    samples: Dict[Tuple[int, int], np.ndarray],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Detailed per‑anchor Creutz on (2,2) returning sigma, sigma_err and valid fraction.
    """
    out = {"sigma": float('nan'), "sigma_err": float('nan'), "valid_anchor_fraction": 0.0}
    if not all(k in samples for k in [(2, 2), (2, 1), (1, 2), (1, 1)]):
        return out

    W22 = samples[(2, 2)]
    W21 = samples[(2, 1)]
    W12 = samples[(1, 2)]
    W11 = samples[(1, 1)]
    denom = W21 * W12
    numer = W22 * W11
    valid = (denom > eps) & (numer > eps) & np.isfinite(denom) & np.isfinite(numer)
    total = valid.size
    out["valid_anchor_fraction"] = float(np.sum(valid) / max(total, 1))

    if not np.any(valid):
        return out

    ratio = (numer[valid] / denom[valid]).astype(float)
    ratio = ratio[ratio > eps]
    if ratio.size == 0:
        return out

    chi = -np.log(ratio)
    chi = np.clip(chi, 0.0, None)
    sigma = float(np.mean(chi))
    n = chi.size
    sigma_err = float(np.std(chi, ddof=1) / max(n**0.5, 1.0))
    out["sigma"] = sigma
    out["sigma_err"] = sigma_err
    return out


# ---------- Global area‑law fit helper (cross‑check) ----------

def _choose_fit_sizes(L: int, sizes: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Select sizes in area‑law window: R,T≥2 and R,T≤floor(L/3); prefer at least one T≥4 if available."""
    max_rt = max(L // 3, 2)
    cand = [(R, T) for (R, T) in sizes if (R >= 2 and T >= 2 and R <= max_rt and T <= max_rt)]
    if not cand:
        return []
    # Prefer to keep at least one with T>=4 if available; otherwise keep as-is
    have_t4 = any(T >= 4 for _, T in cand)
    if have_t4:
        return cand
    return cand  # fine if no T>=4 exists at this L


def fit_area_law_from_samples(
    samples: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[float, float]:
    """
    Fit ln W(R,T) = -σ R T - μ (R+T) - ν using available sizes in samples.
    Returns (sigma, sigma_err). Use as a cross‑check, not the primary.
    """
    sizes = list(samples.keys())
    use = _choose_fit_sizes(int(np.sqrt(list(samples.values())[0].size)), sizes)
    if not use:
        return float('nan'), float('nan')

    # Build design and targets by averaging over anchors for each (R,T)
    rows = []
    ys = []
    ws = []
    for (R, T) in use:
        W = samples[(R, T)]
        # Guard tiny values inside the log, then average
        Wc = np.maximum(W, 1e-15)
        y = np.log(Wc).mean()
        var = np.var(np.log(Wc))  # anchor variance as weight proxy
        rows.append([-(R * T), -(R + T), -1.0])  # [σ, μ, ν] coefficients
        ys.append(y)
        ws.append(1.0 / max(var, 1e-12))

    X = np.asarray(rows, float)
    y = np.asarray(ys, float)
    Wt = np.diag(ws)

    # Weighted least squares
    XtWX = X.T @ Wt @ X
    XtWy = X.T @ Wt @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return float('nan'), float('nan')

    sigma = float(beta[0])
    # approximate error from diagonal of (X^T W X)^{-1}
    cov = np.linalg.pinv(XtWX)
    sigma_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    return sigma, sigma_err
