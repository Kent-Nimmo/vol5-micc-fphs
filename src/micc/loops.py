# loops.py — updated to support multi-size Creutz averaging (2,2) & (3,3)
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
    N = U.shape[1]
    M = np.eye(N, dtype=complex)
    for i in range(R):  # bottom edge (+x)
        M = M @ U[_link_index(L, x + i, y, 0)]
    for j in range(T):  # right edge (+y)
        M = M @ U[_link_index(L, x + R, y + j, 1)]
    for i in range(R):  # top edge (−x)
        M = M @ U[_link_index(L, x + R - 1 - i, y + T, 0)].conj().T
    for j in range(T):  # left edge (−y)
        M = M @ U[_link_index(L, x, y + T - 1 - j, 1)].conj().T
    return np.trace(M)

def measure_wilson_loops(
    U: np.ndarray,
    L: int,
    sizes: Iterable[Tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
    return_samples: bool = False,
):
    """Average rectangular Wilson loops; optionally return per‑anchor samples."""
    results: Dict[Tuple[int, int], float] = {}
    samples: Dict[Tuple[int, int], np.ndarray] = {}

    for (R, T) in sizes:
        vals = np.empty((L, L), dtype=float)
        for x in range(L):
            for y in range(L):
                tr = _plaquette_trace(U, L, x, y, R, T).real / U.shape[1]
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
    Legacy single-window σ via (2,2) Creutz; prefers per‑anchor when available.
    χ = -log( (W22 * W11) / (W21 * W12) ).
    """
    if samples is not None and all(k in samples for k in [(2, 2), (2, 1), (1, 2), (1, 1)]):
        W22 = samples[(2, 2)]; W21 = samples[(2, 1)]
        W12 = samples[(1, 2)]; W11 = samples[(1, 1)]
        denom = W21 * W12; numer = W22 * W11
        valid = (denom > eps) & (numer > eps) & np.isfinite(denom) & np.isfinite(numer)
        if not np.any(valid): return float('nan'), float('nan')
        ratio = (numer[valid] / denom[valid]).astype(float)
        ratio = ratio[ratio > eps]
        if ratio.size == 0: return float('nan'), float('nan')
        chi = -np.log(ratio); chi = np.clip(chi, 0.0, None)
        n = chi.size
        return float(np.mean(chi)), float(np.std(chi, ddof=1) / max(n**0.5, 1.0))

    # Fallback from averaged loops
    W22 = wloops.get((2, 2)); W21 = wloops.get((2, 1))
    W12 = wloops.get((1, 2)); W11 = wloops.get((1, 1))
    if None in (W22, W21, W12, W11): return float('nan'), float('nan')
    denom = W21 * W12; numer = W22 * W11
    if denom <= eps or numer <= eps: return float('nan'), float('nan')
    ratio = numer / denom
    if ratio <= eps: return float('nan'), float('nan')
    sigma = max(-np.log(ratio), 0.0)
    # rough propagated error (kept)
    rel_err = 0.10; var_ratio = ratio**2 * (4 * rel_err**2)
    sigma_err = 0.5 * (var_ratio**0.5) / ratio
    return float(sigma), float(sigma_err)

def estimate_string_tension_multi(
    samples: Dict[Tuple[int, int], np.ndarray],
    sizes: Iterable[Tuple[int, int]],
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Multi-window Creutz: average valid per‑anchor χ from each (R,T) in 'sizes'.
    Uses χ(R,T) = -log( W(R,T) W(R-1,T-1) / [ W(R-1,T) W(R,T-1) ] ).
    Returns (sigma, sigma_err) where sigma_err is SEM over all valid anchor χ.
    """
    chis: List[float] = []
    for (R, T) in sizes:
        keys = ((R, T), (R - 1, T), (R, T - 1), (R - 1, T - 1))
        if any((k not in samples) for k in keys):
            continue
        WRT  = samples[(R, T)]
        WRmT = samples[(R - 1, T)]
        WRTm = samples[(R, T - 1)]
        WRmTm = samples[(R - 1, T - 1)]
        denom = WRmT * WRTm
        numer = WRT * WRmTm
        valid = (denom > eps) & (numer > eps) & np.isfinite(denom) & np.isfinite(numer)
        if not np.any(valid):
            continue
        ratio = (numer[valid] / denom[valid]).astype(float)
        ratio = ratio[ratio > eps]
        if ratio.size == 0:
            continue
        chi = -np.log(ratio)
        chi = np.clip(chi, 0.0, None)
        chis.append(chi)

    if not chis:
        return float('nan'), float('nan')

    all_chi = np.concatenate(chis, axis=None)
    n = all_chi.size
    sigma = float(np.mean(all_chi))
    sigma_err = float(np.std(all_chi, ddof=1) / max(n**0.5, 1.0))
    return sigma, sigma_err

# (fit_area_law_from_samples kept as-is)
