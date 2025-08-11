# interference.py  —  patched
"""
Interference visibility measurement for MICC.

Changes:
- Twist applied along the vertical leg of path2 (shared cut).
- Normalized correlator I_norm(φ) using Hilbert–Schmidt norms.
- Cosine fit via (weighted) least squares with R^2; Fourier visibility from
  the first harmonic; safe normalization and optional clipping to [0,1].
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Iterable, Dict, Optional


def _link_index(L: int, x: int, y: int, mu: int) -> int:
    return ((x % L) * L + (y % L)) * 2 + mu


def _path_product(U: np.ndarray, L: int, path: Iterable[tuple[int, int, int]]) -> np.ndarray:
    N = U.shape[1]
    M = np.eye(N, dtype=complex)
    for (x, y, mu) in path:
        if mu >= 0:
            idx = _link_index(L, x, y, mu)
            M = M @ U[idx]
        else:
            idx = _link_index(L, x, y, -mu - 1)
            M = M @ U[idx].conj().T
    return M


def _hs_norm(M: np.ndarray) -> float:
    """Hilbert–Schmidt norm sqrt(Tr(M M†))."""
    return float(np.sqrt(np.trace(M @ M.conj().T).real))


def build_twist_scan(
    U: np.ndarray,
    L: int,
    group: str,
    num_phi: int = 32,
    std_tol: float = 1e-12,
    normalize: bool = True,
    eps_norm: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute I(φ) = Re Tr(U_path1 U_path2(φ)†) with twist on the *vertical leg*
    of path2 (the shared cut). If normalize=True, divide by an HS norm scale.
    """
    num_links, N, _ = U.shape

    # path1: +x then +y; path2: +y then +x
    path1 = [(i, 0, 0) for i in range(L - 1)] + [(L - 1, j, 1) for j in range(L - 1)]
    path2 = [(0, j, 1) for j in range(L - 1)] + [(i, L - 1, 0) for i in range(L - 1)]

    M1 = _path_product(U, L, path1)
    # Untwisted path2 for normalization scale
    M2_0 = _path_product(U, L, path2)

    g = group.upper()
    if g == "SU2":
        diag_gen = np.array([1.0, -1.0], dtype=float)
    elif g == "SU3":
        diag_gen = np.array([1.0, -1.0, 0.0], dtype=float)
    else:
        raise ValueError(f"Unsupported group: {group}")

    phi_values = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)
    I_values = np.zeros_like(phi_values, dtype=float)

    # Normalization scale
    scale = 1.0
    if normalize:
        s1 = _hs_norm(M1)
        s2 = _hs_norm(M2_0)
        scale = max(s1 * s2, eps_norm)

    vert_len = L - 1
    for k, phi in enumerate(phi_values):
        phases = np.exp(1j * phi * diag_gen)
        twist = np.diag(phases)

        # Build M2 with per-link twist on vertical leg only
        M2_phi = np.eye(N, dtype=complex)
        for idx_link, (x, y, mu) in enumerate(path2):
            if mu >= 0:
                idx = _link_index(L, x, y, mu)
                if idx_link < vert_len and mu == 1:
                    M2_phi = M2_phi @ (twist @ U[idx])
                else:
                    M2_phi = M2_phi @ U[idx]
            else:
                idx = _link_index(L, x, y, -mu - 1)
                M2_phi = M2_phi @ U[idx].conj().T

        corr = np.trace(M1 @ M2_phi.conj().T).real
        I_values[k] = float(corr / scale)

    # Fallback in case scan is numerically flat
    if np.std(I_values) < std_tol:
        for k, phi in enumerate(phi_values):
            phases = np.exp(1j * phi * diag_gen)
            twist = np.diag(phases)
            M2_phi = twist @ M2_0
            corr = np.trace(M1 @ M2_phi.conj().T).real
            I_values[k] = float(corr / scale)

    return phi_values, I_values


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Weighted least squares; returns beta and SSE."""
    if w is None:
        W = None
        beta, residuals, *_ = np.linalg.lstsq(X, y, rcond=None)
        if residuals.size:  # numpy packs SSE in residuals[0] when full rank
            sse = float(residuals[0])
        else:
            yhat = X @ beta
            sse = float(np.sum((y - yhat) ** 2))
        return beta, sse
    w = np.asarray(w, dtype=float)
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    yhat = X @ beta
    sse = float(((y - yhat) ** 2 * w).sum())
    return beta, sse


def fit_visibility(
    phi_values: np.ndarray,
    I_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    eps_scale: float = 0.0,
    clip_to_unit: bool = True,
) -> Dict[str, float]:
    """
    Return both cosine-fit visibility and Fourier-based visibility plus R^2.

    - Cosine fit: I ≈ I0 + A cos(φ + φ0) via (weighted) least squares.
    - Fourier visibility: first harmonic amplitude normalized by I0.
    - Safe normalization: V = A / max(|I0|, eps_local) with
      eps_local = max(eps, eps_scale * |I0|).
    """
    phi_values = np.asarray(phi_values, dtype=float)
    I_values = np.asarray(I_values, dtype=float)

    # Design matrix [1, cos φ, sin φ]
    X = np.column_stack([np.ones_like(phi_values), np.cos(phi_values), np.sin(phi_values)])

    beta, sse = _weighted_lstsq(X, I_values, weights)
    a0, a1, b1 = beta
    A_fit = float(np.hypot(a1, b1))
    phi0 = float(np.arctan2(-b1, a1))
    I0 = float(a0)

    # Weighted R^2
    if weights is None:
        ybar = float(I_values.mean())
        sst = float(((I_values - ybar) ** 2).sum())
    else:
        w = np.asarray(weights, dtype=float)
        ybar = float((w * I_values).sum() / max(w.sum(), 1e-15))
        sst = float((w * (I_values - ybar) ** 2).sum())
    R2 = float(1.0 - (sse / max(sst, 1e-15)))

    # Safe normalization and clipping
    eps_local = max(eps, eps_scale * abs(I0))
    V_fit = float(A_fit / max(abs(I0), eps_local))

    # Fourier visibility (first harmonic)
    M = len(I_values)
    c1 = (2.0 / M) * np.sum(I_values * np.exp(-1j * phi_values))
    A_fourier = float(abs(c1))
    V_fourier = float(A_fourier / max(abs(I0), eps_local))

    V = V_fourier
    if clip_to_unit:
        V = float(np.clip(V, 0.0, 1.0))
        V_fit = float(np.clip(V_fit, 0.0, 1.0))

    return {
        "A": A_fit,
        "I0": I0,
        "phi0": phi0,
        "R2": R2,
        "visibility_fit": V_fit,
        "visibility_fourier": V_fourier,
        # Prefer Fourier visibility for downstream
        "visibility": V,
    }
