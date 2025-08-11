# measurement.py  —  patched
"""
Measurement channel for MICC.

Gauge‑covariant dephasing acting on a random fraction f of links:
for each selected link U_i, sample a real kick δ ~ N(0, f) and update
U_i ← exp(i δ A) U_i, with A a fixed diagonal generator (per group).
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def _diag_generator(group_upper: str) -> np.ndarray:
    """Return a diagonal generator for the given gauge group."""
    if group_upper == 'SU2':
        # σ_z/2
        return np.array([0.5, -0.5], dtype=float)
    elif group_upper == 'SU3':
        # λ3‑like diag([1,-1,0]); simple, traceless, diagonal
        return np.array([1.0, -1.0, 0.0], dtype=float)
    else:
        raise ValueError(f"Unsupported group: {group_upper}")


def apply_dephasing(U: np.ndarray, f: float, group: str, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """Apply gauge‑covariant dephasing to a random subset of links.

    Parameters
    ----------
    U : (num_links, N, N) complex array
        Gauge links.
    f : float in [0,1]
        Fraction of links to dephase (rounded to nearest integer count).
    group : 'SU2' | 'SU3'
    rng : numpy.random.Generator

    Returns
    -------
    (U_new, measured_count)
    """
    num_links, N, _ = U.shape
    f = float(f)
    if not (0.0 <= f <= 1.0):
        raise ValueError("f must be between 0 and 1")

    measured_count = int(round(f * num_links))
    if measured_count <= 0:
        return U.copy(), 0

    indices = rng.choice(num_links, size=measured_count, replace=False)
    diag_gen = _diag_generator(group.upper())

    U_new = U.copy()
    # Var(δ) = f per cycle per link (scalar kick along chosen diagonal generator)
    for idx in indices:
        delta = rng.normal(loc=0.0, scale=np.sqrt(f))
        phases = np.exp(1j * delta * diag_gen)
        phase_mat = np.diag(phases)
        U_new[idx] = phase_mat @ U_new[idx]

    return U_new, measured_count


def apply_dephasing_with_indices(
    U: np.ndarray,
    f: float,
    group: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """Like apply_dephasing, but also returns measured indices and hit fraction.

    Returns
    -------
    (U_new, measured_count, indices, hit_fraction)
    """
    num_links, N, _ = U.shape
    f = float(f)
    if not (0.0 <= f <= 1.0):
        raise ValueError("f must be between 0 and 1")

    measured_count = int(round(f * num_links))
    indices = np.empty(0, dtype=int)
    if measured_count > 0:
        indices = rng.choice(num_links, size=measured_count, replace=False)
    diag_gen = _diag_generator(group.upper())

    U_new = U.copy()
    for idx in indices:
        delta = rng.normal(loc=0.0, scale=np.sqrt(f))
        phases = np.exp(1j * delta * diag_gen)
        phase_mat = np.diag(phases)
        U_new[idx] = phase_mat @ U_new[idx]

    hit_fraction = float(measured_count) / float(num_links if num_links > 0 else 1)
    return U_new, measured_count, indices, hit_fraction
