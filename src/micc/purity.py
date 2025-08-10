"""Coherence and purity proxies for MICC.

This module provides two practical measures of the coherence of the
gauge field configuration without explicitly constructing a density
matrix.  Both proxies operate on per‑link gauge matrices and return
a single scalar summarising the degree of coherence across all links.

1. **Coherence factor (C)**

   For each link matrix ``U_i`` of dimension ``N×N`` we compute
   ``C_i = |Tr(U_i)| / N``.  Averaging ``C_i`` over all links yields
   a coherence factor between 0 and 1; higher values indicate greater
   phase alignment.  For diagonal matrices this reduces to the mean
   absolute average of the diagonal phases.

2. **Spectral entropy proxy (P_entropy)**

   The eigen‑phases of all link matrices are collected into a single
   histogram over ``num_bins`` bins spanning ``[−π, π]``.  The
   Shannon entropy of this distribution quantifies dispersion of the
   phases.  A uniform distribution yields maximal entropy ``H_max``.
   The purity proxy is defined as ``P = 1 - H/H_max``.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_coherence_factor(U: np.ndarray) -> float:
    """Compute the average coherence factor C.

    Parameters
    ----------
    U : ndarray
        Link matrices of shape (num_links, N, N).

    Returns
    -------
    float
        Coherence factor between 0 and 1.
    """
    num_links, N, _ = U.shape
    traces = np.trace(U, axis1=1, axis2=2)
    C_values = np.abs(traces) / N
    return float(np.mean(C_values))


def compute_spectral_entropy(U: np.ndarray, num_bins: int = 32) -> float:
    """Compute the spectral entropy proxy P.

    Parameters
    ----------
    U : ndarray
        Link matrices of shape (num_links, N, N).  Assumed unitary.
    num_bins : int, optional
        Number of bins for the eigen‑phase histogram.

    Returns
    -------
    float
        Purity proxy ``P = 1 - H/H_max``.
    """
    num_links, N, _ = U.shape
    # For diagonal U the eigenvalues are the diagonal entries
    # Compute phases for each diagonal entry in (−π, π]
    phases = []
    for i in range(num_links):
        diag = np.diagonal(U[i])
        phases.extend(np.angle(diag))
    phases = np.array(phases, dtype=float)
    # Histogram over [−π, π]
    hist, _ = np.histogram(phases, bins=num_bins, range=(-np.pi, np.pi), density=False)
    total = hist.sum()
    if total == 0:
        return float('nan')
    p = hist / total
    # Avoid log(0) by masking zero entries
    mask = p > 0
    H = -np.sum(p[mask] * np.log(p[mask]))
    H_max = np.log(num_bins)
    # Normalise
    P = 1.0 - H / H_max
    return float(P)