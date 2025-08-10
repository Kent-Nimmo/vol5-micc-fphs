"""Measurement channel for MICC.

This module defines functions to inject controlled dephasing (partial
measurements) into gauge link variables.  The measurement model
implemented here corresponds to a gauge‑covariant dephasing channel
acting independently on a randomly chosen fraction ``f`` of links.  For
each selected link ``U_i`` belonging to a gauge group ``G`` the
procedure samples a Lie‑algebra kick ``δ`` from a zero‑mean Gaussian
with variance ``f`` and updates the link via

.. math::

   U_i \leftarrow \exp(\mathrm{i}\,δ A)\,U_i,

where ``A`` is a fixed diagonal generator of the Lie algebra.

Because the generators used here are diagonal the exponentiation
reduces to multiplying diagonal elements by complex phase factors.

The function ``apply_dephasing`` returns a new array of link matrices
with the noise applied and reports how many links were measured.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_dephasing(U: np.ndarray, f: float, group: str, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """Apply gauge‑covariant dephasing to a random subset of links.

    Parameters
    ----------
    U : ndarray
        Array of gauge link matrices of shape ``(num_links, N, N)``.
        These matrices are assumed to be diagonal (but the function
        works for arbitrary unitary matrices by performing matrix
        multiplication on the left).
    f : float
        Fraction of links to which dephasing noise is applied.  Must
        satisfy ``0 ≤ f ≤ 1``.
    group : str
        Gauge group name, one of ``'SU2'`` or ``'SU3'``.  Determines
        which diagonal generator is used.
    rng : numpy.random.Generator
        Random number generator for selecting measured links and
        sampling noise amplitudes.

    Returns
    -------
    (U_new, measured_count) : tuple
        A tuple containing the updated link array and the number of
        links that were dephased.
    """
    num_links, N, _ = U.shape
    f = float(f)
    if f < 0.0 or f > 1.0:
        raise ValueError("f must be between 0 and 1")
    # Determine number of measured links (round to nearest integer)
    measured_count = int(round(f * num_links))
    # Copy U to avoid modifying original
    U_new = U.copy()
    if measured_count == 0:
        return U_new, 0
    # Sample indices without replacement
    indices = rng.choice(num_links, size=measured_count, replace=False)
    # Choose diagonal generator according to group
    group_upper = group.upper()
    if group_upper == 'SU2':
        # σ_z/2 generator diagonal entries
        diag_gen = np.array([0.5, -0.5], dtype=float)
    elif group_upper == 'SU3':
        # λ3-like generator diag([1, -1, 0])
        diag_gen = np.array([1.0, -1.0, 0.0], dtype=float)
    else:
        raise ValueError(f"Unsupported group: {group}")
    # For each selected link apply noise
    for idx in indices:
        # Sample a real kick δ from N(0, f)
        delta = rng.normal(loc=0.0, scale=np.sqrt(f))
        phases = np.exp(1j * delta * diag_gen)
        phase_mat = np.diag(phases)
        # Left‑multiply the link: U_i ← exp(i δ A) U_i
        U_new[idx] = phase_mat @ U_new[idx]
    return U_new, measured_count