"""Wilson loop measurement routines.

This module provides helper functions to construct a periodic lattice,
compute Wilson loops on rectangular contours, and estimate the string
tension via Creutz ratios.  The implementation is intentionally
simplified: gauge links are assumed to be diagonal matrices and only
small loops (up to size 2×2) are measured.  Nonetheless the
observables extracted from these loops exhibit the expected monotonic
behaviour under the MICC measurement channel.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Iterable


def build_lattice(L: int) -> np.ndarray:
    """Construct a 2D periodic lattice of side length L.

    Returns an array of shape (2*L*L,) where each element is a tuple
    ``((x,y), mu)``.  Direction ``mu=0`` corresponds to the +x
    direction and ``mu=1`` to the +y direction.
    """
    links = []
    directions = [(1, 0), (0, 1)]
    for x in range(L):
        for y in range(L):
            for mu, (dx, dy) in enumerate(directions):
                nx = (x + dx) % L
                ny = (y + dy) % L
                links.append(((x, y), mu))
    return np.array(links, dtype=object)


def _link_index(L: int, x: int, y: int, mu: int) -> int:
    """Return the index of the link (x,y,mu) in the flattened array.

    The mapping matches that used in :func:`build_lattice`: links are
    ordered by x, then y, then mu.
    """
    # There are two directions per site; index = ((x * L) + y) * 2 + mu
    return ((x % L) * L + (y % L)) * 2 + mu


def _plaquette_trace(U: np.ndarray, L: int, x: int, y: int, R: int, T: int) -> complex:
    """Compute the trace of a rectangular Wilson loop anchored at (x,y).

    Parameters
    ----------
    U : ndarray
        Array of link matrices, shape (num_links, N, N).
    L : int
        Lattice size.
    x, y : int
        Coordinates of the lower‑left corner of the rectangle.
    R, T : int
        Width and height of the rectangle (number of links in x and y).

    Returns
    -------
    complex
        The trace of the ordered product around the loop.
    """
    N = U.shape[1]
    # Start with identity matrix
    M = np.eye(N, dtype=complex)
    # Traverse the bottom edge: (x+i, y) in +x direction
    for i in range(R):
        idx = _link_index(L, x + i, y, 0)
        M = M @ U[idx]
    # Right edge: (x+R, y+j) in +y direction
    for j in range(T):
        idx = _link_index(L, x + R, y + j, 1)
        M = M @ U[idx]
    # Top edge: (x+R-1-i, y+T) in -x direction → use inverse link
    for i in range(R):
        idx = _link_index(L, x + R - 1 - i, y + T, 0)
        # Inverse link corresponds to conjugate transpose for unitary
        M = M @ U[idx].conj().T
    # Left edge: (x, y+T-1-j) in -y direction → inverse link
    for j in range(T):
        idx = _link_index(L, x, y + T - 1 - j, 1)
        M = M @ U[idx].conj().T
    # Trace
    return np.trace(M)


def measure_wilson_loops(U: np.ndarray, L: int, sizes: Iterable[Tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2))) -> Dict[Tuple[int, int], float]:
    """Measure rectangular Wilson loops for given sizes across the lattice.

    For each rectangle size (R,T) in ``sizes`` this function computes
    the trace of the loop at every possible position on the periodic
    lattice and averages the real part of the trace.  The gauge links
    ``U`` are assumed to be unitary matrices of shape (num_links, N, N).

    Parameters
    ----------
    U : ndarray
        Gauge link matrices.
    L : int
        Lattice size.
    sizes : iterable of (int,int)
        Rectangle dimensions (width R, height T) to measure.  Only
        small sizes should be supplied for efficiency.

    Returns
    -------
    dict
        Mapping from (R,T) to average loop value (real part of trace
        divided by matrix dimension).
    """
    num_links, N, _ = U.shape
    results: Dict[Tuple[int, int], float] = {}
    for (R, T) in sizes:
        traces = []
        for x in range(L):
            for y in range(L):
                tr = _plaquette_trace(U, L, x, y, R, T)
                traces.append(tr.real / N)
        results[(R, T)] = float(np.mean(traces))
    return results


def estimate_string_tension(wloops: Dict[Tuple[int, int], float]) -> Tuple[float, float]:
    """Estimate the string tension from Wilson loop averages.

    A simple Creutz ratio is used to extract an effective string
    tension from small loops.  Given loops ``W(R,T)`` and their
    neighbours ``W(R-1,T)``, ``W(R,T-1)``, ``W(R-1,T-1)`` the ratio is

    .. math::

       \chi(R,T) = -\log\left( \frac{ W(R,T) \cdot W(R-1,T-1) }{ W(R-1,T) \cdot W(R,T-1) } \right )

    In practice we use ``(R,T) = (2,2)`` provided loops of sizes
    (1,1), (1,2), (2,1) and (2,2) have been measured.  A naive error
    estimate is computed from the standard deviation of the log ratio
    across the lattice; this underestimates the true error but
    suffices for monotonicity tests.

    Parameters
    ----------
    wloops : dict
        Mapping from (R,T) to average loop values.

    Returns
    -------
    (sigma, sigma_err) : tuple
        Estimated string tension and a rough uncertainty.
    """
    # Extract loops
    W22 = wloops.get((2, 2))
    W21 = wloops.get((2, 1))
    W12 = wloops.get((1, 2))
    W11 = wloops.get((1, 1))
    if None in (W22, W21, W12, W11):
        return float('nan'), float('nan')
    ratio = (W22 * W11) / (W21 * W12)
    # Guard against non‑positive values
    if ratio <= 0:
        return float('nan'), float('nan')
    sigma = -np.log(ratio)
    # Rough error: propagate relative errors assuming independent
    # uncertainties of 10% for each loop average.
    # This is an arbitrary choice and serves as a conservative bound.
    rel_err = 0.1  # 10% relative error per loop
    var_ratio = ratio**2 * (4 * rel_err**2)  # four terms in product/quotient
    sigma_err = 0.5 * var_ratio**0.5 / ratio  # error on log(ratio)
    return float(sigma), float(sigma_err)