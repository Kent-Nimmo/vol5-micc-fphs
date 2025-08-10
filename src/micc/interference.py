"""Interference visibility measurement for MICC.

This module implements a simple gauge‑invariant two‑path interferometer
and a routine to extract the fringe visibility from a twist scan.  The
setup mirrors that used in the FPHS specification: two homotopic
paths connecting the corners of the lattice are compared after
inserting a twist (phase) along a cut shared by both paths.  For
performance we assume that gauge link matrices are diagonal and use
closed‑form exponentiation for the twist.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable, Dict


def _link_index(L: int, x: int, y: int, mu: int) -> int:
    """Compute the flat index of the link (x,y,mu)."""
    return ((x % L) * L + (y % L)) * 2 + mu


def _path_product(U: np.ndarray, L: int, path: Iterable[Tuple[int, int, int]]) -> np.ndarray:
    """Compute the ordered product of link matrices along a path.

    Parameters
    ----------
    U : ndarray
        Link matrices, shape (num_links, N, N).
    L : int
        Lattice size.
    path : iterable of (x,y,mu)
        Sequence of link coordinates specifying the path.  Each link
        contributes ``U[link_index]`` to the product.  Links with mu
        negative imply travelling in the negative direction and are
        represented by the Hermitian conjugate of ``U``.

    Returns
    -------
    ndarray
        Product matrix of shape (N,N).
    """
    N = U.shape[1]
    M = np.eye(N, dtype=complex)
    for (x, y, mu) in path:
        if mu >= 0:
            idx = _link_index(L, x, y, mu)
            M = M @ U[idx]
        else:
            # Negative mu indicates reverse traversal: use Hermitian conjugate
            idx = _link_index(L, x, y, -mu - 1)
            M = M @ U[idx].conj().T
    return M


def build_twist_scan(U: np.ndarray, L: int, group: str, num_phi: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the two‑path correlator I(φ) over a range of twist angles.

    Two homotopic paths from (0,0) to (L-1,L-1) are defined: path1
    travels first in +x then in +y, while path2 travels first in +y
    then in +x.  A twist φ is inserted via a diagonal phase operator
    along the second path only.  The correlator is

    .. math::
       I(φ) = \mathrm{Re}\,\mathrm{Tr}\bigl( U_{\text{path1}} U_{\text{path2}}(φ)^{\dagger} \bigr).

    Parameters
    ----------
    U : ndarray
        Link matrices, shape (num_links, N, N).
    L : int
        Lattice size.
    group : str
        Gauge group (``'SU2'`` or ``'SU3'``).  Determines the dimension
        and the diagonal generator used for the twist.
    num_phi : int, optional
        Number of equally spaced points in [0, 2π] to sample.  Must be
        ≥ 2.

    Returns
    -------
    (phi_values, I_values) : tuple of ndarrays
        Arrays of shape (num_phi,) containing the twist angles and
        corresponding correlator values.
    """
    num_links, N, _ = U.shape
    # Define two paths from (0,0) to (L-1,L-1)
    # Path1: along +x for L-1 steps then +y for L-1 steps
    path1 = []
    for i in range(L - 1):
        path1.append((i, 0, 0))  # (x+i, y, mu=0)
    for j in range(L - 1):
        path1.append((L - 1, j, 1))  # (x=L-1, y+j, mu=1)
    # Path2: along +y then +x
    path2 = []
    for j in range(L - 1):
        path2.append((0, j, 1))
    for i in range(L - 1):
        path2.append((i, L - 1, 0))
    # Compute base products
    M1 = _path_product(U, L, path1)
    M2 = _path_product(U, L, path2)
    # Determine diagonal generator for twist
    if group.upper() == 'SU2':
        diag_gen = np.array([1.0, -1.0], dtype=float)
    elif group.upper() == 'SU3':
        diag_gen = np.array([1.0, -1.0, 0.0], dtype=float)
    else:
        raise ValueError(f"Unsupported group: {group}")
    # Sample phi values
    phi_values = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)
    I_values = np.zeros_like(phi_values, dtype=float)
    # Compute correlator for each phi
    for idx, phi in enumerate(phi_values):
        # Twist operator on path2: multiply M2 on the left by phase diag
        phases = np.exp(1j * phi * diag_gen)
        twist = np.diag(phases)
        M2_phi = twist @ M2
        corr = np.trace(M1 @ M2_phi.conj().T)
        I_values[idx] = corr.real
    return phi_values, I_values


def fit_visibility(phi_values: np.ndarray, I_values: np.ndarray) -> Dict[str, float]:
    """Fit the twist correlator to extract visibility parameters.

    The fitting model is

    .. math::
       I(φ) ≈ I_0 + A \cos(φ + φ_0).

    We rewrite this as a linear combination of cos(φ) and sin(φ):

    .. math::
       I(φ) = a_0 + a_1 \cos(φ) + b_1 \sin(φ),

    which we fit by ordinary least squares.  The visibility amplitude
    and phase offset are recovered via

    .. math::
       A = \sqrt{a_1^2 + b_1^2},\quad φ_0 = \arctan2(-b_1, a_1),\quad I_0 = a_0.

    Parameters
    ----------
    phi_values : ndarray
        Twist angles (radians).
    I_values : ndarray
        Measured correlator values.

    Returns
    -------
    dict
        Dictionary with keys ``A``, ``I0`` and ``phi0``.
    """
    # Build design matrix: columns [1, cos(phi), sin(phi)]
    X = np.column_stack([
        np.ones_like(phi_values),
        np.cos(phi_values),
        np.sin(phi_values),
    ])
    # Solve least squares: minimise ||X beta - I||
    beta, *_ = np.linalg.lstsq(X, I_values, rcond=None)
    a0, a1, b1 = beta
    A = np.sqrt(a1 ** 2 + b1 ** 2)
    phi0 = float(np.arctan2(-b1, a1))
    I0 = float(a0)
    return {"A": float(A), "I0": I0, "phi0": phi0}