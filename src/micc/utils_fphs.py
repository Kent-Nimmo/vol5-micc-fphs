"""Utility functions for the MICC simulation.

This module collects common helper routines used throughout the
measurement‑induced context collapse (MICC) simulation.  Where
possible these functions are pure and deterministic given their
inputs.  Randomness is confined to functions that explicitly accept
a random number generator or seed.

Key functionality provided here:

* **Fractal dimension and pivot weight** – functions to compute the
  logistic fractal dimension ``D(n)`` and the associated pivot weight
  ``g(D)`` as described in the FPHS specification.  The logistic
  dimension is defined so that ``D(0) = 2`` regardless of the
  parameter ``kappa``.  The pivot weight defaults to the linear map
  ``g(D) = D/3`` consistent with the Volume‑3 kernel specification.
* **Flip count generation** – an adaptation of the tick‑flip simulator
  from the ``vol4-loop-fluctuation-sim`` repository.  This routine
  generates a per‑link flip count array for an ``L×L`` lattice via
  random walks over the tick‑flip operator algebra.  The algorithm
  mirrors the original implementation but allows for fewer steps per
  link in order to reduce computational cost.  Flip counts are
  deterministic given a seed and simulation parameters.
* **Kernel construction** – for the current version of MICC we use
  constant diagonal kernels for the SU(2) and SU(3) gauge groups.  A
  simple helper builds these kernels with shapes ``(num_links, N, N)``
  where ``N`` is the dimension of the gauge group (2 or 3).  The
  kernels embed a single diagonal generator (σ_z for SU(2) and
  λ₃-like for SU(3)) and may be scaled further by the pivot weight.
* **Exponentiation of gauge potentials** – fast routines to compute
  group elements ``U`` from scalar weights ``gD`` by exponentiating a
  diagonal generator.  Because we use diagonal generators the
  exponentiation reduces to computing complex exponentials of the
  entries.
* **Hashing utilities** – convenience wrappers around ``hashlib`` to
  compute SHA‑256 hashes of numpy arrays and JSON‑serialisable
  objects.  These hashes are recorded in the output for provenance.

The functions defined here are intentionally kept small and
self‑contained so that they can be unit‑tested independently of the
larger simulation pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Tuple

import numpy as np


def logistic_dimension(n: np.ndarray, kappa: float, n0: float = 0.0) -> np.ndarray:
    """Compute the logistic fractal dimension D(n).

    The definition matches that used in the FPHS pipeline:

    .. math::

       D(n) = 1 + \frac{2}{1 + \exp(kappa \cdot (n - n0))}

    With ``kappa=0`` the function returns a constant 2.  The
    midpoint ``n0`` is zero by default so that ``D(0) = 2``.

    Parameters
    ----------
    n : ndarray
        Array of flip counts per link.
    kappa : float
        Logistic slope parameter.  Larger values lead to steeper
        transitions in the fractal dimension as a function of flip
        count.
    n0 : float, optional
        Midpoint parameter of the logistic; default is 0.0.

    Returns
    -------
    ndarray
        Array of the same shape as ``n`` with the corresponding
        fractal dimensions.
    """
    n = np.asarray(n, dtype=float)
    if kappa == 0.0:
        return np.full_like(n, 2.0, dtype=float)
    return 1.0 + 2.0 / (1.0 + np.exp(kappa * (n - n0)))


def pivot_weight(D: np.ndarray) -> np.ndarray:
    """Compute the pivot weight g(D) from the fractal dimension.

    According to the Volume‑3 kernel specification the pivot weight
    multiplies the base kernel by ``g(D) = D / 3``.  The simple
    proportionality constant ``1/3`` ensures that the resulting gauge
    potentials have the correct scaling for SU(2) and SU(3).

    Parameters
    ----------
    D : ndarray
        Fractal dimension array.

    Returns
    -------
    ndarray
        Pivot weight array of the same shape as ``D``.
    """
    return D / 3.0


def _flip_operator_functions():
    """Return the tick‑flip operator functions F, S, X, C, Phi.

    These functions are defined as in the original flip‑count simulator
    and operate on a ``TickState`` instance (distribution and context
    depth).  We implement them here locally to avoid importing
    external packages.  See volume‑4 documentation for details.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TickState:
        distribution: np.ndarray
        N: int

        def __post_init__(self):
            if self.distribution.ndim != 1:
                raise ValueError("distribution must be 1D")
            if self.distribution.size != 2 * self.N + 1:
                raise ValueError("distribution length must be 2*N+1")

    def F_op(state: TickState) -> TickState:
        dist = state.distribution
        L = dist.size
        new = np.zeros_like(dist)
        if L > 1:
            new[0] = dist[0] + dist[1]
        else:
            new[0] = dist[0]
        if L > 2:
            new[1 : L - 1] = dist[2:]
        if L > 0:
            new[L - 1] = 0.0
        return TickState(new, state.N)

    def S_op(state: TickState) -> TickState:
        dist = state.distribution
        L = dist.size
        new = np.zeros_like(dist)
        if L > 2:
            new[1 : L - 1] = dist[0 : L - 2]
        if L > 1:
            new[L - 1] = dist[L - 1] + dist[L - 2]
        else:
            new[0] = dist[0]
        return TickState(new, state.N)

    def X_op(state: TickState) -> TickState:
        return TickState(state.distribution[::-1].copy(), state.N)

    def C_op(state: TickState) -> TickState:
        dist = state.distribution
        rev = dist[::-1]
        return TickState(0.5 * (dist + rev), state.N)

    def Phi_op(state: TickState) -> TickState:
        return C_op(X_op(state))

    return TickState, [F_op, S_op, X_op, C_op, Phi_op]


def _build_default_lattice(L: int) -> np.ndarray:
    """Construct a 2D periodic lattice of side length L.

    Returns an array of length ``2*L*L`` where each element is a
    ``((x,y), mu)`` tuple.  Directions ``mu=0`` and ``mu=1`` correspond
    respectively to positive x and positive y directions.  Periodic
    boundary conditions wrap indices around at the lattice edges.
    """
    links = []
    directions = [(1, 0), (0, 1)]  # mu=0: +x, mu=1: +y
    for x in range(L):
        for y in range(L):
            for mu, (dx, dy) in enumerate(directions):
                nx = (x + dx) % L
                ny = (y + dy) % L
                links.append(((x, y), mu))
    return np.array(links, dtype=object)


def generate_flip_counts(L: int, seed: int, N: int = 2, steps_per_link: int = 20) -> np.ndarray:
    """Generate tick‑flip counts for all links on an L×L lattice.

    This routine follows the algorithm of ``generate_flip_counts.py`` from
    ``vol4-loop-fluctuation-sim`` but limits the number of operator
    sequences per link to control computational cost.  For each link,
    the tick distribution is initialised to a delta function, and the
    five tick‑flip operators (F, S, X, C, Φ) are applied repeatedly.
    The number of times the distribution changes at a link‑specific
    "watch" index is counted.  The watch index depends on the link
    coordinates and the direction ``mu``.

    Parameters
    ----------
    L : int
        Lattice size (side length).
    seed : int
        Random seed controlling the random walk.  Each seed yields a
        deterministic flip count array.
    N : int, optional
        Context depth; the distribution length is ``2*N+1``.  The
        default value (2) matches the original simulator.
    steps_per_link : int, optional
        Number of tick‑flip sequences applied per link.  Lower values
        reduce run time at the expense of noisier flip counts.

    Returns
    -------
    ndarray
        Array of length ``2*L*L`` containing integer flip counts per
        link.
    """
    TickState, ops = _flip_operator_functions()
    lattice = _build_default_lattice(L)
    counts = np.zeros(len(lattice), dtype=int)
    rng = np.random.default_rng(seed)
    for idx, link in enumerate(lattice):
        # Initialise delta distribution
        dist0 = np.zeros(2 * N + 1)
        centre = N
        dist0[centre] = 1.0
        state = TickState(dist0, N)
        ((x, y), mu) = link
        watch_idx = (x + y + mu) % (2 * N + 1)
        flip_count = 0
        # Perform a random walk: for each step apply all operators in
        # random order.  Shuffling the operator sequence ensures each
        # cycle sees a different ordering.
        for _ in range(steps_per_link):
            rng.shuffle(ops)
            for op in ops:
                new_state = op(state)
                # Count changes at the watch index
                if not np.isclose(new_state.distribution[watch_idx], state.distribution[watch_idx]):
                    flip_count += 1
                state = new_state
        counts[idx] = flip_count
    return counts


def load_flip_counts(L: int, seed: int, cache: dict | None = None) -> np.ndarray:
    """Load or generate flip counts for a given lattice size and seed.

    In the current implementation there is no on‑disk cache; flip
    counts are generated on the fly via :func:`generate_flip_counts`.
    A simple in‑memory cache may be provided to avoid recomputation
    during a single run.  The context depth and number of steps per
    link are fixed to modest values to keep the simulation tractable.

    Parameters
    ----------
    L : int
        Lattice size.
    seed : int
        Random seed.
    cache : dict, optional
        Mapping from ``(L, seed)`` to flip count arrays.  If the
        combination is present the stored array is returned instead of
        regenerating counts.

    Returns
    -------
    ndarray
        Flip count array of length ``2*L*L``.
    """
    if cache is not None and (L, seed) in cache:
        return cache[(L, seed)]
    counts = generate_flip_counts(L, seed)
    if cache is not None:
        cache[(L, seed)] = counts
    return counts


def _get_generator(group: str) -> np.ndarray:
    """Return a diagonal generator matrix for a given gauge group.

    For SU(2) we use σ_z/2 = diag(1/2, -1/2).  For SU(3) we use a
    λ₃‑like generator diag(1, -1, 0).  This helper centralises the
    choice of diagonal generators used throughout kernel construction
    and exponentiation.

    Parameters
    ----------
    group : str
        Gauge group; one of ``'SU2'`` or ``'SU3'`` (case‑insensitive).

    Returns
    -------
    ndarray
        A square diagonal matrix of shape ``(N, N)`` with real dtype.
    """
    g = group.upper()
    if g == "SU2":
        return np.array([[0.5, 0.0], [0.0, -0.5]], dtype=float)
    elif g == "SU3":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    else:
        raise ValueError(f"Unsupported group: {group}")


def compute_kernel_eigs(D_vals: np.ndarray) -> np.ndarray:
    """Compute per‑dimension spectral radii (ρ_i) from fractal dimensions.

    This routine follows the prescription in Volume 3: for each
    fractal dimension ``D_i`` in ``D_vals`` it constructs a
    tridiagonal matrix ``M^(i)`` of size ``N × N`` (where
    ``N=len(D_vals)``) with diagonal entries ``D_i - 2*g(D_i)`` and
    off‑diagonals ``g(D_i)``.  The spectral radius (largest absolute
    eigenvalue) of each ``M^(i)`` is returned.  The pivot function
    ``g(D)`` is fixed to ``D/3`` according to the kernel specification.

    Parameters
    ----------
    D_vals : ndarray
        One‑dimensional array of fractal dimensions.

    Returns
    -------
    ndarray
        Array of spectral radii ``rho`` of the same length as
        ``D_vals``.
    """
    import numpy.linalg as la

    D_vals = np.asarray(D_vals, dtype=float)
    N = D_vals.size
    rhos = np.zeros_like(D_vals, dtype=float)
    for idx, Di in enumerate(D_vals):
        gi = Di / 3.0
        # Construct tridiagonal matrix with constant diagonal and off‑diagonals
        M = np.zeros((N, N), dtype=float)
        for j in range(N):
            M[j, j] = Di - 2.0 * gi
            if j > 0:
                M[j, j - 1] = gi
            if j < N - 1:
                M[j, j + 1] = gi
        eigs = la.eigvals(M)
        rhos[idx] = float(np.max(np.abs(eigs)))
    return rhos


def build_kernel_from_rhos(rhos: np.ndarray, L: int, group: str) -> np.ndarray:
    """Construct a per‑link kernel array from spectral radii.

    Given an array of spectral radii ``rhos`` derived from fractal
    dimensions, this helper tiles or truncates ``rhos`` to match the
    number of oriented links on an ``L×L`` lattice (``2*L*L``).  The
    resulting scalar sequence is multiplied by the diagonal generator
    appropriate for the gauge group to produce a matrix‑valued kernel
    of shape ``(num_links, N, N)``.

    Parameters
    ----------
    rhos : ndarray
        One‑dimensional array of spectral radii (real positive).
    L : int
        Lattice size (side length).  The number of links is
        ``2*L*L``.
    group : str
        Gauge group (``'SU2'`` or ``'SU3'``).

    Returns
    -------
    ndarray
        Kernel array of shape ``(2*L*L, N, N)`` with real dtype.
    """
    rhos = np.asarray(rhos, dtype=float)
    num_links = 2 * L * L
    # Tile the spectral radii to match the number of links
    reps = int(np.ceil(num_links / rhos.size))
    rhos_tiled = np.tile(rhos, reps)[:num_links]
    gen = _get_generator(group)
    # Broadcast rhos into matrix form: each link gets rhos_tiled[i] * gen
    N = gen.shape[0]
    kernel = np.zeros((num_links, N, N), dtype=float)
    for i in range(num_links):
        kernel[i] = rhos_tiled[i] * gen
    return kernel


def build_constant_kernel(num_links: int, group: str) -> np.ndarray:
    """Construct a constant diagonal kernel for a gauge group.

    This function is retained for backward compatibility.  It returns
    a diagonal kernel whose entries are independent of the link index.
    For SU(2) the generator is σ_z/2 and for SU(3) it is a λ₃‑like
    generator.  The kernel array has shape ``(num_links, N, N)`` and
    can be multiplied by a pivot weight array externally.

    Parameters
    ----------
    num_links : int
        Number of links in the lattice (``2*L*L``).
    group : str
        Gauge group; one of ``'SU2'`` or ``'SU3'``.

    Returns
    -------
    ndarray
        Kernel array of shape ``(num_links, N, N)``.
    """
    gen = _get_generator(group)
    N = gen.shape[0]
    kernel = np.broadcast_to(gen, (num_links, N, N)).copy()
    return kernel


def exponentiate_potential(gD: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Exponentiate gauge potentials to obtain link matrices.

    Given a pivot weight array ``gD`` of length ``num_links`` and a
    constant diagonal kernel array of shape ``(num_links, N, N)``, this
    routine computes the gauge link matrices ``U`` via

    .. math::

       U_i = \exp( i \cdot gD_i \cdot K )

    Because ``K`` is diagonal the exponentiation reduces to computing
    the complex exponential of each diagonal entry individually.  Off‑
    diagonal elements remain zero.  The returned array has shape
    ``(num_links, N, N)`` with complex dtype.

    Parameters
    ----------
    gD : ndarray
        Pivot weight per link (real valued).
    kernel : ndarray
        Constant diagonal kernel, shape ``(num_links, N, N)``.

    Returns
    -------
    ndarray
        Link matrices ``U`` with shape ``(num_links, N, N)`` and
        complex dtype.
    """
    num_links, N, _ = kernel.shape
    # Ensure gD shape matches
    gD = np.asarray(gD, dtype=float).reshape(num_links)
    # Preallocate complex array
    U = np.zeros_like(kernel, dtype=complex)
    # For each link exponentiate diagonal entries
    for i in range(num_links):
        diag = kernel[i].diagonal()
        phases = np.exp(1j * gD[i] * diag)
        U[i] = np.diag(phases)
    return U


def compute_sha256_of_array(arr: np.ndarray) -> str:
    """Compute the SHA‑256 hash of a numpy array.

    The array is first converted to its bytes representation via
    ``tobytes()``.  The resulting digest is returned as a hex string.
    
    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    str
        Hexadecimal SHA‑256 digest of the array contents.
    """
    m = hashlib.sha256()
    m.update(arr.tobytes())
    return m.hexdigest()


def compute_sha256_of_obj(obj) -> str:
    """Compute the SHA‑256 hash of a JSON‑serialisable object.

    The object is serialised to a UTF‑8 encoded JSON string with
    sorted keys to guarantee determinism.  The resulting digest is
    returned as a hex string.

    Parameters
    ----------
    obj : any
        JSON‑serialisable Python object.

    Returns
    -------
    str
        Hexadecimal SHA‑256 digest of the serialised object.
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode('utf-8')
    return hashlib.sha256(s).hexdigest()