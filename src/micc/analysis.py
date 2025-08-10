"""Statistical analysis utilities for MICC outputs.

This module contains routines to perform simple monotonicity tests and
to fit exponential decay models to visibility data.  These helpers
return dictionaries summarising the fitted parameters and whether
observed trends meet the expected sign criteria.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple


def test_monotonicity(f_values: Iterable[float], y_values: Iterable[float], expectation: str) -> Dict[str, float | bool]:
    """Test whether a sequence y(f) is monotonic in f.

    A linear regression is performed on the pairs ``(f, y)``.  The
    slope and its standard error are estimated via ordinary least
    squares.  The result includes a boolean flag ``is_monotonic`` that
    is True if the slope has the expected sign with 95% confidence.

    Parameters
    ----------
    f_values : iterable of float
        Measurement strengths (independent variable).
    y_values : iterable of float
        Observables corresponding to ``f_values``.
    expectation : {'increase', 'decrease'}
        Expected monotonic trend; determines the sign of the slope.

    Returns
    -------
    dict
        Contains the fitted slope, intercept, slope standard error,
        95% confidence interval half‑width, and a boolean ``is_monotonic``.
    """
    f = np.asarray(list(f_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    # Perform linear regression y = a + b f
    X = np.column_stack([np.ones_like(f), f])
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    # Degrees of freedom
    n = len(f)
    dof = n - 2
    if dof > 0 and residuals.size > 0:
        s_err = np.sqrt(residuals[0] / dof)
        cov = s_err ** 2 * np.linalg.inv(X.T @ X)
        slope_err = np.sqrt(cov[1, 1])
    else:
        # Not enough points to estimate error
        slope_err = np.nan
    # 95% CI half width (approx using z=1.96)
    half_width = 1.96 * slope_err if not np.isnan(slope_err) else np.nan
    # Check monotonicity
    if expectation == 'increase':
        is_monotonic = (slope - half_width) > 0
    elif expectation == 'decrease':
        is_monotonic = (slope + half_width) < 0
    else:
        raise ValueError("expectation must be 'increase' or 'decrease'")
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'slope_err': float(slope_err) if not np.isnan(slope_err) else float('nan'),
        'ci_half_width': float(half_width) if not np.isnan(half_width) else float('nan'),
        'is_monotonic': bool(is_monotonic),
    }


def fit_exponential_decay(f_values: Iterable[float], V_values: Iterable[float]) -> Dict[str, float]:
    """Fit an exponential decay model V(f) ≈ V0 * exp(-α f).

    Taking the natural logarithm yields ``log(V) = log(V0) - α f``.  We
    perform linear regression on ``log(V)`` versus ``f`` to estimate
    ``alpha`` and ``log(V0)``.  Standard errors are computed as in
    :func:`test_monotonicity` and returned for ``alpha``.

    Parameters
    ----------
    f_values : iterable of float
        Measurement strengths.
    V_values : iterable of float
        Visibility values (must be positive).

    Returns
    -------
    dict
        Contains ``alpha``, ``alpha_err``, ``V0``, and ``V0_err`` (log
        transformed errors are propagated to the amplitude domain).
    """
    f = np.asarray(list(f_values), dtype=float)
    V = np.asarray(list(V_values), dtype=float)
    # Filter out non‑positive values to avoid log issues
    mask = V > 0
    f = f[mask]
    V = V[mask]
    logV = np.log(V)
    X = np.column_stack([np.ones_like(f), f])
    beta, residuals, rank, s = np.linalg.lstsq(X, logV, rcond=None)
    intercept, slope = beta
    alpha = -slope
    logV0 = intercept
    # Errors
    n = len(f)
    dof = n - 2
    if dof > 0 and residuals.size > 0:
        s_err = np.sqrt(residuals[0] / dof)
        cov = s_err ** 2 * np.linalg.inv(X.T @ X)
        slope_err = np.sqrt(cov[1, 1])
        intercept_err = np.sqrt(cov[0, 0])
    else:
        slope_err = np.nan
        intercept_err = np.nan
    alpha_err = float(slope_err) if not np.isnan(slope_err) else float('nan')
    V0 = float(np.exp(logV0))
    V0_err = float(np.exp(logV0) * intercept_err) if not np.isnan(intercept_err) else float('nan')
    return {
        'alpha': float(alpha),
        'alpha_err': float(alpha_err),
        'V0': V0,
        'V0_err': V0_err,
    }