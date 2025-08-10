"""MICC simulation package.

This package contains the modules required to perform the
measurement‑induced context collapse (MICC) simulation on top of the
Fractal Pivot Hamiltonian System (FPHS) gauge core.  Modules are
organized by responsibility:

* :mod:`measurement` – Implements the measurement channel and noise
  injection used to partially read out links during the simulation.
* :mod:`loops` – Routines for building the lattice, computing Wilson
  loops, and estimating the string tension via Creutz ratios.
* :mod:`interference` – Functions to construct gauge‑invariant two‑path
  interferometers and extract the visibility from twist scans.
* :mod:`purity` – Coherence and purity proxies derived from per‑link
  matrices.
* :mod:`utils_fphs` – Common utilities: loading flip counts, logistic
  dimension computation, kernel construction, exponentiation of gauge
  potentials and dephasing noise.
* :mod:`analysis` – Statistical analysis routines including
  monotonicity tests and exponential fits.

The top‑level orchestrator script :mod:`run_micc` ties together
configuration parsing, data preparation, simulation, and result
serialization.

All functions are typed where feasible and strive to be pure and
side‑effect free (aside from random number generation and I/O).
"""

from .measurement import apply_dephasing
from .loops import (build_lattice,
                    measure_wilson_loops,
                    estimate_string_tension)
from .interference import (build_twist_scan,
                           fit_visibility)
from .purity import (compute_coherence_factor,
                     compute_spectral_entropy)
from .utils_fphs import (logistic_dimension,
                         pivot_weight,
                         load_flip_counts,
                         build_constant_kernel,
                         exponentiate_potential)
from .analysis import (test_monotonicity,
                       fit_exponential_decay)

__all__ = [
    'apply_dephasing',
    'build_lattice',
    'measure_wilson_loops',
    'estimate_string_tension',
    'build_twist_scan',
    'fit_visibility',
    'compute_coherence_factor',
    'compute_spectral_entropy',
    'logistic_dimension',
    'pivot_weight',
    'load_flip_counts',
    'build_constant_kernel',
    'exponentiate_potential',
    'test_monotonicity',
    'fit_exponential_decay',
]