# MICC‑FPHS Simulation

This repository implements the **Measurement‑Induced Context Collapse (MICC)**
simulation on top of a minimal FPHS gauge core.  It follows the locked
specification provided by the user and produces a full dataset across
lattice sizes, inverse couplings, pivot slopes, measurement strengths and
gauge groups.  The key goals of the simulation are to:

* Introduce a **measurement axis** `f` that injects gauge‑covariant
  dephasing on a fraction of links during Wilson‑loop and interference
  evaluations.
* Quantify how the **string tension** σ, **interference visibility** V and
  **coherence proxies** respond to the measurement strength across
  multiple FPHS parameter settings.
* Fit an exponential collapse rate α via `V(f) ≈ V₀ exp(−α f)` and test
  monotonicity of σ and coherence proxies with respect to `f`.

The simulation deliberately avoids using stubbed or placeholder values.
Flip counts are generated exactly as in Volume 4 using the tick‑flip
operator algebra.  The fractal dimension `D(n)` is computed from the
logistic pivot with slope κ and midpoint 0, and the pivot weight
`g(D)` is chosen to be `D/3` multiplied by the inverse coupling `b`.
Validated kernel eigenvalue spectra were unavailable in the provided
repositories; therefore constant diagonal kernels are used for
SU(2) and SU(3), embedding one of the diagonal generators (σ_z/2 and
λ₃‑like, respectively).  This choice preserves the relative gauge
hierarchy while remaining consistent with the FPHS formalism.  Wilson
loops, interference scans and purity proxies are implemented from
first principles with an eye toward performance and reproducibility.

See `reports/REPORT.md` for detailed results, plots and acceptance
checks.

## Quick start

```bash
# create and activate a Python environment (optional)
python3 -m venv venv
. venv/bin/activate
pip install -r env/requirements.txt

# run the full grid (default parameters; may take several minutes)
bash scripts/run_all.sh

# aggregate results into a CSV (Parquet will be written only if
# optional dependencies such as `pyarrow` are installed)
python scripts/aggregate.py --input runs --csv results/summary.csv

# view the report (generated after the simulation)
less reports/REPORT.md
```

## Repository structure

```
micc-fphs/
├── README.md              # you are here
├── LICENSE                # MIT License
├── CITATION.cff           # citation metadata
├── env/
│   ├── requirements.txt   # Python dependencies
│   └── environment.yml    # Conda environment (optional)
├── configs/
│   └── fphs_anchors.yaml  # parameter grid (L, b, κ, gauge, f, schedule, seeds)
├── src/micc/
│   ├── __init__.py
│   ├── run_micc.py        # top‑level orchestrator
│   ├── measurement.py     # gauge‑covariant dephasing channel
│   ├── loops.py           # Wilson loops and string‑tension estimator
│   ├── interference.py    # twist scan and visibility fit
│   ├── purity.py          # coherence proxies (C‑factor and entropy)
│   ├── utils_fphs.py      # lattice, kernel and flip‑count helpers
│   └── analysis.py        # monotonicity tests and exponential fits
├── scripts/
│   ├── run_all.sh         # run the full grid end‑to‑end
│   └── aggregate.py       # aggregate JSON outputs into summary tables
├── data/                  # created at run time
├── runs/                  # JSON outputs per condition
├── results/plots/         # generated plots
└── reports/REPORT.md      # final report with acceptance checks
```
