MICC-FPHS: Measurement-Induced Context Collapse on the FPHS Gauge Core
MICC adds a measurement axis 
𝑓
f to the FPHS gauge core and injects gauge-covariant dephasing while we measure the same observables FPHS organizes—Wilson loops, a twist-scan interference, and coherence proxies. Turning up 
𝑓
f operationalizes AR’s idea that context readout collapses relational coherence and drives the system toward the pivot (centered at 
𝐷
=
2
D=2).

What we test

Interference: 
𝑉
(
𝑓
)
V(f) decreases, often 
𝑉
 ⁣
≈
 ⁣
𝑉
0
𝑒
−
𝛼
𝑓
V≈V 
0
​
 e 
−αf
 .

Confinement: 
𝜎
(
𝑓
)
σ(f) increases (Creutz ratios / area-law).

Coherence: entropy-based proxy 
𝑃
(
𝑓
)
P(f) decreases.

Hierarchy: 
𝜎
SU3
>
𝜎
SU2
σ 
SU3
​
 >σ 
SU2
​
  remains true.

Rates vs anchors: 
𝛼
α and 
∂
𝜎
/
∂
𝑓
∂σ/∂f track 
(
𝑏
,
𝜅
)
(b,κ).

What’s implemented (final run)
Channel: gauge-covariant dephasing 
𝑈
 ⁣
←
 ⁣
𝑒
𝑖
𝛿
𝐴
𝑈
U←e 
iδA
 U with Var(
𝛿
𝐴
δA)=
𝑓
f, fresh random subset each MCMC cycle.

Scheduling: during (primary), plus light pre/post controls.

Critical fix: in “during”, we apply dephasing while measuring (loops + coherence), not just before/after.

Strength: multi-pass kicks (measurement.kick_passes=3) compound the effect without breaking gauge structure.

Visibility: normalized twist scan (
𝜙
∈
[
0
,
2
𝜋
]
ϕ∈[0,2π], 64 points, 16 cycles), weighted least squares cosine fit; pooled exp-collapse fit across seeds for 
𝛼
α and 
𝑅
2
R 
2
 .

String tension 
𝜎
σ: multi-window Creutz averaging (2,2) & (3,3) with per-anchor validity filtering; jackknife errors; area-law cross-check retained.

Coherence: entropy-based proxy 
𝑃
:
=
1
−
𝐻
/
𝐻
max
⁡
P:=1−H/H 
max
​
  (primary) and C-factor (secondary).

Anchors: SU(2), SU(3) (fundamental), 
𝐿
∈
{
8
,
12
}
L∈{8,12}, 
𝑏
∈
{
3.0
,
3.5
}
b∈{3.0,3.5}, 
𝜅
∈
{
0.75
,
1.0
}
κ∈{0.75,1.0}, 
𝑓
∈
{
0
,
0.05
,
0.10
,
0.20
,
0.30
,
0.50
}
f∈{0,0.05,0.10,0.20,0.30,0.50}, 5 seeds.

Final results (headline)
Monotone trends (during):
V ↓: 16/16 σ ↑: 16/16 P ↓: 16/16

Gauge hierarchy @ 
𝑓
=
0
f=0: SU3 > SU2 in 8/8 matched anchors.

Pooled visibility fits (per anchor):
R² ≥ 0.8 in 13/16 anchors (≥0.5 in 14/16).

Collapse rate 
𝛼
α (pooled across seeds):

SU2: median 1.54 (range ~0.084–1.78)

SU3: median 2.02 (range ~1.58–4.11)

These satisfy the acceptance gates: monotonicities with 95% CI, pooled 
𝑅
2
≥
0.8
R 
2
 ≥0.8 in ≥1 anchor per gauge (in fact many), and the SU3>SU2 hierarchy at 
𝑓
=
0
f=0.

Quick start
Windows (Anaconda PowerShell Prompt)
powershell
Copy
Edit
conda activate fphs
cd C:\Users\kentn\vol5-micc-fphs
$env:PYTHONPATH = (Join-Path $PWD "src")

# Run the grid → JSON per condition in .\runs
python -m micc.run_micc --config configs\fphs_anchors.yaml --output runs

# Aggregate → CSV (+Parquet if pyarrow is present)
python scripts\aggregate.py --input runs --csv results\summary.csv --parquet results\summary.parquet

# Acceptance report (QC gates)
python scripts\accept.py --summary results\summary.csv --r2-min 0.8
Git Bash (optional)
bash
Copy
Edit
cd ~/vol5-micc-fphs
export PYTHONPATH="$(pwd)/src"
python -m micc.run_micc --config configs/fphs_anchors.yaml --output runs
python scripts/aggregate.py --input runs --csv results/summary.csv --parquet results/summary.parquet
python scripts/accept.py --summary results/summary.csv --r2-min 0.8
The folders runs/, results/, reports/ are created/filled on first run.

Configuration (key knobs)
configs/fphs_anchors.yaml (final run values):

Visibility: num_phi: 64, cycles: 16, eps: 1e-12, eps_scale: 3e-6

Measurement: selection: "uniform", kick_passes: 3

Loops/σ:
creutz_sizes: [[2,2],[3,3]], measurement_cycles: 6, block_size: 64, min_valid_fraction: 0.95

Grid: lattice_sizes: [8,12], inverse_couplings: [3.0,3.5], pivot_slopes: [0.75,1.0], gauge_groups: ["SU2","SU3"], f_grid as above, seeds: [0,1,2,3,4]

Repository structure
bash
Copy
Edit
micc-fphs/
├── README.md                 # this file
├── LICENSE
├── CITATION.cff
├── env/
│   ├── requirements.txt
│   └── environment.yml
├── configs/
│   └── fphs_anchors.yaml     # anchors, f-grid, and MICC knobs
├── src/micc/
│   ├── __init__.py
│   ├── run_micc.py           # orchestrator (measure-during, pooled fits)
│   ├── measurement.py        # gauge-covariant dephasing channel
│   ├── loops.py              # Wilson loops, multi-window Creutz
│   ├── interference.py       # twist scan (normalized), WLS cosine fit
│   ├── purity.py             # entropy proxy + C-factor
│   ├── utils_fphs.py         # FPHS helpers, hashing
│   └── analysis.py           # monotonicity + pooled V(f) fits
├── scripts/
│   ├── aggregate.py          # JSON → CSV/Parquet
│   └── accept.py             # QC gates + acceptance report
├── data/                     # inputs (e.g., D_values.csv)
├── runs/                     # JSON outputs per condition (created)
├── results/
│   ├── summary.csv           # aggregated table
│   ├── summary.parquet       # (if pyarrow installed)
│   └── plots/                # figures
└── reports/
    └── REPORT.md             # summary, plots, α tables, acceptance
Reproducibility & provenance
All input artifacts and configs are hashed (SHA-256) into the per-condition JSON and the acceptance report (see artifacts and MANIFEST.json if generated).

No placeholders: measurement is applied directly to the simulated link field derived from FPHS anchors; kernels/links are built from the validated FPHS pipeline and recorded by hash.

One-command reruns from a clean machine (Anaconda or pip), with acceptance gating automated by scripts/accept.py.

Theory linkage (why this matters for AR)
With a single knob 
𝑓
f, MICC shows coherent, monotone responses in interference, confinement, and coherence—and the rates scale with FPHS anchors 
(
𝑏
,
𝜅
)
(b,κ) and preserve the SU3>SU2 hierarchy. That gives operational teeth to AR’s claim: context readout is a real control variable that pushes the system toward the 
𝐷
 ⁣
→
 ⁣
2
D→2 pivot, re-weighting relational phases and tightening classical structure.

This README updates and supersedes the earlier version that referenced constant diagonal kernels and lacked the measure-during and pooled-fit details. 