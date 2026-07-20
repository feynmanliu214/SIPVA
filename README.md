# SIPVA — Simultaneous Impact Parameter Variation Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2411.06452-b31b1b.svg)](https://arxiv.org/abs/2411.06452)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Detecting secular perturbations in Kepler planetary systems by measuring
**transit impact-parameter variations** (db/dt) directly from the light curves.

This is the reference implementation for **Liu & Pu (2024)**,
*Detecting Secular Perturbations in Kepler Planetary Systems Using Simultaneous
Impact Parameter Variation Analysis (SIPVA)*
([arXiv:2411.06452](https://arxiv.org/abs/2411.06452)).

## Overview

A precessing planetary orbit changes the chord the planet traces across its host
star, so the transit **impact parameter** `b` — and hence the transit duration
`t14` — drifts slowly with time. SIPVA measures that drift by folding a linear
time-dependent impact-parameter model, `b(t) = b0 + (db/dt)·t`, **directly into
the MCMC** and fitting all transits of a Kepler Object of Interest (KOI)
simultaneously. This is more sensitive than fitting each transit independently
and avoids the cost of full N-body integrations.

The pipeline handles the full path from raw data to a significance-tested db/dt:

- **Light curves** are pulled from MAST via [Lightkurve](https://lightkurve.github.io/lightkurve/)
  and detrended per Kepler quarter (transit-masked Matérn-3/2 Gaussian Process, or Savitzky–Golay).
- **Transit models** use [PyTransit](https://pytransit.readthedocs.io/) (Mandel & Agol quadratic).
- **Priors** are built live from the NASA Exoplanet Archive DR25 KOI table (`q1_q17_dr25_koi`)
  via `astroquery`, with theory-derived limb-darkening priors from PyLDTk.
- **Two fits** are run per target: a per-transit *Individual Fit* (each `b` fit separately) and the
  *simultaneous* SIPVA fit (global `db/dt` via MLE + `emcee`), plus a per-transit regression cross-check.
- **Detectability** and injection-recovery tooling generate synthetic light curves (white or
  realistic red noise) to forecast SNR and validate the method.

### Model corrections (2026-07)

Four corrections to the real-Kepler fitting path (results published earlier predate them;
refits in progress):

- **Catalog-only SIPVA priors** (`sipva_prior_spec` in `src/core/priors.py`): `b0 ~ U(0,1)`,
  `db/dt ~ N(0, 0.2² yr⁻²)`; individual-fit posteriors now seed optimizers/walkers only and
  never enter the prior density. Catalog ± errors are mean-symmetrized (was quadrature).
- **db/dt boundary removed**: the old hard `|db/dt| ≤ 0.075 yr⁻¹` support is gone; the
  operative constraint is `0 ≤ b_j ≤ 1` per retained epoch, with a broad `±1 yr⁻¹`
  numerical guard (verified inactive).
- **Consistent eccentric geometry**: `cos i = (b/a)(1 + e·sinω)/(1 − e²)` on every
  real-Kepler path; individual fits use `EccentricTransitAnalysis`
  (`src/core/ta_eccentric.py`), whose derived `t14`/`t23` also use the sampled `e, ω`.
- **Finite-exposure integration**: Kepler long-cadence points are evaluated with 15-point
  exposure integration (exposure from `TIMEDEL` metadata, with class-constant fallback);
  short cadence is unchanged. Validation: `tests/test_model_corrections.py`.

## Citation

If you use this code, please cite the paper:

> Liu, Z. & Pu, B. (2024). *Detecting Secular Perturbations in Kepler Planetary Systems
> Using Simultaneous Impact Parameter Variation Analysis (SIPVA).*
> arXiv:2411.06452. <https://arxiv.org/abs/2411.06452>

```bibtex
@article{Liu2024SIPVA,
  title         = {Detecting Secular Perturbations in Kepler Planetary Systems Using
                   Simultaneous Impact Parameter Variation Analysis (SIPVA)},
  author        = {Liu, Zhixing and Pu, Bonan},
  year          = {2024},
  eprint        = {2411.06452},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.EP},
  doi           = {10.48550/arXiv.2411.06452},
  url           = {https://arxiv.org/abs/2411.06452}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## Repository layout

```
src/core/    importable library — single-purpose modules (flat-absolute imports)
  pipeline.py        TDV / db-dt pipeline entry points (get_time_and_flux, TDV_fit, execute_TDV_func)
  data.py            NASA Exoplanet Archive KOI adapter + light-curve download, ephemerides, detrending
  model.py           forward transit-flux model + stellar-density / uncertainty helpers
  priors.py          parameter priors (catalog-derived / synthetic / theory-derived)
  fitting.py         log-prior + log-likelihoods; per-transit and global db/dt fits (MLE + emcee)
  analysis.py        impact-parameter / duration trend regression, significance tests, result plots
  gp_detrend.py      transit-masked Matérn-3/2 GP baseline detrending (celerite2, non-JAX)
  limb_darkening.py  theory-derived (PyLDTk) Kipping q1/q2 limb-darkening priors
  noise.py           correlated (red) noise + observational gaps for synthetic light curves
  detectability.py   SNR / detectability forecasting + synthetic light-curve generation
src/archive/  superseded / third-party, kept for reference
              (oblate-transit models; Mandel & Agol via I. Crossfield — verbatim)
scripts/     runnable entry points + SLURM batch scripts (run_tdv.py, run_koi_batch.py, generate_SNR.py, …)
notebooks/   analysis notebooks (SNR, TDV ROC); archive/ = superseded
tests/       prior golden-value + segment-contamination / keep-for-fit checks
requirements.txt  pinned dependencies
data/        light curves and fit outputs — NOT tracked; created at <repo>/data at runtime
```

## Installation

Developed and run on the [TACC Stampede3](https://www.tacc.utexas.edu/systems/stampede3/)
cluster with **Python 3.12** and a self-contained `venv` (no conda). Any Python 3.12
environment works.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Exact pinned versions are in [`requirements.txt`](requirements.txt). Key dependencies:

- Transit fitting: `pytransit`, `batman-package`
- Light curves / catalog: `lightkurve`, `astropy`, `astroquery` (NASA Exoplanet Archive)
- Detrending: `celerite2` (GP, non-JAX backend), `scipy`
- Limb darkening: `pyldtk`
- Sampling / posteriors: `emcee`, `corner`
- Core / plotting: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`

On an HPC module system (e.g. Stampede3), load the base Python module first, then
build the venv on top of it:

```bash
module load python/3.12.11
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The library uses **flat-absolute imports** and its modules read/write `../data/...`
**relative to the current working directory**, so run entry points from a first-level
subdirectory (not the repo root):

```bash
cd scripts
python run_tdv.py                   # TDV / db-dt fit for KOI 841.02 (default)
python run_tdv.py 99.01 841.02      # one or more KOIs
python generate_SNR.py              # SNR detectability forecasting
```

Running `python scripts/run_tdv.py` from the repo root would make `../data` resolve
*outside* the repo — always `cd scripts` first. The scripts add `src/core` to `sys.path`
automatically; notebooks run from `notebooks/` and include a one-line shim
(`sys.path.insert(... "../src/core")`).

### Detrending method

Real-KOI light curves are detrended per Kepler quarter by one of two methods, selectable with
`--detrend` (or the `TDV_DETREND` env var; the CLI flag wins):

- **`gp` (default)** — a transit-masked Matérn-3/2 Gaussian-Process baseline (`celerite2`, non-JAX).
  In-transit cadences are down-weighted by error inflation, the GP fits the out-of-transit stellar
  variability, and flux is divided by the predicted baseline. Best preserves transit shape (`t14`,
  `b`) for db/dt work and handles correlated red noise / spot rotation.
- **`savgol`** — polynomial-fill + Savitzky–Golay divide.

```bash
cd scripts
python run_tdv.py 841.02 --detrend gp        # default
python run_tdv.py 841.02 --detrend savgol    # fall back to Savitzky–Golay
TDV_DETREND=savgol python run_koi_batch.py   # whole-batch via env var
```

Per-quarter products too short or too gappy for a stable GP fit automatically fall back to savgol
(or median-normalization for pathologically short products); the synthetic SNR grids are unaffected
(they load pre-generated curves and bypass detrending). The method used is recorded as
`detrend_method` in each `tdv_metrics_koi_<X>.json` and in `tdv_batch_summary.csv`.

**Output isolation.** Set `TDV_OUTPUT_ROOT` to redirect all per-KOI outputs (default
`../data/Output_data`), e.g. to compare methods without clobbering:

```bash
cd scripts
TDV_DETREND=savgol TDV_OUTPUT_ROOT=../data/Output_data_savgol python run_koi_batch.py
TDV_DETREND=gp     TDV_OUTPUT_ROOT=../data/Output_data_gp     python run_koi_batch.py
```

## Output data

The pipeline writes per-KOI results to `data/Output_data/koi-<X>/`:

| File | Contents |
|---|---|
| `folded_transits_koi_<X>.{png,pdf}` | Folded light curve, points colored by transit epoch |
| `linear_regression_koi_<X>.{png,pdf}` | Impact-parameter and transit-duration regressions vs. time |
| `per_transit_fits_koi_<X>.csv` | Per-transit individual fits; leading `transit_number` (orbital epoch), then each parameter's `<param>_median/_lerr/_uerr` |
| `parameters_koi_<X>.csv` | Global-fit parameters, long format: `parameter,value,err_lower,err_upper,unit` |
| `tdv_metrics_koi_<X>.json` | Scalar db/dt metrics (see below) |

**Parameter symbols / units** (`parameters_koi_<X>.csv`): `rho` **effective (photometric) density**
ρ_eff ≡ ρ⋆ + p³ρ_p in g/cm³ — the transit shape constrains the density only through
(a/R⋆)³ = G P² ρ_eff/(3π), so the sampled quantity is the *total* density, not the stellar density
alone (the p³ρ_p term is ≈10⁻⁴ g/cm³ for KOI 377.02, far below the posterior width, so
ρ_eff ≈ ρ⋆ in practice); `tc` mid-transit time (day; pinned to 0 in the global fit), `p` period
(day), `k2` = (Rp/R⋆)², `secw` = √e·cos ω, `sesw` = √e·sin ω, `q1`/`q2` Kipping limb-darkening,
`b` impact parameter, `db_dt` change of `b` per year (1/yr). Errors are asymmetric
(`err_lower`/`err_upper`). See [`scripts/rhoeff_correction_audit.py`](scripts/rhoeff_correction_audit.py)
for the per-KOI audit of the p³ρ_p correction.

**Metrics** (`tdv_metrics_koi_<X>.json`): `num_transit`; `db_dt_global` ± `db_dt_global_err` and
`db_dt_global_zscore` (global-fit db/dt and its significance); `db_dt_linreg` ± `db_dt_linreg_err`
(per-transit linear-regression cross-check); `t_score_b` (impact-parameter slope significance);
`t_score_t14` (duration slope significance, `null` when too few transits for the duration fit);
`detrend_method` (`gp` or `savgol`, provenance; omitted for synthetic runs).
**All db/dt quantities are in `1/yr`.**

## Smoke test

Transit model + light-curve stack:

```bash
python -c "from pytransit import QuadraticModel; import lightkurve, numpy as np; \
t=np.linspace(-0.1,0.1,200); m=QuadraticModel(); m.set_data(t); \
print(m.evaluate(k=0.1, ldc=[0.3,0.2], t0=0., p=2.5, a=8., i=1.57).min())"
```

KOI catalog adapter (needs network; expects ≈31.33 d for K00841.02):

```bash
cd src/core && PYTHONPATH=. python -c "from data import get_koi; \
print(get_koi(841.02).koi_period)"
```

## License

Released under the [MIT License](LICENSE).
