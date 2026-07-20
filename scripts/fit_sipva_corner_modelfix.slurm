#!/bin/bash
# Staged validations 1+2 of the 2026-07-19 model corrections (docs/2026-07-19_model_corrections_plan.md),
# on branch fix/model-corrections-20260719:
#   1. KOI 103.01 full SIPVA re-fit, --fit-only  (boundary-truncation check: the db/dt posterior
#      must no longer pile up at the removed +/-0.075 bound, and the +/-1.0 safety support must
#      be inactive);
#   2. KOI 377.02 full SIPVA re-fit, --fit-only, then the versioned corner plot
#      (eccentric + exposure-integrated model).
# Outputs go ONLY to the staged validation subroot (../data/Output_data_modelfix_20260719/staged);
# production roots (Output_data, Output_data_gp) are never written. The 16-KOI batch is on hold.
# Submit FROM the scripts/ dir:
#     cd scripts && sbatch fit_sipva_corner_modelfix.slurm
#SBATCH -J sipva_modelfix
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the per-transit pool itself
#SBATCH -t 08:00:00          # 95 + 35 per-transit fits (parallel) + two emcee 32x10000 global fits
#SBATCH -A TG-ATM170020
#SBATCH -o sipva_modelfix.%j.out
#SBATCH -e sipva_modelfix.%j.err

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export TDV_N_WORKERS=32                                  # per-transit pool (112-core SPR node)
export TDV_MAKE_PLOTS=1                                  # folded-transit + regression figures too
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending
export TDV_OUTPUT_ROOT=../data/Output_data_modelfix_20260719/staged   # staged validation subroot

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT branch=$(git branch --show-current)"

# 1. KOI 103.01 validation (fit-only: the corner plotter is specialized to KOI 377.02)
../.venv/bin/python fit_sipva_corner.py 103.01 --fit-only

# 2. KOI 377.02 validation + versioned corner plot
../.venv/bin/python fit_sipva_corner.py 377.02 --fit-only
../.venv/bin/python plot_sipva_corner.py 377.02 \
    --outdir ../data/Output_data_modelfix_20260719/staged/koi-377.02/corner_v4_modelfix

echo "STAGED VALIDATIONS 1+2 DONE"
