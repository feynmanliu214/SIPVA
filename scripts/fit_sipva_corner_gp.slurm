#!/bin/bash
# Re-fit one KOI's SIPVA (global db/dt) posterior with the GP detrender and render its ApJ corner
# plot, writing into the DEDICATED GP output root (../data/Output_data_gp) so the result lives with
# the rest of the GP batch (not the default ../data/Output_data savgol-era root). Runs
# fit_sipva_corner.py, which does the full pipeline fit with save_posterior=True and then the plot.
# Single KOI -> no outer KOI pool; the per-transit fork pool works as designed on a compute node.
# Submit FROM the scripts/ dir:
#     cd scripts && sbatch fit_sipva_corner_gp.slurm
#SBATCH -J sipva_377_gp
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the per-transit pool itself
#SBATCH -t 02:00:00          # one KOI: per-transit fits (parallel) + emcee 32x10000 global fit
#SBATCH -A TG-ATM170020
#SBATCH -o sipva_377_gp.%j.out
#SBATCH -e sipva_377_gp.%j.err

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export TDV_N_WORKERS=32                                  # per-transit pool (~33 transits, 112-core node)
export TDV_MAKE_PLOTS=1                                  # folded-transit + regression figures too
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending
export TDV_OUTPUT_ROOT=../data/Output_data_gp           # dedicated GP root (the point of this re-run)

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"
../.venv/bin/python fit_sipva_corner.py 377.02
