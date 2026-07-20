#!/bin/bash
# Re-run the KOIs that failed in the first batch (job 3190299), on one Stampede3 spr node.
#   142.01, 377.02                  : absent from DR25 -> now from the `cumulative` table (get_koi fallback).
#   103.01, 841.02, 872.01, 1320.01 : MA-quadratic ZeroDivisionError -> now rejected by the
#                                     reject-on-raise guard in fitting.logprob_dbdt/neg_log_likelihood.
#   209.02                          : PyTransit per-transit broadcast error -> retried with fresh
#                                     per-transit (TDV_SEED_BASE) + global (KOI_GLOBAL_SEED) RNG.
# Submit FROM the scripts/ dir:
#     cd scripts && sbatch -A TG-ATM170020 run_koi_tdv_rerun.slurm
#SBATCH -J koi_tdv_rerun
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o koi_tdv_rerun.%j.out
#SBATCH -e koi_tdv_rerun.%j.err
# (#SBATCH -A TG-ATM170020   # pass -A on the sbatch command line, or uncomment here)

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export KOI_LIST="209.02"             # final holdout: degenerate 1-point segment now dropped (MIN_TRANSIT_POINTS)
export KOI_OUTER_WORKERS=1           # single KOI -> run in-process, give all cores to its per-transit pool
export TDV_N_WORKERS=48
export TDV_MAKE_PLOTS=1
export TDV_SEED_BASE=67890           # fresh per-transit seed (was 12345) -> retry 209.02
export KOI_GLOBAL_SEED=67890         # reseed the global db/dt fit too -> retry 872.01
export TDV_DETREND=gp                # detrending method: gp (default) | savgol

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD kois=$KOI_LIST outer=$KOI_OUTER_WORKERS inner=$TDV_N_WORKERS seed=$TDV_SEED_BASE gseed=$KOI_GLOBAL_SEED detrend=$TDV_DETREND"
../.venv/bin/python run_koi_batch.py
