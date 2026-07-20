#!/bin/bash
# Confirmation run: re-fit ONLY KOI 1856.01 with the corrupted epoch-31 segment dropped, to test
# whether epoch 31 drove db_dt_global to +0.059 (z=+2.26). Writes to a dedicated output root so the
# rhofix results stay intact. Submit FROM scripts/:   cd scripts && sbatch verify_ep31.slurm
#SBATCH -J ep31_drop
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver; it forks the inner per-transit pool itself
#SBATCH -t 01:00:00          # one KOI, 30 transits
#SBATCH -A TG-ATM170020
#SBATCH -o ep31_drop.%j.out
#SBATCH -e ep31_drop.%j.err

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export TDV_N_WORKERS=13                                  # inner per-transit pool
export TDV_MAKE_PLOTS=1
export TDV_DETREND=gp
export TDV_OUTPUT_ROOT=../data/Output_data_gp_rhofix_noep31   # dedicated: keep rhofix intact

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"
PYTHONPATH=../src/core ../.venv/bin/python verify_ep31_drop.py
