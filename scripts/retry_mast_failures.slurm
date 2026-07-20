#!/bin/bash
# One-off retry of the 3 KOIs that failed the 16-KOI GP run (job 3205214) on a transient MAST 503
# (get_light_curve returned 0 products -> get_transit_arrays crashed on np.concatenate([])). NOT a
# code regression: 139.01 reproduces cleanly when MAST is up. Writes into the SAME Output_data_gp
# root (these koi-* dirs do not exist yet, so it only ADDS them). run_koi_batch.py rewrites
# tdv_batch_summary.csv per-invocation, so this would clobber the 13 good rows -- the summary is
# rebuilt from all 16 per-KOI JSONs afterward (see rebuild step), not trusted from this run.
# Submit FROM scripts/:   cd scripts && sbatch retry_mast_failures.slurm
#SBATCH -J koi_gp_retry
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-ATM170020
#SBATCH -o koi_gp_retry.%j.out
#SBATCH -e koi_gp_retry.%j.err

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export KOI_OUTER_WORKERS=3
export TDV_N_WORKERS=13
export TDV_MAKE_PLOTS=1
export TDV_DETREND=gp
export TDV_OUTPUT_ROOT=../data/Output_data_gp
export KOI_LIST="139.01,806.01,872.01"

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD outer=$KOI_OUTER_WORKERS inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"
echo "KOI_LIST=$KOI_LIST"
../.venv/bin/python run_koi_batch.py
