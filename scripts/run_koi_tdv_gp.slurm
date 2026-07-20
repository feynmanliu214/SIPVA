#!/bin/bash
# Re-run the 16 real Kepler KOIs through the TDV pipeline with the GP detrender (TDV_DETREND=gp),
# on one Stampede3 spr node. Writes to a DEDICATED output root (../data/Output_data_gp) so the
# existing savgol results under data/Output_data/Kepler Planets/ are left untouched and the two can
# be compared side by side. The KOI list is run_koi_batch.py's built-in _KOIS (the same 16).
# Submit FROM the scripts/ dir:
#     cd scripts && sbatch run_koi_tdv_gp.slurm
#SBATCH -J koi_tdv_gp
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the outer KOI pool itself
#SBATCH -t 04:00:00          # 16 KOIs, ~2 outer waves; raise if a KOI has very many transits
#SBATCH -A TG-ATM170020
#SBATCH -o koi_tdv_gp.%j.out
#SBATCH -e koi_tdv_gp.%j.err

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

# Two-level parallelism: OUTER whole-KOI pool x INNER per-transit pool. 8 x 13 = 104 cores on the
# 112-core spr node, peak ~113 procs (well under the 300-proc ulimit). Tune for other node types.
export KOI_OUTER_WORKERS=8
export TDV_N_WORKERS=13
export TDV_MAKE_PLOTS=1                                  # per-KOI folded-transit + regression figures
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending
export TDV_OUTPUT_ROOT=../data/Output_data_gp           # dedicated root: keep savgol 'Kepler Planets/' intact

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) outer=$KOI_OUTER_WORKERS inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"
../.venv/bin/python run_koi_batch.py
