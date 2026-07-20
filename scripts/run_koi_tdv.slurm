#!/bin/bash
# Run the TDV pipeline on a list of REAL Kepler KOIs in parallel, on one Stampede3 spr node.
# Submit FROM the scripts/ dir (the pipeline's per-KOI output helpers write to ../data/Output_data):
#     cd scripts && sbatch -A TG-ATM170020 run_koi_tdv.slurm
#SBATCH -J koi_tdv
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the outer KOI pool itself
#SBATCH -t 04:00:00          # 16 KOIs, ~2 outer waves; raise if a KOI has very many transits
#SBATCH -o koi_tdv.%j.out
#SBATCH -e koi_tdv.%j.err
# (#SBATCH -A TG-ATM170020   # pass -A on the sbatch command line, or uncomment here)

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

# Two-level parallelism: OUTER whole-KOI pool x INNER per-transit pool. 8 x 13 = 104 cores on the
# 112-core spr node, peak ~113 procs (well under the 300-proc ulimit). Tune for other node types.
export KOI_OUTER_WORKERS=8
export TDV_N_WORKERS=13
export TDV_MAKE_PLOTS=1                                  # per-KOI folded-transit + regression figures
export TDV_DETREND=gp                                    # detrending method: gp (default) | savgol
# export TDV_OUTPUT_ROOT=../data/Output_data_gp         # uncomment to isolate this run's outputs

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- per-KOI outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) outer=$KOI_OUTER_WORKERS inner=$TDV_N_WORKERS detrend=$TDV_DETREND"
../.venv/bin/python run_koi_batch.py
