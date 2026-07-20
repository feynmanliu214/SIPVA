#!/bin/bash
# Run the 50-system NULL set (db/dt = 0) through the TDV pipeline -- false-positive / specificity test.
# Generate the null set first:  .venv/bin/python scripts/generate_null_systems.py
# Submit FROM the scripts/ dir:  cd scripts && sbatch -A TG-ATM170020 run_null_tdv.slurm
#SBATCH -J null_tdv
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00          # 50 systems in one wave; makespan set by the largest (~960 transits)
#SBATCH -o null_tdv.%j.out
#SBATCH -e null_tdv.%j.err
# (#SBATCH -A TG-ATM170020   # pass -A on the sbatch command line, or uncomment here)

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=50
export SNR_LEVELS=0                 # the null set lives in lightcurves/SNR_0 + SNR_0.csv
export SNR_SUMMARY_TAG=_null        # writes tdv_recovery_summary_null.csv (no clobber of the signal run)
export SNR_SEED_BASE=99000          # distinct seed base for the null run's prior offsets

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) workers=$SNR_N_WORKERS levels=$SNR_LEVELS"
../.venv/bin/python multiprocess.py
