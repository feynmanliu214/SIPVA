#!/bin/bash
# Run the 50-system white-noise TDV grid at LLR/SNR = 5 (extends the 10/20/30/50/100 grid).
# Generate the data first:  .venv/bin/python scripts/generate_synthetic_systems.py --targets 5 --n-systems 50 --seed 5000
# Submit FROM the scripts/ dir:  cd scripts && sbatch -A TG-ATM170020 run_snr5_tdv.slurm
#SBATCH -J snr5_tdv
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00          # 50 systems in one wave; generous cap so the largest (~729 transits) finishes
#SBATCH -o snr5_tdv.%j.out
#SBATCH -e snr5_tdv.%j.err
# (#SBATCH -A TG-ATM170020   # pass -A on the sbatch command line, or uncomment here)

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=50
export SNR_LEVELS=5                 # the LLR=5 set lives in lightcurves/SNR_5 + SNR_5.csv
export SNR_SUMMARY_TAG=_snr5        # writes tdv_recovery_summary_snr5.csv (no clobber of other runs)
export SNR_SEED_BASE=55000          # distinct seed base for this run's prior offsets

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) workers=$SNR_N_WORKERS levels=$SNR_LEVELS"
../.venv/bin/python multiprocess.py
