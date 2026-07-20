#!/bin/bash
# Run the 250-system white-noise TDV grid on one Stampede3 spr node.
# Submit FROM the scripts/ dir (the pipeline's per-system output helpers write to ../data/Output_data):
#     cd scripts && sbatch run_snr_tdv.slurm
#SBATCH -J snr_tdv
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the worker pool itself
#SBATCH -t 08:00:00          # set from the calibration run (largest system sets the makespan); 48 h cap
#SBATCH -o snr_tdv.%j.out
#SBATCH -e snr_tdv.%j.err
# (#SBATCH -A <allocation>   # uncomment + set if your default allocation is not configured)

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=110
export SNR_SEED_BASE=12345

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- per-system outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) workers=$SNR_N_WORKERS"
../.venv/bin/python multiprocess.py
