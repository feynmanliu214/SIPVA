#!/bin/bash
# Run the 300-system realistic (Kepler-noise) TDV grid at LLR/SNR = 0, 5, 10, 20, 30, 50.
# Generate the missing realistic sets first (10/20/30/50 already exist):
#   .venv/bin/python scripts/generate_synthetic_systems.py --noise-model realistic --targets 5 --n-systems 50 --seed 5000
#   .venv/bin/python scripts/generate_null_systems_realistic.py
# Submit FROM the scripts/ dir:  cd scripts && sbatch -A TG-ATM170020 run_realistic_tdv.slurm
#SBATCH -J real_tdv
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 18:00:00          # 300 systems, ~3 waves; padded so the largest (~960 transits) finishes
#SBATCH -o real_tdv.%j.out
#SBATCH -e real_tdv.%j.err
# (#SBATCH -A TG-ATM170020   # pass -A on the sbatch command line, or uncomment here)

set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=110
export SNR_LEVELS=0,5,10,20,30,50    # 0 = realistic null; curves live in lightcurves_realistic/SNR_<level>
export SNR_LC_SET=realistic          # read lightcurves_realistic/ (white+red Kepler noise)
export SNR_NAME_PREFIX=real_         # outputs -> koi-real_syn_snr*; never collide with the white run
export SNR_SUMMARY_TAG=_realistic    # writes tdv_recovery_summary_realistic.csv
export SNR_SEED_BASE=77000           # population-level comparison (prior offsets independent of white)

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) workers=$SNR_N_WORKERS levels=$SNR_LEVELS set=$SNR_LC_SET"
../.venv/bin/python multiprocess.py
