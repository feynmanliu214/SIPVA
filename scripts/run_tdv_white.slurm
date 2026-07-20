#!/bin/bash
# TDV fit on the WHITE-noise SNR grid: SNR 0/5/10/20/30/50, 50 systems each = 300 fits.
# Reads data/SNR_data/lightcurves/SNR_<level>/<name>.npz (truth: SNR_<level>.csv), then relocates
# the per-system koi-syn_snr* outputs into data/Output_data/white_noise/.
# Submit FROM the scripts/ dir:  cd scripts && sbatch -A TG-ATM170020 run_tdv_white.slurm
#SBATCH -J tdv_white
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the worker pool itself
#SBATCH -t 08:00:00          # 300 systems, <=100 transits each -> a few waves of 110 workers
#SBATCH -o tdv_white.%j.out
#SBATCH -e tdv_white.%j.err
# (pass -A TG-ATM170020 on the sbatch command line, or uncomment:)
# #SBATCH -A TG-ATM170020

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=110
export SNR_LC_SET=white               # reads lightcurves/  (asserts white-noise npz keys)
export SNR_LEVELS=0,5,10,20,30,50     # 0 = the db/dt=0 null (false-positive test)
export SNR_SUMMARY_TAG=_white         # -> data/SNR_data/tdv_recovery_summary_white.csv
export SNR_SEED_BASE=12345            # prior-offset seed (shared with the realistic run)
# SNR_NAME_PREFIX unset -> un-prefixed koi-syn_snr* output folders.

cd "$SLURM_SUBMIT_DIR"   # must be the repo's scripts/ dir -- per-system outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD workers=$SNR_N_WORKERS levels=$SNR_LEVELS set=$SNR_LC_SET"
../.venv/bin/python multiprocess.py

# Relocate this run's per-system outputs into their own folder. Safe post-fit: multiprocess.py's
# aggregation already read the flat koi-* dirs before exiting. Glob koi-syn_snr* is disjoint from the
# realistic run's koi-real_syn_snr*, so the two jobs never touch each other's folders.
DEST=../data/Output_data/white_noise
mkdir -p "$DEST"
shopt -s nullglob
outs=(../data/Output_data/koi-syn_snr*)
if (( ${#outs[@]} )); then
    mv "${outs[@]}" "$DEST"/
    echo "[done] moved ${#outs[@]} white output dirs -> $DEST"
else
    echo "[warn] no koi-syn_snr* output dirs found to relocate"
fi
