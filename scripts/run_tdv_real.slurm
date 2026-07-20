#!/bin/bash
# TDV fit on the REALISTIC (white + CDPP-calibrated red) SNR grid: SNR 0/5/10/20/30/50, 50 each = 300 fits.
# Reads data/SNR_data/lightcurves_realistic/SNR_<level>/<name>.npz (truth: the SAME SNR_<level>.csv as
# the white run -- realistic reuses those systems), then relocates the per-system koi-real_syn_snr*
# outputs into data/Output_data/real_kepler_noise/.
# Submit FROM the scripts/ dir:  cd scripts && sbatch -A TG-ATM170020 run_tdv_real.slurm
#SBATCH -J tdv_real
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the worker pool itself
#SBATCH -t 08:00:00          # 300 systems, <=100 transits each -> a few waves of 110 workers
#SBATCH -o tdv_real.%j.out
#SBATCH -e tdv_real.%j.err
# (pass -A TG-ATM170020 on the sbatch command line, or uncomment:)
# #SBATCH -A TG-ATM170020

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export SNR_N_WORKERS=110
export SNR_LC_SET=realistic           # reads lightcurves_realistic/  (white + red noise)
export SNR_LEVELS=0,5,10,20,30,50     # 0 = the db/dt=0 null (false-positive test)
export SNR_NAME_PREFIX=real_          # koi-real_syn_snr* -> no collision with the white run's flat dirs
export SNR_SUMMARY_TAG=_real          # -> data/SNR_data/tdv_recovery_summary_real.csv
export SNR_SEED_BASE=12345            # SAME prior-offset seed as white: isolates the noise-model effect

cd "$SLURM_SUBMIT_DIR"   # must be the repo's scripts/ dir -- per-system outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD workers=$SNR_N_WORKERS levels=$SNR_LEVELS set=$SNR_LC_SET prefix=$SNR_NAME_PREFIX"
../.venv/bin/python multiprocess.py

# Relocate this run's per-system outputs into their own folder. Safe post-fit: aggregation already read
# the flat koi-* dirs. Glob koi-real_syn_snr* is disjoint from the white run's koi-syn_snr*.
DEST=../data/Output_data/real_kepler_noise
mkdir -p "$DEST"
shopt -s nullglob
outs=(../data/Output_data/koi-real_syn_snr*)
if (( ${#outs[@]} )); then
    mv "${outs[@]}" "$DEST"/
    echo "[done] moved ${#outs[@]} realistic output dirs -> $DEST"
else
    echo "[warn] no koi-real_syn_snr* output dirs found to relocate"
fi
