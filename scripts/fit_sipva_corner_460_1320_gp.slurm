#!/bin/bash
# Re-fit the SIPVA (global db/dt) posteriors for KOI 460.01 and 1320.01 with the GP detrender and
# render their ApJ corner plots, writing into the DEDICATED GP output root (../data/Output_data_gp)
# alongside their existing GP results. Each KOI runs fit_sipva_corner.py -> full pipeline fit with
# save_posterior=True, then the corner plot. The two KOIs run SEQUENTIALLY in this single job; each
# fit forks its own per-transit pool, which works as designed on a compute node (it deadlocks on the
# login node, so this MUST go through sbatch).
# Submit FROM the scripts/ dir:
#     cd scripts && sbatch fit_sipva_corner_460_1320_gp.slurm
#SBATCH -J sipva_460_1320_gp
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the per-transit pool itself
#SBATCH -t 03:00:00          # two KOIs sequentially (73 + 84 transits); the 33-transit 377 run took ~14 min
#SBATCH -A TG-ATM170020
#SBATCH -o sipva_460_1320_gp.%j.out
#SBATCH -e sipva_460_1320_gp.%j.err

set -uo pipefail             # NOT -e: one KOI failing must not skip the other; failures tracked below

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

export TDV_N_WORKERS=48                                  # per-transit pool: 2 waves for 73/84 transits on a 112-core node
export TDV_MAKE_PLOTS=1                                  # folded-transit + regression figures too
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending
export TDV_OUTPUT_ROOT=../data/Output_data_gp           # dedicated GP root (matches these KOIs' existing results)

cd "$SLURM_SUBMIT_DIR"       # must be the repo's scripts/ dir -- outputs are '..'-relative
echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"

rc=0
for koi in 460.01 1320.01; do
    echo "=== $(date -u +%FT%TZ) starting SIPVA fit for KOI $koi ==="
    ../.venv/bin/python fit_sipva_corner.py "$koi"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "=== KOI $koi: OK ==="
    else
        echo "=== KOI $koi: FAILED (exit $status) ===" >&2
        rc=1
    fi
done
echo "=== $(date -u +%FT%TZ) all done, rc=$rc ==="
exit $rc
