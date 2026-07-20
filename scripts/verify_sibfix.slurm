#!/bin/bash
# Verification run for the segment-contamination fix
# (docs/2026-06-12_fix_segment_contamination_plan.md): sibling-transit masking (RC1), near-center
# coverage cut (RC2), baseline-quality guard (RC3), NaN-safe derived durations (RC4), plus the
# prior-dominated b audit. Re-runs the 6 affected KOIs + 2 clean controls through the GP pipeline
# into a DEDICATED output root (../data/Output_data_gp_sibfix) so prior snapshots
# (Output_data_gp, Output_data_gp_rhofix) are left intact for a before/after comparison.
# Submit FROM the scripts/ dir:   cd scripts && sbatch verify_sibfix.slurm   (account already set)
#SBATCH -J koi_sibfix
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; it forks the outer KOI pool itself
#SBATCH -t 04:00:00          # 8 KOIs, one outer wave; raise if 142.01 (126 transits) runs long
#SBATCH -A TG-ATM170020
#SBATCH -o koi_sibfix.%j.out
#SBATCH -e koi_sibfix.%j.err

set -euo pipefail

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

# Two-level parallelism: OUTER whole-KOI pool x INNER per-transit pool. 8 x 13 = 104 cores on the
# 112-core spr node, peak ~113 procs (well under the 300-proc ulimit).
export KOI_OUTER_WORKERS=8
export TDV_N_WORKERS=13
export TDV_MAKE_PLOTS=1                                  # per-KOI folded-transit + regression figures
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending
# Dedicated root for THIS fix. Changed from verify_rhofix.slurm's Output_data_gp_rhofix on purpose:
# a literal copy would guard-and-run against the existing rhofix root.
export TDV_OUTPUT_ROOT=../data/Output_data_gp_sibfix

# 6 affected KOIs (segment contamination in the prior GP/rhofix runs) + 2 clean controls (103.01, 2698.01).
export KOI_LIST="103.01,2698.01,841.02,142.01,1856.01,209.02,137.02,460.01"

cd "$SLURM_SUBMIT_DIR"        # must be the repo's scripts/ dir -- outputs are '..'-relative

# ENFORCED clean-root guard: the output writers create/overwrite fixed filenames IN PLACE inside an
# existing root, so a re-run into a populated root would silently mix two runs. Refuse and stop
# before run_koi_batch.py starts rather than after. Runs on the resolved TDV_OUTPUT_ROOT above.
test ! -e "$TDV_OUTPUT_ROOT" || { echo "refusing: $TDV_OUTPUT_ROOT exists"; exit 1; }

echo "host=$(hostname) cwd=$PWD RLIMIT_NPROC(ulimit -u)=$(ulimit -u) outer=$KOI_OUTER_WORKERS inner=$TDV_N_WORKERS detrend=$TDV_DETREND out=$TDV_OUTPUT_ROOT"
echo "KOI_LIST=$KOI_LIST"
../.venv/bin/python run_koi_batch.py
