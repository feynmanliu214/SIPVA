#!/bin/bash
# Run-to-run z-score stability study for ONE real KOI. Runs up to N_MAX SIPVA (global db/dt) fits
# SEQUENTIALLY on the same real light curve, each with a distinct RNG seed (TDV_SEED_BASE), and STOPS
# at the first replication whose global-fit z-score has |z| > THRESH. Every trial's z-score is
# appended to a master-log CSV immediately (checkpoint), so a killed job keeps all completed trials.
# Goal: the run-to-run distribution of the SIPVA z-score (mean/std) + the trial count to a >3sigma
# crossing. z-score only -> per-system plots off (TDV_MAKE_PLOTS=0), no posterior/corner saved.
#
# The KOI is the first script argument. Submit ONE job per KOI (they run in parallel; each is
# sequential internally), overriding the job name / log files at submit time:
#     cd scripts
#     sbatch -J zscore_460  -o zscore_460.%j.out  -e zscore_460.%j.err  fit_zscore_study.slurm 460.01
#     sbatch -J zscore_1320 -o zscore_1320.%j.out -e zscore_1320.%j.err fit_zscore_study.slurm 1320.01
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1                 # one driver process; each fit forks its own per-transit pool
#SBATCH -t 06:00:00          # realistic: 15 fits x ~25 min ~= 6 h; short wall backfills sooner. The
                             #   loop is resume-safe (checkpointed log), so if it hits the wall just
                             #   resubmit the SAME command and it continues after the logged trials.
#SBATCH -A TG-ATM170020
#SBATCH -o zscore_study.%j.out
#SBATCH -e zscore_study.%j.err

set -uo pipefail             # NOT -e: a failed/timed-out trial must not abort the campaign

KOI="${1:?usage: sbatch [-J .. -o .. -e ..] fit_zscore_study.slurm <koi>}"
N_MAX=15                     # cap on replications per KOI (stop earlier on the first |z|>THRESH)
THRESH=3
PER_TRIAL_TIMEOUT=5400       # 90 min/trial safety kill (normal ~40 min) so one hung fit can't eat the wall

export MPLBACKEND=Agg                                   # headless: never touch a display
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export TDV_N_WORKERS=48                                  # per-transit pool: 2 waves for 73/84 transits on a 112-core node
export TDV_MAKE_PLOTS=0                                  # z-score only: skip folded/regression figures
export TDV_DETREND=gp                                   # GP (transit-masked Matern-3/2) detrending

cd "$SLURM_SUBMIT_DIR"       # must be the repo's scripts/ dir -- outputs are '..'-relative

KOI_INT=${KOI%.*}                                        # 460.01 -> 460 ; used to build a traceable seed base
STUDY_ROOT=../data/zscore_study
LOG="$STUDY_ROOT/zscore_log_koi-$KOI.csv"
mkdir -p "$STUDY_ROOT"

echo "host=$(hostname) cwd=$PWD KOI=$KOI N_MAX=$N_MAX thresh=$THRESH inner=$TDV_N_WORKERS detrend=$TDV_DETREND log=$LOG"

# Resume-safe: if a prior job for this KOI already logged trials, continue AFTER them (so seeds,
# which are KOI_INT*1000+trial, stay unique across resumes); and if any prior trial already crossed
# |z|>THRESH there is nothing left to do. Lets us use a short wall without wasting completed trials.
done_trials=0
if [ -s "$LOG" ]; then
    done_trials=$(( $(wc -l < "$LOG") - 1 ))            # data rows = lines - header
    [ "$done_trials" -lt 0 ] && done_trials=0
    if awk -F, 'NR>1 && $10==1 {f=1} END{exit !f}' "$LOG"; then
        echo "RESULT KOI=$KOI already has a |z|>$THRESH crossing in $LOG; nothing to do"
        exit 0
    fi
fi
start=$(( done_trials + 1 ))
if [ "$start" -gt "$N_MAX" ]; then
    echo "RESULT KOI=$KOI already ran $done_trials trials (>= N_MAX=$N_MAX), no crossing; nothing to do"
    exit 0
fi
[ "$done_trials" -gt 0 ] && echo "resuming at trial $start ($done_trials already logged in $LOG)"

crossed_at=0
for trial in $(seq "$start" "$N_MAX"); do
    export TDV_SEED_BASE=$(( KOI_INT * 1000 + trial ))  # distinct per (KOI, trial): 460001, 460002, ...
    export TDV_OUTPUT_ROOT="$STUDY_ROOT/koi-$KOI/trial_$(printf '%03d' "$trial")"
    echo "=== $(date -u +%FT%TZ) KOI $KOI trial $trial/$N_MAX seed_base=$TDV_SEED_BASE out=$TDV_OUTPUT_ROOT ==="
    timeout "$PER_TRIAL_TIMEOUT" ../.venv/bin/python run_zscore_trial.py "$KOI" "$trial" "$LOG"
    code=$?
    if [ "$code" -eq 42 ]; then
        echo "=== KOI $KOI: |z|>$THRESH CROSSING at trial $trial ==="
        crossed_at=$trial
        break
    elif [ "$code" -eq 3 ]; then
        echo "=== KOI $KOI trial $trial: fit FAILED, continuing ==="
    elif [ "$code" -ne 0 ]; then
        echo "=== KOI $KOI trial $trial: unexpected exit $code (timeout=124?), continuing ===" >&2
    fi
done

if [ "$crossed_at" -gt 0 ]; then
    echo "RESULT KOI=$KOI first_crossing_trial=$crossed_at (of max $N_MAX)"
else
    echo "RESULT KOI=$KOI no_crossing within $N_MAX trials"
fi
echo "=== $(date -u +%FT%TZ) done; full per-trial z-score log: $LOG ==="
