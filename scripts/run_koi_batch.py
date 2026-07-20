#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the live TDV pipeline on a list of REAL Kepler KOIs in parallel.

This is the real-KOI analogue of ``multiprocess.py`` (which runs the synthetic SNR grid). Each KOI
goes through the full catalog-driven pipeline via ``execute_TDV_func``: download the light curve
(MAST), per-transit fits, then the global db/dt fit -- using catalog priors plus the PyLDTk
limb-darkening Normal prior. Parallelism is two-level:

  * OUTER pool  (KOI_OUTER_WORKERS): several whole KOIs run concurrently.
  * INNER pool  (TDV_N_WORKERS):     each KOI's own per-transit fits, inside pipeline._n_workers.

Choose OUTER * INNER <= cores and keep total processes under the 300-proc ulimit. On an spr node
(112 cores) the defaults give 8 x 13 = 104 cores, peak ~113 processes.

The PyLDTk limb-darkening cache is PRE-WARMED serially before the pool so concurrent workers never
race the shared ~/.ldtk download. Per-KOI failures are isolated (one bad fit cannot abort the job)
and an aggregated recovery summary is written at the end.

Run from scripts/ (the pipeline's per-system output helpers write to ../data/Output_data):
    cd scripts && ../.venv/bin/python run_koi_batch.py
"""
import os
import sys
import json
import zlib
import argparse
import pathlib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp
import numpy as np
import pandas as pd

# Pin native threads BEFORE numpy/numba/pytransit import (mirrors pipeline.py): the nested worker
# pools must stay single-threaded or numba's default OpenMP layer would spawn a CPU-sized thread
# pool per worker and blow RLIMIT_NPROC=300.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: never touch a display

# Make src/core importable and anchor the summary path to the repo root (NOT the cwd), so the
# aggregate reads/writes are correct regardless of where the job launches from. (The pipeline's own
# per-KOI output helpers are '..'-relative and rely on cwd=scripts/.)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "core"))
# Per-KOI outputs land under TDV_OUTPUT_ROOT (default ../data/Output_data). The summary/failure
# writes and the per-KOI JSON reads below must point at the SAME root the pipeline's save helpers
# use; both resolve TDV_OUTPUT_ROOT relative to cwd=scripts/, so they stay in lockstep.
_env_out_root = os.environ.get("TDV_OUTPUT_ROOT")
_OUTPUT_DIR = pathlib.Path(_env_out_root).resolve() if _env_out_root else _REPO_ROOT / "data" / "Output_data"

from pipeline import execute_TDV_func
from limb_darkening import koi_ld_prior


# The KOIs to fit (override with KOI_LIST="103.01,137.02,..."). Kept as strings: get_koi accepts
# either, and the per-KOI output folder is koi-<this string>.
_KOIS = ["103.01", "137.02", "139.01", "142.01", "209.02", "377.01", "377.02", "460.01",
         "806.01", "841.02", "872.01", "1320.01", "1423.01", "1856.01", "2698.01", "2770.01"]

# Summary metric keys read back from each KOI's tdv_metrics JSON (written by TDV_fit). The
# segment-contamination fix (sibling masking / coverage / baseline / prior-dominated audit) adds the
# pre-fit data-cut counters; missing keys fall back to NaN in _aggregate, so older JSONs still load.
_METRIC_KEYS = ['num_transit', 'n_transit_excluded', 'num_transit_used_global', 'rho_reject_factor',
                'db_dt_global', 'db_dt_global_err', 'db_dt_global_zscore',
                'db_dt_linreg', 'db_dt_linreg_err', 't_score_b', 't_score_t14',
                'n_cadences_sibling_masked', 'epochs_sibling_affected', 'n_transit_no_coverage',
                'n_transit_bad_baseline', 'n_transit_too_few_points', 'n_sc_to_lc_fallback',
                'n_prior_dominated']


def _koi_list():
    env = os.environ.get("KOI_LIST")
    if env:
        return [k.strip() for k in env.split(",") if k.strip()]
    return list(_KOIS)


def _prewarm_ld_cache(kois):
    """Compute & cache the PyLDTk limb-darkening prior for each star serially, before forking, so
    concurrent workers hit the cache instead of racing the shared ~/.ldtk download. Never fatal: a
    failure here just means that KOI falls back to the uniform q1/q2 prior in the fit."""
    print(f"[prewarm] limb-darkening cache for {len(kois)} KOIs ...", flush=True)
    for koi in kois:
        try:
            q = koi_ld_prior(koi)
            print(f"[prewarm] {koi}: {'NP q1/q2 ' + str(tuple(round(x, 4) for x in q)) if q else 'uniform (no LD prior)'}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prewarm] {koi}: FAILED ({e!r}); will use uniform q1/q2 prior", flush=True)


def _run_one_koi(koi):
    """Run the full TDV pipeline for one real KOI. Top-level (picklable) for the outer pool. Never
    raises: returns a status record so one failed KOI cannot abort the whole job. execute_TDV_func
    already writes this KOI's CSVs/JSON/figures to data/Output_data/koi-<koi>/."""
    try:
        # Optional reseed of the OUTER worker's RNG (governs the global db/dt fit's DE seeding and
        # emcee walker init -- TDV_SEED_BASE only reseeds the separate per-transit worker processes).
        # Opt-in via KOI_GLOBAL_SEED so default runs are unchanged; used by the reseed re-run to
        # nudge a fit off a pathological walker draw (e.g. a PyTransit divide-by-zero).
        gseed = os.environ.get("KOI_GLOBAL_SEED")
        if gseed:
            # crc32 (not hash()) for a deterministic, restart-stable per-KOI offset.
            np.random.seed((int(gseed) + zlib.crc32(str(koi).encode())) % (2 ** 32))
        ok, err = execute_TDV_func(koi)
        return {'koi': koi, 'status': 'ok' if ok else 'error', 'error': err or ''}
    except Exception as e:  # noqa: BLE001 -- catch a hard crash inside the call too
        return {'koi': koi, 'status': 'error', 'error': f"{e!r} | {traceback.format_exc()}"}


def _aggregate(kois, status_by_koi, error_by_koi, detrend_method):
    """One summary row per planned KOI. For 'ok' KOIs read the per-KOI tdv_metrics JSON; otherwise
    emit NaN metrics so the table always has one row per KOI and never crashes on a missing file.
    detrend_method is the run-level provenance stamped on every row."""
    rows, errors = [], []
    for koi in kois:
        status = status_by_koi.get(koi, 'missing')
        out = {'koi': koi, 'detrend_method': detrend_method, 'status': status}
        for k in _METRIC_KEYS:
            out[k] = np.nan

        json_path = _OUTPUT_DIR / f"koi-{koi}" / f"tdv_metrics_koi_{koi}.json"
        if status == 'ok' and json_path.exists():
            with open(json_path) as fh:
                m = json.load(fh)
            for k in _METRIC_KEYS:
                out[k] = m.get(k, np.nan)
        elif status == 'ok':
            out['status'] = 'ok_no_json'  # worker said ok but no JSON -> flag the anomaly
        rows.append(out)
        errors.append(error_by_koi.get(koi, ''))

    summary = pd.DataFrame(rows, columns=['koi', 'detrend_method'] + _METRIC_KEYS + ['status'])
    summary_path = _OUTPUT_DIR / "tdv_batch_summary.csv"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    fail_mask = summary['status'] != 'ok'
    failures = summary[fail_mask].copy()
    failures['error'] = [e for e, keep in zip(errors, fail_mask) if keep]
    fail_path = _OUTPUT_DIR / "tdv_batch_failures.csv"
    return summary, summary_path, failures, fail_path


def main():
    ap = argparse.ArgumentParser(description="Run the TDV pipeline on a list of real Kepler KOIs.")
    ap.add_argument("--detrend", choices=["gp", "savgol"], default=None,
                    help="Detrending method (default: TDV_DETREND env, else 'gp').")
    args = ap.parse_args()

    # Resolve once and export, so every forked/submitted worker's execute_TDV_func sees the same
    # method (same mechanism as TDV_N_WORKERS below). CLI flag wins over the env, else default gp.
    detrend_method = args.detrend or os.environ.get("TDV_DETREND", "gp")
    os.environ["TDV_DETREND"] = detrend_method

    kois = _koi_list()

    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1

    # Outer concurrency (whole KOIs at once) and inner per-transit workers. Defaults aim for an spr
    # node: 8 x 13 = 104 cores, peak ~113 procs. Both overridable from the SLURM script.
    outer = int(os.environ.get("KOI_OUTER_WORKERS", str(min(len(kois), 8))))
    outer = max(1, min(outer, len(kois)))
    inner = int(os.environ.get("TDV_N_WORKERS", str(max(1, avail // outer))))
    os.environ["TDV_N_WORKERS"] = str(inner)   # inherited by forked workers -> pipeline._n_workers
    os.environ.setdefault("TDV_MAKE_PLOTS", "1")  # real-KOI runs get figures by default

    print(f"[info] {len(kois)} KOIs | detrend={detrend_method} | avail_cores={avail} "
          f"outer={outer} inner={inner} (<= {outer * inner} cores, ~{1 + outer + outer * inner} procs) "
          f"plots={os.environ['TDV_MAKE_PLOTS']}", flush=True)

    # Pre-warm the LD cache serially in the parent (before any fork).
    _prewarm_ld_cache(kois)

    status_by_koi, error_by_koi = {}, {}
    if outer == 1:
        for i, koi in enumerate(kois, 1):
            rec = _run_one_koi(koi)
            status_by_koi[koi] = rec['status']
            if rec['status'] != 'ok':
                error_by_koi[koi] = rec['error']
            print(f"[{i}/{len(kois)}] koi {koi}: {rec['status']}", flush=True)
    else:
        ctx = mp.get_context('fork')  # fork inherits compiled numba + the warmed caches
        with ProcessPoolExecutor(max_workers=outer, mp_context=ctx) as ex:
            fut_to_koi = {ex.submit(_run_one_koi, koi): koi for koi in kois}
            done = 0
            for fut in as_completed(fut_to_koi):
                done += 1
                koi = fut_to_koi[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # worker died (OOM/segfault) -> record, keep going
                    rec = {'koi': koi, 'status': 'worker_died',
                           'error': f"{e!r} | {traceback.format_exc()}"}
                status_by_koi[koi] = rec['status']
                if rec['status'] != 'ok':
                    error_by_koi[koi] = rec['error']
                print(f"[{done}/{len(kois)}] koi {koi}: {rec['status']}", flush=True)

    summary, summary_path, failures, fail_path = _aggregate(kois, status_by_koi, error_by_koi,
                                                            detrend_method)
    if not failures.empty:
        failures.to_csv(fail_path, index=False)

    n_ok = int((summary['status'] == 'ok').sum())
    print(f"[done] {n_ok}/{len(kois)} ok. Summary -> {summary_path}", flush=True)
    if len(summary) - n_ok:
        print(f"[done] {len(summary) - n_ok} non-ok KOIs -> {fail_path}", flush=True)


if __name__ == "__main__":
    main()
