#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the TDV pipeline on the synthetic white-noise SNR grid (50 systems x 5 SNR = 250 fits).

For each injected system this loads the canonical pre-generated **white-noise** light curve
(``data/SNR_data/lightcurves/SNR_<level>/<name>.npz``, whose seeded 200 ppm noise matches the
``achieved_snr`` in the SNR CSVs), deliberately offsets the true inputs to mimic literature-based
priors, and runs the system through the two-stage TDV fit using those offset priors (NOT catalog
priors). Systems are independent; the per-system RNG seed makes the prior offsets reproducible
(the curve noise is already fixed at generation time).

Designed for one Stampede3 ``spr`` node (112 cores, 128 GB): a flat process pool runs whole
systems in parallel, each system's internal per-transit fit forced serial (TDV_N_WORKERS=1), and
per-system figures suppressed (TDV_MAKE_PLOTS=0). Per-task failures are isolated so one bad fit
cannot abort the job, and a recovery summary is written at the end.

Run from ``scripts/`` (the pipeline's save helpers write to ``../data/Output_data``):
    cd scripts && ../.venv/bin/python multiprocess.py
"""
import os
import sys
import json
import glob
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Make src/core importable and anchor data paths to the repo root (NOT the cwd) so input reads and
# the summary/failure writes are correct regardless of where the job launches from. (The pipeline's
# own per-system output helpers are '..'-relative and rely on cwd=scripts/.)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "core"))
_SNR_DATA_DIR = _REPO_ROOT / "data" / "SNR_data"
# Light-curve set: "white" (default) reads lightcurves/; "realistic" reads lightcurves_realistic/
# (white + CDPP-calibrated red noise). Selected by SNR_LC_SET so the same driver runs either set.
_LC_SET = os.environ.get("SNR_LC_SET", "white")
_LC_DIR = _SNR_DATA_DIR / ("lightcurves_realistic" if _LC_SET == "realistic" else "lightcurves")
# Honor TDV_OUTPUT_ROOT (default ../data/Output_data) for consistency with the pipeline's per-KOI
# save helpers; resolved relative to cwd=scripts/ like those helpers.
_env_out_root = os.environ.get("TDV_OUTPUT_ROOT")
_OUTPUT_DIR = pathlib.Path(_env_out_root).resolve() if _env_out_root else _REPO_ROOT / "data" / "Output_data"
# Optional prefix on the per-system output folder name (e.g. "real_") so a realistic run's
# koi-<name> folders never collide with the white run's. Empty -> unchanged white behavior.
_NAME_PREFIX = os.environ.get("SNR_NAME_PREFIX", "")

from model import calculate_stellar_density
from priors import synthetic_prior_spec
from pipeline import TDV_fit


# The run grid: SNR levels (fixed order) x first 50 systems per level. The fixed ordering makes the
# per-task seed (seed_base + task_index) deterministic and restart-stable.
_SNR_LEVELS = [10, 20, 30, 50, 100]
_ROWS_PER_LEVEL = range(0, 50)

# Per-point photometric error. Matches SIGMA = 1e-4*2 of the saved curves. NOTE: cosmetic only --
# ferr_out never enters a likelihood (the first-stage per-transit fits estimate their own white
# noise; the global stage resets ferr to 2e-4 internally). Kept honest for clarity.
_FERR = 2e-4

# Offset magnitudes for the literature-mimicking priors. The prior is centered on the true
# simulation input plus a Gaussian draw of this scale; the prior *width* is the fixed synthetic
# width from synthetic_prior_spec.
_OFFSET_PERIOD_FRAC = 3e-5   # period: sigma = frac * period
_OFFSET_P_FRAC      = 0.03   # radius ratio p: sigma = frac * p
_OFFSET_B_ABS       = 0.05   # impact parameter b: absolute sigma
_OFFSET_RHO_FRAC    = 0.10   # stellar density rho: sigma = frac * rho


def _read_system(row_number, csv_filename):
    """Read one injected system's parameters from a SNR CSV row."""
    df = pd.read_csv(_SNR_DATA_DIR / csv_filename)
    if not 0 <= row_number < len(df):
        raise IndexError(f"row {row_number} out of range for {csv_filename} ({len(df)} rows)")
    params = df.iloc[row_number]
    return {
        'period':       float(params['PERIOD']),
        'b':            float(params['b']),
        'p':            float(params['p']),
        'RS_OVER_A':    float(params['RS_OVER_A']),
        'db_over_dt':   float(params['DB_OVER_DT']),
        'num_transits': int(params['NUM_TRANSITS']),
        'name':         str(params['name']),
    }


def _load_curves(level, name):
    """Load the saved light curve for one system as lists of float arrays.

    The .npz stores ``times``/``fluxes`` as object arrays of shape (num_transits, 300); convert each
    to a contiguous float array so the downstream filters can reassign shortened/masked arrays back
    in place. For the white set, assert the key set to refuse a realistic-noise file (a guard against
    loading the wrong set). The realistic set carries extra keys (sigma_r, kept_transits, ...) which
    are ignored here -- only times/fluxes are consumed (gap_prob=0 at generation, so every transit is
    present and the per-transit structure matches the white SNR_<level>.csv rows)."""
    path = _LC_DIR / f"SNR_{level}" / f"{name}.npz"
    with np.load(path, allow_pickle=True) as z:
        if _LC_SET == "white":
            assert set(z.files) == {"times", "fluxes"}, \
                f"{path} is not a white-noise curve (keys={set(z.files)})"
        times = [np.asarray(row, dtype=float) for row in z["times"]]
        fluxes = [np.asarray(row, dtype=float) for row in z["fluxes"]]
    return times, fluxes


def _run_one_system(task):
    """Load, offset-prior, and TDV-fit a single synthetic system. Top-level (picklable) for the
    ProcessPoolExecutor. ``task`` is (level, row_number, csv_filename, seed). Never raises: returns
    a status record so one failed fit cannot abort the whole job. The fit's metrics are written to
    data/Output_data/koi-<name>/ by TDV_fit; this returns only {name, snr_level, status, error}."""
    level, row_number, csv_filename, seed = task
    out_name = None
    try:
        np.random.seed(seed)  # reproducible: governs the prior offsets (curve noise is pre-baked)

        s = _read_system(row_number, csv_filename)
        curve_name = s['name']                  # npz filename + truth-CSV key (un-prefixed)
        out_name = _NAME_PREFIX + curve_name    # per-system output folder name (koi-<out_name>)

        times, fluxes = _load_curves(level, curve_name)
        ferr_out = [np.full(len(t), _FERR) for t in times]

        # Deliberately offset the true inputs to mimic literature-based prior centers.
        rho = calculate_stellar_density(s['RS_OVER_A'], s['period'])
        period_off = s['period'] + np.random.normal(0, _OFFSET_PERIOD_FRAC * s['period'])
        p_off      = s['p']      + np.random.normal(0, _OFFSET_P_FRAC * s['p'])
        b_off      = s['b']      + np.random.normal(0, _OFFSET_B_ABS)
        rho_off    = rho         + np.random.normal(0, _OFFSET_RHO_FRAC * rho)

        prior_spec = synthetic_prior_spec(period_off, b_off, rho_off, p_off)

        # Transit i is centered at t = period*i, so the transit-number basis is (t0=0, period).
        TDV_fit(times, fluxes, out_name, ferr_out,
                prior_spec=prior_spec, ephemeris=(0.0, s['period']))
        return {'name': out_name, 'snr_level': level, 'row': row_number, 'status': 'ok', 'error': ''}
    except Exception as e:  # noqa: BLE001 -- per-task isolation is the whole point
        import traceback
        return {'name': out_name if out_name else f"{_NAME_PREFIX}SNR_{level}_row{row_number}",
                'snr_level': level, 'row': row_number, 'status': 'error',
                'error': f"{e!r} | {traceback.format_exc()}"}


def _build_tasks(seed_base, levels):
    """Tasks in fixed (level, row) order with a deterministic per-task seed.

    ``levels`` defaults to the five-SNR grid; override via SNR_LEVELS (e.g. "0" for the null set)."""
    tasks = []
    for level in levels:
        csv_filename = f"SNR_{level}.csv"
        for row in _ROWS_PER_LEVEL:
            tasks.append((level, row, csv_filename, seed_base + len(tasks)))
    return tasks


def _aggregate(tasks, status_by_id, failed_errors):
    """Status-aware aggregation over the planned systems. For 'ok' systems read the per-system
    tdv_metrics JSON; for 'error' (or an unexpectedly missing JSON) emit a row with NaN metrics so
    the summary always has one row per planned system and never crashes on a missing file.

    Status and captured tracebacks are keyed by task identity ``(level, row_number)`` -- NOT the
    display name -- so they resolve correctly even when SNR_NAME_PREFIX rewrites the output-folder
    name (the success/error/worker-death paths all key by the same tuple). Truth is read from the
    CSV by the un-prefixed ``name``; the per-system JSON path uses the prefixed ``out_name``."""
    metric_keys = ['db_dt_global', 'db_dt_global_err', 'db_dt_global_zscore',
                   'db_dt_linreg', 't_score_b']
    rows = []
    errors = []
    csv_cache = {}
    for level, row_number, csv_filename, _seed in tasks:
        if csv_filename not in csv_cache:
            csv_cache[csv_filename] = pd.read_csv(_SNR_DATA_DIR / csv_filename)
        sys_row = csv_cache[csv_filename].iloc[row_number]
        name = str(sys_row['name'])
        out_name = _NAME_PREFIX + name
        status = status_by_id.get((level, row_number), 'missing')

        out = {'name': out_name, 'snr_level': level,
               'true_db_dt': float(sys_row['DB_OVER_DT']),
               'achieved_snr': float(sys_row['achieved_snr']),
               'status': status, 'num_transit_loaded': np.nan}
        for k in metric_keys:
            out[k] = np.nan

        json_path = _OUTPUT_DIR / f"koi-{out_name}" / f"tdv_metrics_koi_{out_name}.json"
        if status == 'ok' and json_path.exists():
            with open(json_path) as fh:
                m = json.load(fh)
            out['num_transit_loaded'] = m.get('num_transit', np.nan)
            for k in metric_keys:
                out[k] = m.get(k, np.nan)
        elif status == 'ok':
            # Worker reported ok but no JSON -- treat as anomaly, keep NaNs, flag it.
            out['status'] = 'ok_no_json'
        rows.append(out)
        errors.append(failed_errors.get((level, row_number), ''))

    summary = pd.DataFrame(rows, columns=['name', 'snr_level', 'true_db_dt', 'achieved_snr',
                                          'db_dt_global', 'db_dt_global_err', 'db_dt_global_zscore',
                                          'db_dt_linreg', 't_score_b', 'num_transit_loaded',
                                          'status'])
    # Tag isolates concurrent runs (e.g. the null set) so they don't clobber each other's summary.
    tag = os.environ.get('SNR_SUMMARY_TAG', '')
    summary_path = _SNR_DATA_DIR / f"tdv_recovery_summary{tag}.csv"
    summary.to_csv(summary_path, index=False)

    # Failures carry their captured traceback (keyed by task identity, attached in task order).
    fail_mask = summary['status'] != 'ok'
    failures = summary[fail_mask].copy()
    failures['error'] = [e for e, keep in zip(errors, fail_mask) if keep]
    fail_path = _SNR_DATA_DIR / f"tdv_failures{tag}.csv"
    return summary, summary_path, failures, fail_path


def main():
    seed_base = int(os.environ.get('SNR_SEED_BASE', '12345'))
    levels_env = os.environ.get('SNR_LEVELS')
    levels = [int(x) for x in levels_env.split(',')] if levels_env else _SNR_LEVELS
    tasks = _build_tasks(seed_base, levels)

    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    default_workers = min(avail, 110)
    n_workers = int(os.environ.get('SNR_N_WORKERS', str(default_workers)))
    n_workers = max(1, min(n_workers, len(tasks)))

    # Warn (don't abort) if a prior run's per-system dirs are present: a full rerun overwrites them
    # deterministically, but a partial rerun would leave stale systems behind.
    existing = glob.glob(str(_OUTPUT_DIR / f"koi-{_NAME_PREFIX}syn_snr*"))
    if existing:
        print(f"[warn] {len(existing)} existing koi-{_NAME_PREFIX}syn_snr* output dirs found in "
              f"{_OUTPUT_DIR}; they will be overwritten where re-run. Delete for a clean slate.",
              flush=True)

    # Each system runs its per-transit fits serially; figures off.
    os.environ['TDV_N_WORKERS'] = '1'
    os.environ['TDV_MAKE_PLOTS'] = '0'

    print(f"[info] {len(tasks)} systems, {n_workers} workers, seed_base={seed_base}", flush=True)

    status_records = {}
    failed_errors = {}
    if n_workers == 1:
        for i, task in enumerate(tasks, 1):
            rec = _run_one_system(task)
            status_records[(rec['snr_level'], rec['row'])] = rec['status']
            if rec['status'] != 'ok':
                failed_errors[(rec['snr_level'], rec['row'])] = rec['error']
            print(f"[{i}/{len(tasks)}] {rec['name']}: {rec['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_task = {ex.submit(_run_one_system, t): t for t in tasks}
            done = 0
            for fut in as_completed(future_to_task):
                done += 1
                try:
                    rec = fut.result()
                except Exception as e:  # worker died (e.g. OOM/segfault) -> record, keep going
                    import traceback
                    level, row_number, csv_filename, _ = future_to_task[fut]
                    rec = {'name': f"{_NAME_PREFIX}SNR_{level}_row{row_number}", 'snr_level': level,
                           'row': row_number, 'status': 'worker_died',
                           'error': f"{e!r} | {traceback.format_exc()}"}
                status_records[(rec['snr_level'], rec['row'])] = rec['status']
                if rec['status'] != 'ok':
                    failed_errors[(rec['snr_level'], rec['row'])] = rec['error']
                print(f"[{done}/{len(tasks)}] {rec['name']}: {rec['status']}", flush=True)

    summary, summary_path, failures, fail_path = _aggregate(tasks, status_records, failed_errors)
    # Write failures with their captured tracebacks (the summary table itself carries no error text).
    if not failures.empty:
        failures.to_csv(fail_path, index=False)

    n_ok = int((summary['status'] == 'ok').sum())
    print(f"[done] {n_ok}/{len(tasks)} ok. Summary -> {summary_path}", flush=True)
    if len(summary) - n_ok:
        print(f"[done] {len(summary) - n_ok} non-ok systems -> {fail_path}", flush=True)


if __name__ == "__main__":
    main()
