#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ONE z-score-study replication of the SIPVA (global db/dt) fit for a real KOI and log its
z-score. Part of the run-to-run z-score stability study (see fit_zscore_study.slurm).

A "trial" = one full catalog-driven pipeline fit (``execute_TDV_func``) on the SAME real light curve
with a distinct RNG seed (``TDV_SEED_BASE``, set by the caller). No posterior/corner is saved and the
per-system plots are off (``TDV_MAKE_PLOTS=0``); we only need the global-fit z-score
(``db_dt_global_zscore``) that the pipeline writes to ``tdv_metrics_koi_<koi>.json`` under
``TDV_OUTPUT_ROOT``.

The z-score is appended to a persistent master-log CSV (header written on first row) immediately, so
a killed/resumed job keeps every completed trial. Exit codes let the SLURM loop implement
"sequential until the first |z| > 3 crossing":

    42 -> |z| > THRESH   (crossing found; caller should STOP)
     0 -> fit ok, |z| <= THRESH (caller CONTINUES to the next trial)
     3 -> fit failed or metrics unreadable (caller CONTINUES; trial logged as 'failed')

Run from the scripts/ directory (paths are '..'-relative, same convention as the rest of the
pipeline). The caller sets TDV_SEED_BASE / TDV_OUTPUT_ROOT / TDV_DETREND in the environment:

    ../.venv/bin/python run_zscore_trial.py 460.01 1 ../data/zscore_study/zscore_log_koi-460.01.csv
"""

import os
import sys
import json
import csv
import argparse
import pathlib

# Make src/core importable (same shim as run_tdv.py / fit_sipva_corner.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "core"))

from pipeline import execute_TDV_func

THRESH = 3.0
FIELDS = ["koi", "trial", "seed_base", "detrend", "status",
          "z", "abs_z", "db_dt", "db_dt_err", "crossed"]


def main():
    ap = argparse.ArgumentParser(description="One z-score-study SIPVA replication for a KOI.")
    ap.add_argument("koi", help="KOI number, e.g. 460.01")
    ap.add_argument("trial", type=int, help="1-based trial index (bookkeeping only)")
    ap.add_argument("log_path", help="Master-log CSV to append this trial's z-score to")
    args = ap.parse_args()

    seed_base = int(os.environ.get("TDV_SEED_BASE", "-1"))
    out_root = os.environ.get("TDV_OUTPUT_ROOT", "")
    detrend = os.environ.get("TDV_DETREND", "gp")

    row = dict(koi=args.koi, trial=args.trial, seed_base=seed_base, detrend=detrend,
               status="", z="", abs_z="", db_dt="", db_dt_err="", crossed="")
    exit_code = 0

    # detrend_method left None -> execute_TDV_func resolves it from TDV_DETREND (gp here).
    ok, err = execute_TDV_func(args.koi, save_posterior=False)
    if not ok:
        row["status"] = "failed"
        exit_code = 3
        print(f"[trial {args.trial}] FIT FAILED: {err}")
    else:
        metrics_path = os.path.join(out_root, f"koi-{args.koi}", f"tdv_metrics_koi_{args.koi}.json")
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            z = float(m["db_dt_global_zscore"])
            crossed = abs(z) > THRESH
            row.update(status="ok", z=z, abs_z=abs(z),
                       db_dt=float(m["db_dt_global"]), db_dt_err=float(m["db_dt_global_err"]),
                       crossed=int(crossed))
            print(f"[trial {args.trial}] z={z:.4f} |z|={abs(z):.4f} crossed={crossed}")
            if crossed:
                exit_code = 42
        except (OSError, KeyError, ValueError) as e:
            row["status"] = "failed"
            exit_code = 3
            print(f"[trial {args.trial}] metrics unreadable at {metrics_path}: {e}")

    # Append to the master log (write the header only when the file is new/empty).
    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    new = (not os.path.exists(args.log_path)) or os.path.getsize(args.log_path) == 0
    with open(args.log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
