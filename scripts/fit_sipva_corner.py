#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the SIPVA (global db/dt) fit for a KOI, persist its post-burn-in posterior, then make the
ApJ corner plot.

This does the expensive part once: it runs the full catalog-driven pipeline
(``execute_TDV_func(..., save_posterior=True)``), which downloads the light curve, runs the
per-transit and global fits, and saves the post-burn-in global-fit samples (model units) to
``sipva_posterior_samples_koi_<koi>.npz`` + a metadata sidecar under ``../data/Output_data/koi-<koi>/``.
It then calls the (cheap, re-runnable) plotting half to render the figure and the display-unit
summary table. Re-style later with ``plot_sipva_corner.py`` alone -- no refit needed.

Run from the ``scripts/`` directory:

    ../.venv/bin/python fit_sipva_corner.py 377.02
"""

import os
import sys
import pathlib
import argparse

# Make src/core importable (same shim as run_tdv.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "core"))

from pipeline import execute_TDV_func
import plot_sipva_corner


def main():
    ap = argparse.ArgumentParser(description="Fit SIPVA posterior for a KOI and make its corner plot.")
    ap.add_argument("koi", nargs="?", default="377.02", help="KOI number (default: 377.02).")
    ap.add_argument("--detrend", choices=["gp", "savgol"], default=None,
                    help="Detrending method (default: TDV_DETREND env, else 'gp').")
    ap.add_argument("--fit-only", action="store_true",
                    help="Run the fit and persist the posterior, but skip the corner plot "
                         "(the plotter's period baseline is specialized to KOI 377.02; use "
                         "this for other KOIs, e.g. the KOI 103.01 validation/smoke runs).")
    args = ap.parse_args()

    out_root = os.path.abspath(os.environ.get("TDV_OUTPUT_ROOT",
                                              os.path.join("..", "data", "Output_data")))
    print(f"[fit_sipva_corner] resolved output root: {out_root}")

    ok, err = execute_TDV_func(args.koi, detrend_method=args.detrend, save_posterior=True)
    if not ok:
        sys.exit(f"SIPVA fit failed for KOI {args.koi}: {err}")

    if not args.fit_only:
        plot_sipva_corner.make_corner(args.koi)


if __name__ == "__main__":
    main()
