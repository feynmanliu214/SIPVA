#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the realistic (Kepler-noise) NULL set (db/dt = 0) for the TDV false-positive test.

This is the realistic-noise counterpart of ``generate_null_systems.py``. It takes the SAME 50 null
systems (the existing ``SNR_0.csv``, i.e. the first 50 SNR_10 parents with db/dt forced to 0),
regenerates each light curve with the IDENTICAL white + CDPP-calibrated red noise convention used
by the realistic signal curves (``save_realistic_curve`` / ``calibrate_cdpp`` in
``generate_synthetic_systems.py``), and writes them under ``lightcurves_realistic/SNR_0/``. Only the
noise model differs from the white null; the systems and db/dt=0 are unchanged, so any |z|>3 here is
a genuine false positive under realistic noise.

Output (drop-in for multiprocess.py with SNR_LC_SET=realistic, SNR_LEVELS includes 0):
    data/SNR_data/lightcurves_realistic/SNR_0/syn_snr0_<NNN>.npz   (white+red, gap_prob=0)
    -> target-0 rows merged into realistic_manifest.csv / realistic_summary.csv
Truth reuses the existing data/SNR_data/SNR_0.csv (db/dt=0); no new CSV is written.

Run under the repo venv (needs pytransit):
    .venv/bin/python scripts/generate_null_systems_realistic.py            # refuses to overwrite
    .venv/bin/python scripts/generate_null_systems_realistic.py --force    # replace existing curves
"""
import argparse
import sys
import time
import pathlib

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the realistic generator's exact noise machinery so null curves are statistically identical
# to the realistic signal set except for db/dt = 0.
from generate_synthetic_systems import (
    _model_fluxes, calculate_stellar_density, calibrate_cdpp, save_realistic_curve,
    write_realistic_manifest, write_realistic_summary,
    DEFAULT_TARGET_CDPP_PPM,
)

_SNR_DATA_DIR = _REPO_ROOT / "data" / "SNR_data"
_TRUTH_CSV = _SNR_DATA_DIR / "SNR_0.csv"      # existing white null: same 50 systems, db/dt=0
_N_SYSTEMS = 50
_NULL_SEED_BASE = 20260608                    # distinct from the white null (20260607) -> reproducible
_RED_TAU_HOURS = 2.0                          # matches the realistic signal curves
_GAP_PROB = 0.0                               # matches the realistic signal curves (no dropped transits)
_MIN_TRANSITS_KEPT = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing lightcurves_realistic/SNR_0 (unlink old .npz first).")
    args = ap.parse_args()

    parent = pd.read_csv(_TRUTH_CSV).head(_N_SYSTEMS)
    lc_dir = _SNR_DATA_DIR / "lightcurves_realistic" / "SNR_0"

    # Overwrite guard, mirroring the realistic generator (generate_synthetic_systems.py:400-407).
    if lc_dir.exists() and any(lc_dir.glob("*.npz")) and not args.force:
        raise SystemExit(f"Refusing to overwrite {lc_dir} (use --force).")
    lc_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for old in lc_dir.glob("*.npz"):
            old.unlink()

    # Calibrate the red-noise GP amplitude once (same anchor as the realistic signal curves).
    cdpp_cfg = calibrate_cdpp(DEFAULT_TARGET_CDPP_PPM, _RED_TAU_HOURS)
    print(f"CDPP calibration (tau={_RED_TAU_HOURS} h): target={cdpp_cfg['target_cdpp']:.1f}  "
          f"white={cdpp_cfg['cdpp_white_6p5']:.2f}  red={cdpp_cfg['cdpp_red_6p5']:.2f}  "
          f"model_total={cdpp_cfg['cdpp_total_6p5']:.2f} ppm  "
          f"-> sigma_r={cdpp_cfg['sigma_r']:.3e} ({1e6 * cdpp_cfg['sigma_r']:.1f} ppm/point)",
          flush=True)

    t0 = time.time()
    manifest_rows = []
    for idx, p in parent.iterrows():
        period = float(p["PERIOD"]); b = float(p["b"]); ror = float(p["p"])
        rs_over_a = float(p["RS_OVER_A"]); num_transit = int(p["NUM_TRANSITS"])
        rho = calculate_stellar_density(rs_over_a, period)
        name = str(p["name"])   # syn_snr0_<NNN>, row-aligned with SNR_0.csv

        # db/dt = 0 -> constant b across all transits (the true null).
        times, model_fluxes = _model_fluxes(period, num_transit, b, ror, rho, 0.0)
        rec = {"seed": _NULL_SEED_BASE + idx}
        # Positional signature: (lc_dir, name, rec, times, model_fluxes, target, row_index,
        #                        cdpp_cfg, gap_prob, min_transits_kept)
        mrow = save_realistic_curve(lc_dir, name, rec, times, model_fluxes, 0, idx,
                                    cdpp_cfg, _GAP_PROB, _MIN_TRANSITS_KEPT)
        manifest_rows.append(mrow)
        print(f"[{idx + 1}/{len(parent)}] {name}  N={num_transit}", flush=True)

    elapsed = time.time() - t0
    # Merge target-0 rows into the realistic manifest + summary (keeps the signal levels intact).
    write_realistic_manifest(manifest_rows, [0])
    summary = {"target": 0, "accepted": len(manifest_rows), "draws": len(manifest_rows),
               "acceptance_rate": 1.0, "achieved_mean": 0.0, "achieved_std": 0.0,
               "elapsed_s": elapsed}
    write_realistic_summary([summary], [0])
    print(f"[done] {len(manifest_rows)} realistic null systems -> {lc_dir}", flush=True)


if __name__ == "__main__":
    main()
