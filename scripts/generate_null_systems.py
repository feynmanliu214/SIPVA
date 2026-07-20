#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a matched NULL set (db/dt = 0) for the TDV false-positive test.

The SNR grid never injects a zero drift (DB_OVER_DT is solved into the band [0.01, 0.03]), so it
cannot measure how often the pipeline reports a spurious detection when no signal is present. This
builds that null: it takes the SAME first-50 systems we ran at SNR 10 (identical period, b, p,
rho, num_transits), sets db/dt = 0, and regenerates each light curve with the IDENTICAL white-noise
convention used for the signal set -- noise-free Kepler-LD model (``_model_fluxes``) plus seeded
200 ppm white noise (SIGMA). Only the drift is removed, so any |z| > 3 in the recovery is a genuine
false positive.

Output (drop-in compatible with multiprocess.py's loader):
    data/SNR_data/lightcurves/SNR_0/syn_snr0_<NNN>.npz   (keys: times, fluxes)
    data/SNR_data/SNR_0.csv                              (DB_OVER_DT=0, achieved_snr=0)

Run under the repo venv (needs pytransit):
    .venv/bin/python scripts/generate_null_systems.py
"""
import csv
import sys
import pathlib

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the signal generator's exact model + save convention so null curves are statistically
# identical to the SNR set except for db/dt = 0.
from generate_synthetic_systems import (_model_fluxes, save_lightcurve, CSV_COLUMNS,
                                         SIGMA, calculate_stellar_density)

_SNR_DATA_DIR = _REPO_ROOT / "data" / "SNR_data"
_PARENT_CSV = _SNR_DATA_DIR / "SNR_10.csv"   # matched parent population
_N_SYSTEMS = 50                              # first 50 rows -> match the SNR-10 run subset
_NULL_NOISE_SEED_BASE = 20260607             # fixed -> reproducible null noise realizations


def main():
    parent = pd.read_csv(_PARENT_CSV).head(_N_SYSTEMS)
    lc_dir = _SNR_DATA_DIR / "lightcurves" / "SNR_0"
    lc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, p in parent.iterrows():
        period = float(p["PERIOD"]); b = float(p["b"]); ror = float(p["p"])
        rs_over_a = float(p["RS_OVER_A"]); num_transit = int(p["NUM_TRANSITS"])
        rho = calculate_stellar_density(rs_over_a, period)
        name = f"syn_snr0_{idx:03d}"

        # db/dt = 0 -> constant b across all transits (the true null).
        times, model_fluxes = _model_fluxes(period, num_transit, b, ror, rho, 0.0)
        noise_rng = np.random.default_rng(_NULL_NOISE_SEED_BASE + idx)
        noisy = [f + noise_rng.normal(0.0, SIGMA, size=f.size) for f in model_fluxes]
        save_lightcurve(lc_dir, name, times, noisy)

        rows.append([b, ror, period, 0.0, rs_over_a, num_transit,
                     float(p["rho_star"]), 0.0, name])
        print(f"[{idx + 1}/{len(parent)}] {name}  N={num_transit}", flush=True)

    csv_path = _SNR_DATA_DIR / "SNR_0.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    print(f"[done] {len(rows)} null systems -> {csv_path} and {lc_dir}", flush=True)


if __name__ == "__main__":
    main()
