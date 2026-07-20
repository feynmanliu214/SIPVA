#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-transit boxcar SNR of the white-noise synthetic TDV systems.

For each target-LLR category (code label ``SNR`` = manuscript LLR_theory =
5/10/20/30/50) read the first 50 systems from ``data/SNR_data/SNR_<target>.csv``
and report the 16th/50th/84th percentiles of the *system-level median*
per-transit SNR.

Per transit j (j = 0 .. NUM_TRANSITS-1), matching the injected model exactly
(detectability.py:152, generate_synthetic_systems.py:124):

    t_c,j  = PERIOD * j              [days]   (t_ref = first transit = 0)
    b_j    = b0 + DB_OVER_DT * PERIOD * j / 365.0        (DB_OVER_DT is per YEAR)
    a_R    = 1 / RS_OVER_A           (CSV stores Rs/a; the T14 formula needs a/Rs)
    T_dur,j= (PERIOD / pi) * arcsin( sqrt( ((1+p)^2 - b_j^2) / (a_R^2 - b_j^2) ) )   [days]
    N_cad,j= T_dur,j / dt,   dt = 1/1440 day  (kept continuous)
    SNR_j  = (p^2 / sigma_phot) * sqrt(N_cad,j),   sigma_phot = 2e-4

CADENCE: dt = 1 min, matching how the white light curves were ACTUALLY sampled
(sigma_phot = 2e-4 is applied to every 1-min point; verified against the saved .npz).
The SNRs are large (medians ~23-230) for two real reasons, not an arithmetic error:
  (1) idealized-bright noise -- 200 ppm/min ~ a Kp~11 short-cadence target; and
  (2) the target-LLR rejection sampler couples LLR to planet radius (median p rises
      0.018 -> 0.060 across targets 5 -> 50, and SNR ∝ p^2).
Do NOT rescale to long cadence: dividing by sqrt(29.4) would report SNRs for data
that were never simulated (the curves are genuinely 1-min, sigma=2e-4).

System value = median_j(SNR_j); category stats = percentiles across the 50
system medians. Transits are NOT pooled across systems (equal weight per system).

Reads only the CSVs; changes no pipeline code. Run under any Python with numpy:
    .venv/bin/python scripts/compute_per_transit_snr.py
"""

import csv
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SNR_DIR = REPO_ROOT / "data" / "SNR_data"

TARGETS = [5, 10, 20, 30, 50]     # code SNR label == manuscript LLR_theory
SIGMA_PHOT = 2e-4                 # per normalized-flux white-noise sigma (200 ppm),
                                  # applied to every 1-min sample in the white light curves
DT_DAY = 1.0 / 1440.0            # 1-minute cadence in days (matches the simulated sampling)
YEAR_DAYS = 365.0               # DB_OVER_DT is per year; matches injected /365.0
N_SYSTEMS = 50                   # use the first 50 systems per category


def system_median_snr(row):
    """Median per-transit boxcar SNR for one synthetic system.

    Returns (median_snr, n_clipped) where n_clipped counts transits whose
    arcsin argument fell outside [0, 1] and was clipped (should be 0 for these
    non-grazing, geometry-checked systems).
    """
    b0 = float(row["b"])
    p = float(row["p"])
    period = float(row["PERIOD"])
    db_dt = float(row["DB_OVER_DT"])
    a_R = 1.0 / float(row["RS_OVER_A"])
    n_tr = int(float(row["NUM_TRANSITS"]))

    j = np.arange(n_tr)
    b_j = b0 + db_dt * period * j / YEAR_DAYS

    arg = ((1.0 + p) ** 2 - b_j ** 2) / (a_R ** 2 - b_j ** 2)
    clipped = np.clip(arg, 0.0, 1.0)
    n_clipped = int(np.sum(arg != clipped))

    t_dur = (period / np.pi) * np.arcsin(np.sqrt(clipped))   # days
    n_cad = t_dur / DT_DAY
    snr_j = (p ** 2 / SIGMA_PHOT) * np.sqrt(n_cad)
    return float(np.median(snr_j)), n_clipped


def category_percentiles(target):
    path = SNR_DIR / f"SNR_{target}.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))[:N_SYSTEMS]

    medians = []
    total_clipped = 0
    for row in rows:
        m, nc = system_median_snr(row)
        medians.append(m)
        total_clipped += nc

    medians = np.array(medians)
    q16, q50, q84 = np.percentile(medians, [16, 50, 84])
    return {
        "target": target,
        "n_systems": len(rows),
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "n_clipped": total_clipped,
    }


def main():
    stats = [category_percentiles(t) for t in TARGETS]

    print("\nPer-transit boxcar SNR of white-noise synthetic systems")
    print(f"(dt={DT_DAY*1440:.4g} min, sigma={SIGMA_PHOT:.0e}, as simulated; "
          "system-level median per-transit SNR;")
    print(f" percentiles across {N_SYSTEMS} systems)\n")
    print("| Target LLR_theory | 16th percentile | Median | 84th percentile |")
    print("| ----------------: | --------------: | -----: | --------------: |")
    for s in stats:
        print(f"| {s['target']:>17d} | {s['q16']:>15.1f} | {s['q50']:>6.1f} "
              f"| {s['q84']:>15.1f} |")

    n_clipped = sum(s["n_clipped"] for s in stats)
    n_used = sum(s["n_systems"] for s in stats)
    print(f"\nsystems used: {n_used} (50/category); "
          f"arcsin-arg clips: {n_clipped}")


if __name__ == "__main__":
    main()
