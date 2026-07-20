#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the combined per-transit linear-regression figure into two separate plots per KOI.

For every planet under ``data/Output_data/Kepler Planets/``, this reads the saved
``per_transit_fits_koi_*.csv`` (no re-fitting) and reproduces the two regressions the
pipeline runs, each on its own axes:

  1. linear_regression_impact_parameter_koi_<koi>  -- b vs transit time (all transits)
  2. linear_regression_transit_duration_koi_<koi>  -- T14 [min] vs transit time (valid T14)

The fit matches ``src.core.analysis.Linear_regression`` exactly: weighted least squares
(curve_fit, absolute_sigma=True) with per-point sigma = max(lower, upper) error. The
duration regression is drawn only when > 4 valid transits exist (pipeline rule). Computed
t-scores / db/dt are cross-checked against each planet's tdv_metrics JSON.

Run from the repo root under the venv:
    .venv/bin/python scripts/plot_kepler_linreg_split.py
"""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PLANETS_DIR = _REPO_ROOT / "data" / "Output_data" / "Kepler Planets"

_APJ_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
}


def _linfit(x, y, yerr):
    """Weighted linear fit identical to analysis.Linear_regression.

    Returns (slope, slope_err, intercept, t_value, r_squared).
    """
    model = lambda xx, A, B: A * xx + B
    p, pcov = curve_fit(model, x, y, p0=[0, 0], sigma=yerr, absolute_sigma=True)
    p_sigma = np.sqrt(np.diag(pcov))
    slope, slope_err = p[0], p_sigma[0]
    t_value = slope / slope_err if slope_err else np.nan
    y_fit = model(x, *p)
    ssr = np.sum((y - y_fit) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ssr / sst if sst else np.nan
    return slope, slope_err, p[1], t_value, r_squared


def _panel(koi, x, y, yerr, ylabel, stem, folder):
    """Single-axes regression figure (measurements + weighted fit line)."""
    x, y, yerr = np.asarray(x, float), np.asarray(y, float), np.asarray(yerr, float)
    slope, slope_err, intercept, t_value, r_squared = _linfit(x, y, yerr)

    with mpl.rc_context(_APJ_RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, slope * xs + intercept, color="C3",
                label=r"fit ($t={:.2f}$, $R^2={:.2f}$)".format(t_value, r_squared))
        ax.errorbar(x, y, yerr, fmt="o", ms=3, color="C0", ecolor="0.6",
                    elinewidth=0.8, capsize=1.5, label="measurements")
        ax.set_xlabel("Transit time [BKJD, days]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"KOI {koi}")
        ax.legend(loc="best")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
    return slope, t_value


def process_planet(folder):
    koi = folder.name.replace("koi-", "")
    csvs = glob.glob(str(folder / "per_transit_fits_koi_*.csv"))
    if not csvs:
        print(f"  [skip] {folder.name}: no per_transit_fits CSV")
        return
    df = pd.read_csv(csvs[0])

    # --- Impact parameter: b vs transit time, all transits (pipeline uses the full arrays).
    bm = df.dropna(subset=["tc_1_median", "b_1_median"])
    b_x = bm["tc_1_median"].to_numpy()
    b_y = bm["b_1_median"].to_numpy()
    b_err = np.maximum(bm["b_1_lerr"].to_numpy(), bm["b_1_uerr"].to_numpy())
    b_slope, b_t = _panel(koi, b_x, b_y, b_err, r"Impact parameter $b$",
                          f"linear_regression_impact_parameter_koi_{koi}", folder)

    # --- Transit duration (TDV): T14 [min] vs transit time, valid T14 only, > 4 transits.
    dm = df.dropna(subset=["tc_1_median", "t14_1_median"])
    note = ""
    d_t = None
    if len(dm) > 4:
        d_x = dm["tc_1_median"].to_numpy()
        d_y = dm["t14_1_median"].to_numpy() * 24 * 60                      # days -> minutes
        d_err = np.maximum(dm["t14_1_lerr"].to_numpy(), dm["t14_1_uerr"].to_numpy()) * 24 * 60
        _, d_t = _panel(koi, d_x, d_y, d_err, "Transit duration [min]",
                        f"linear_regression_transit_duration_koi_{koi}", folder)
    else:
        note = f"  (duration regression skipped: only {len(dm)} valid transits)"

    # --- Cross-check against the stored metrics JSON.
    chk = ""
    mj = glob.glob(str(folder / "tdv_metrics_*.json"))
    if mj:
        m = json.load(open(mj[0]))
        chk = (f"  [json t_b={m.get('t_score_b'):+.2f}"
               f" db_dt_linreg={m.get('db_dt_linreg'):+.4f}")
        chk += (f" t_t14={m.get('t_score_t14'):+.2f}]" if m.get("t_score_t14") is not None
                else " t_t14=None]")
    print(f"  {folder.name}: b t={b_t:+.2f} (db/dt_linreg={b_slope*365:+.4f}/yr)"
          + (f", T14 t={d_t:+.2f}" if d_t is not None else ", T14 -")
          + chk + note)


def main():
    folders = sorted(p for p in _PLANETS_DIR.iterdir() if p.is_dir())
    print(f"Splitting linear-regression figures for {len(folders)} planets "
          f"(my recompute | stored JSON):")
    for folder in folders:
        process_planet(folder)
    print("Done. Wrote linear_regression_impact_parameter_* and "
          "linear_regression_transit_duration_* (PNG+PDF) into each KOI folder.")


if __name__ == "__main__":
    main()
