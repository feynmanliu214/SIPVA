#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication (ApJ) impact-parameter linear-regression figures for Output_data_gp.

For every planet under ``data/Output_data_gp/koi-*/`` this remakes the impact
parameter (b vs transit time) regression figure, ApJ-styled, with:

  * the best-fit weighted-least-squares line,
  * a shaded 1-sigma confidence band built from the FULL fit covariance,
        sigma_fit(x)^2 = [1, x] Cov(beta) [1, x]^T,
  * an on-panel annotation
        m_b^IF = <slope> +/- <slope_err>  [yr^-1]
        Z_b^IF = <slope significance>
    (no R^2 shown).

Self-consistency (per request): the plotted line, the 1-sigma band, the
displayed slope, its uncertainty and Z all come from ONE weighted regression
recomputed from the saved per-transit b and 1-sigma errors -- never a saved
slope spliced onto a freshly computed intercept/band. The recomputed slope
(x365 -> yr^-1) and Z = slope/slope_err are then cross-checked against each
planet's stored tdv_metrics JSON (db_dt_linreg, t_score_b). If ANY planet
disagrees beyond tolerance the script reports the discrepancy and writes NO
figures, because Figure 5 and Table 3 must share the same regression convention.

The regression convention reproduces src.core.pipeline exactly:
    x   = tc_1_median  (all transits, raw BKJD days)
    y   = b_1_median
    sig = max(b_1_lerr, b_1_uerr)      (model.calculate_uncertainty)
    curve_fit(A*x + B, sigma=sig, absolute_sigma=True);  db_dt_linreg = A * 365

Run from the repo root under the venv:
    .venv/bin/python scripts/plot_impact_parameter_linreg_gp.py
"""
import glob
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "Output_data_gp"

_DAYS_PER_YEAR = 365.0      # pipeline uses *365 exactly (pipeline.py:640)

# Cross-check tolerances against the stored JSON (db_dt_linreg [yr^-1], t_score_b).
_RTOL = 2e-3
_ATOL_SLOPE = 1e-4
_ATOL_Z = 2e-2

_APJ_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
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

_FIT_C = "#c1272d"        # red fit line + band
_DATA_C = "#1f3b73"       # navy markers
_XLABEL = r"Transit mid-time, $t_c$ [BKJD, days]"
_YLABEL = r"Impact parameter, $b$"

# Standalone per-KOI panels are rendered in the combined-grid's cell format
# (small fonts, NO axis labels) at ONE fixed figure size + axes box, with no
# tight bbox, so the 16 panels tile into a clean 4x4 in LaTeX/Overleaf (add the
# shared x/y axis labels around the assembled block there).
_PANEL_FIGSIZE = (2.3, 1.8)
_PANEL_ADJUST = dict(left=0.16, right=0.97, bottom=0.15, top=0.86)


def _fmt_val_err(v, e):
    """Format value +/- error in yr^-1 with the error carried to ~2 sig figs."""
    if not (np.isfinite(e) and e > 0):
        return f"{v:.3g}", f"{e:.3g}"
    exp = math.floor(math.log10(abs(e)))
    ndec = max(0, -(exp - 1))            # two significant figures of the error
    ndec = min(ndec, 6)
    return f"{v:.{ndec}f}", f"{e:.{ndec}f}"


def _wls(x, y, sig):
    """Weighted linear fit identical to analysis.Linear_regression.

    Returns dict with slope/intercept (per day), their errors, full 2x2 pcov,
    db_dt_linreg [yr^-1], its error, and Z = slope/slope_err.
    """
    model = lambda xx, A, B: A * xx + B
    p, pcov = curve_fit(model, x, y, p0=[0.0, 0.0], sigma=sig, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    slope, slope_err = float(p[0]), float(perr[0])
    z = slope / slope_err if slope_err else np.nan
    return {
        "slope_day": slope,
        "slope_day_err": slope_err,
        "intercept": float(p[1]),
        "pcov": pcov,
        "m_yr": slope * _DAYS_PER_YEAR,
        "m_yr_err": slope_err * _DAYS_PER_YEAR,
        "Z": z,
    }


def _load_planet(folder):
    """Read per-transit b data + stored JSON; recompute the WLS fit. Returns a record."""
    koi = folder.name.replace("koi-", "")
    csvs = glob.glob(str(folder / "per_transit_fits_koi_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0]).dropna(subset=["tc_1_median", "b_1_median",
                                             "b_1_lerr", "b_1_uerr"])
    x = df["tc_1_median"].to_numpy(float)
    y = df["b_1_median"].to_numpy(float)
    sig = np.maximum(df["b_1_lerr"].to_numpy(float), df["b_1_uerr"].to_numpy(float))

    fit = _wls(x, y, sig)

    stored = {}
    mj = glob.glob(str(folder / "tdv_metrics_*.json"))
    if mj:
        with open(mj[0]) as fh:
            m = json.load(fh)
        stored = {"db_dt_linreg": m.get("db_dt_linreg"),
                  "t_score_b": m.get("t_score_b")}

    return {"koi": koi, "folder": folder, "n": len(x),
            "x": x, "y": y, "sig": sig, "fit": fit, "stored": stored}


def _check(rec):
    """Compare recomputed slope (yr^-1) and Z to the stored JSON. Returns (ok, msgs)."""
    fit, st = rec["fit"], rec["stored"]
    msgs, ok = [], True
    s_slope, s_z = st.get("db_dt_linreg"), st.get("t_score_b")
    if s_slope is None or s_z is None:
        return False, [f"KOI {rec['koi']}: stored db_dt_linreg / t_score_b missing"]
    if not np.isclose(fit["m_yr"], s_slope, rtol=_RTOL, atol=_ATOL_SLOPE):
        ok = False
        msgs.append(f"KOI {rec['koi']}: slope recomputed {fit['m_yr']:+.6f} vs "
                    f"stored {s_slope:+.6f} yr^-1 (delta {fit['m_yr']-s_slope:+.2e})")
    if not np.isclose(fit["Z"], s_z, rtol=_RTOL, atol=_ATOL_Z):
        ok = False
        msgs.append(f"KOI {rec['koi']}: Z recomputed {fit['Z']:+.4f} vs "
                    f"stored {s_z:+.4f} (delta {fit['Z']-s_z:+.2e})")
    return ok, msgs


def _draw_panel(ax, rec, annot_fs=8.0, title=True, title_fs=None):
    """Draw one b-vs-time regression panel: band, fit line, data, annotation."""
    x, y, sig, fit = rec["x"], rec["y"], rec["sig"], rec["fit"]
    pcov = fit["pcov"]

    xs = np.linspace(x.min(), x.max(), 300)
    yfit = fit["slope_day"] * xs + fit["intercept"]
    # sigma_fit(x)^2 = [x, 1] Cov [x, 1]^T  (param order [slope, intercept]).
    var = pcov[0, 0] * xs ** 2 + 2.0 * pcov[0, 1] * xs + pcov[1, 1]
    sfit = np.sqrt(np.clip(var, 0.0, None))

    ax.fill_between(xs, yfit - sfit, yfit + sfit, color=_FIT_C, alpha=0.20,
                    lw=0, zorder=1)
    ax.plot(xs, yfit, color=_FIT_C, lw=1.2, zorder=4)
    ax.errorbar(x, y, sig, fmt="o", ms=3, color=_DATA_C, mec=_DATA_C,
                ecolor="0.6", elinewidth=0.7, capsize=1.2, zorder=3)

    mstr, estr = _fmt_val_err(fit["m_yr"], fit["m_yr_err"])
    annot = (r"$m_b^{\mathrm{IF}} = %s \pm %s\ \mathrm{yr^{-1}}$" % (mstr, estr)
             + "\n" + r"$Z_b^{\mathrm{IF}} = %.2f$" % fit["Z"])
    ax.text(0.045, 0.955, annot, transform=ax.transAxes, ha="left", va="top",
            fontsize=annot_fs,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      lw=0.6, alpha=0.85), zorder=5)

    if title:
        ax.set_title(f"KOI {rec['koi']}", fontsize=title_fs)
    ax.margins(x=0.03)


def _make_per_koi(rec):
    """Grid-format standalone panel; overwrites linear_regression_koi_<koi>.{png,pdf}.

    Matches the combined grid's cell look (title 8 pt, annotation 6.5 pt,
    ticks 8 pt) with NO x/y axis labels, so the 16 panels tile into a clean 4x4
    in LaTeX/Overleaf (add the two shared axis labels around the block there).
    Every panel uses one fixed figure size + axes box (no tight bbox) so the
    plot areas align when tiled.
    """
    with mpl.rc_context(_APJ_RC):
        fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE)
        _draw_panel(ax, rec, annot_fs=6.5, title=True, title_fs=8)
        fig.subplots_adjust(**_PANEL_ADJUST)
        stem = rec["folder"] / f"linear_regression_koi_{rec['koi']}"
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}")
        plt.close(fig)


def _make_grid(recs):
    """Single multi-panel grid figure of all planets -> data/Output_data_gp/."""
    n = len(recs)
    ncol = 4
    nrow = math.ceil(n / ncol)
    with mpl.rc_context(_APJ_RC):
        # Wide/short full-text-width footprint (was 7.1 x 8.0): panels become
        # ~1.6:1 (wide) and the whole figure takes less vertical space.
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 5.2),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < n:
                _draw_panel(ax, recs[i], annot_fs=6.5, title=True, title_fs=8)
            else:
                ax.set_visible(False)
        fig.supxlabel(_XLABEL, fontsize=10)
        fig.supylabel(_YLABEL, fontsize=10)
        # tight_layout sets sensible outer margins (incl. the suplabels); then
        # tighten only the inter-panel gaps (full BKJD ticks are retained, so the
        # gaps shrink only as far as the tick numbers/titles allow).
        fig.tight_layout(rect=(0.015, 0.015, 1, 1))
        fig.subplots_adjust(wspace=0.30, hspace=0.50)
        stem = _DATA_DIR / "linear_regression_impact_parameter_grid"
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
    return f"{stem}.png"


def main():
    folders = sorted((p for p in _DATA_DIR.iterdir()
                      if p.is_dir() and p.name.startswith("koi-")),
                     key=lambda p: float(p.name.replace("koi-", "")))
    recs = [r for r in (_load_planet(f) for f in folders) if r is not None]
    print(f"Loaded {len(recs)} planets from {_DATA_DIR}")

    # --- Phase 1: recompute & verify against stored JSON before drawing anything.
    print("\nCross-check (recomputed | stored):")
    print(f"  {'KOI':<9}{'n':>4}  {'m_b^IF [yr^-1]':>26}  {'Z_b^IF':>18}   ok")
    all_ok, problems = True, []
    for r in recs:
        ok, msgs = _check(r)
        all_ok &= ok
        problems += msgs
        f, s = r["fit"], r["stored"]
        print(f"  {r['koi']:<9}{r['n']:>4}  "
              f"{f['m_yr']:+.5f} | {s.get('db_dt_linreg'):+.5f}   "
              f"{f['Z']:+8.3f} | {s.get('t_score_b'):+8.3f}   {'OK' if ok else 'XX'}")

    if not all_ok:
        print("\n*** DISCREPANCY: recomputed regression does not match stored JSON. ***")
        for m in problems:
            print("   - " + m)
        print("No figures written (Figure 5 and Table 3 must share one convention).")
        sys.exit(1)
    print("\nAll planets consistent with stored db_dt_linreg / t_score_b.")

    # --- Phase 2: render (per-KOI overwrite + combined grid).
    for r in recs:
        _make_per_koi(r)
    grid = _make_grid(recs)
    print(f"Wrote per-KOI linear_regression_koi_<koi>.(png,pdf) for {len(recs)} planets "
          f"(overwritten in place).")
    print(f"Wrote grid figure: {grid} (+ .pdf)")


if __name__ == "__main__":
    main()
