#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic (throwaway): plot the offending per-transit segments for the 6 problem KOIs.

For each flagged epoch: detrended flux vs time-from-predicted-center, the per-transit fitted
model (CSV medians), the catalog-expected transit span, and any sibling-planet transit span
predicted from the sibling's own (Holczer-aware) ephemeris. One multi-panel PNG per KOI in
data/diag_segments/.

Run from repo root:  PYTHONPATH=src/core MPLBACKEND=Agg .venv/bin/python scripts/diag_plot_offenders.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import evaluate_transit_flux
from data import TransitEphemeris

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "diag_segments")
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Output_data_gp_rhofix")

# epochs to plot, per KOI (CSV transit_number basis); None -> auto (worst offenders)
PLOT_EPOCHS = {
    "841.02": [25, 31, 20, 8],            # 2 long-t14 + 2 NaN-t14 healthy-looking
    "137.02": [19, 37, 63, 73],           # duration outliers (63 = no-sibling case)
    "142.01": [9, 85, 107, 119, 4, 22],   # prior-returned b + 2 NaN-t14 controls
    "209.02": [40, 67, 6, 13, 29, 54],    # sibling-captured + prior-returned
    "460.01": None,                       # deep-flux finder below
    "1856.01": None,                      # deep-flux finder below
}


def load_koi(k):
    z = np.load(os.path.join(DIR, f"koi-{k}.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    df = pd.read_csv(os.path.join(CSV, f"koi-{k}", f"per_transit_fits_koi_{k}.csv"))
    return z, meta, df


def match_row(df, tmin, tmax, center):
    """CSV row whose fitted tc is nearest the segment's predicted center (rows and segments are
    written in the same order, but tc-matching is robust to any masking mismatch)."""
    d = np.abs(df.tc_1_median.values - center)
    i = int(np.argmin(d))
    return df.iloc[i], d[i]


def pv_from_row(r):
    return [r.rho_median, r.tc_1_median, r.p_1_median, r.b_1_median, r.k2_1_median,
            r.secw_1_median, r.sesw_1_median, r.q1_Kepler_median, r.q2_Kepler_median]


def sibling_spans(meta, c, window_d):
    """(label, offset_hours, half_dur_hours, depth_ppm) for sibling transits within the window."""
    out = []
    for name, s in meta["siblings"].items():
        if s["holczer"]:
            h = s["holczer"]
            eph = TransitEphemeris(h["t0"], h["P"], np.array(h["n"]), np.array(h["oc_min"]))
        else:
            eph = TransitEphemeris(s["t0"], s["P"])
        m = int(round((c - eph.t0) / eph.period))
        for mm in (m - 1, m, m + 1):
            cs = float(eph.predict(mm))
            off_d = cs - c
            half_d = s["dur_h"] / 48.0
            if abs(off_d) < window_d + half_d:
                out.append((name, off_d * 24.0, s["dur_h"] / 2.0, s["depth_ppm"]))
    return out


def plot_epochs(k, eps, tag):
    z, meta, df = load_koi(k)
    t, f, centers, epochs = z["t"], z["f"], z["centers"], z["epochs"]
    cat_dur_h = meta["cat_dur_h"]
    n = len(eps)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    for ax in axs.flat[n:]:
        ax.remove()
    for j, ep in enumerate(eps):
        ax = axs.flat[j]
        # find the segment whose CSV row has transit_number == ep
        rows = df[df.transit_number == ep]
        if len(rows) == 0:
            ax.set_title(f"ep {ep}: no CSV row"); continue
        r = rows.iloc[0]
        i = int(np.argmin(np.abs(centers - r.tc_1_median)))
        c = centers[i]
        tt, ff = np.asarray(t[i], float), np.asarray(f[i], float)
        dt_h = (tt - c) * 24.0
        ax.plot(dt_h, ff, ".", ms=3, color="0.4", label="detrended flux")
        # fitted model on a dense grid
        grid = np.linspace(tt.min(), tt.max(), 1200)
        try:
            mod = evaluate_transit_flux(pv_from_row(r), grid)
            ax.plot((grid - c) * 24.0, mod, "-", color="crimson", lw=1.5, label="per-transit fit")
        except Exception as e:
            ax.text(0.02, 0.02, f"model failed: {e}", transform=ax.transAxes, fontsize=7)
        # expected target transit (catalog duration, predicted center)
        ax.axvspan(-cat_dur_h / 2, cat_dur_h / 2, color="tab:blue", alpha=0.12,
                   label="expected target transit")
        ax.axvline(0, color="tab:blue", lw=0.8, ls=":")
        # fitted tc
        ax.axvline((r.tc_1_median - c) * 24.0, color="crimson", lw=0.8, ls="--", label="fitted tc")
        # siblings
        win_d = max(abs(dt_h.min()), abs(dt_h.max())) / 24.0
        for (nm, off_h, half_h, depth) in sibling_spans(meta, c, win_d):
            ax.axvspan(off_h - half_h, off_h + half_h, color="tab:orange", alpha=0.2)
            ax.text(off_h, ax.get_ylim()[0] if False else 1.0005, nm, color="tab:orange",
                    fontsize=7, ha="center")
        t14m = r.t14_1_median * 24 * 60 if np.isfinite(r.t14_1_median) else float("nan")
        berr = (r.b_1_lerr + r.b_1_uerr) / 2
        npts = len(tt)
        cad_min = float(np.median(np.diff(np.sort(tt)))) * 1440
        ax.set_title(f"KOI {k} ep{ep}: t14={t14m:.0f}min b={r.b_1_median:.2f}±{berr:.2f} "
                     f"rho={r.rho_median:.2f} k2={r.k2_1_median:.4f}\n"
                     f"n={npts} cad={cad_min:.1f}min rho_ok={r.rho_consistent}", fontsize=8)
        ax.set_xlabel("hours from predicted center")
        if j % ncol == 0:
            ax.set_ylabel("normalized flux")
        if j == 0:
            ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    out = os.path.join(DIR, f"diag_{k}_{tag}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


def deep_flux_epochs(k, nworst=6):
    """Epochs whose minimum flux is far below the rest (the 460.01 / 1856.01 complaint)."""
    z, meta, df = load_koi(k)
    t, f, centers = z["t"], z["f"], z["centers"]
    mins = np.array([float(np.min(np.asarray(x, float))) for x in f])
    order = np.argsort(mins)
    eps = []
    print(f"KOI {k}: segment flux minima (worst {nworst}):")
    for i in order[:nworst]:
        rows = df.iloc[[int(np.argmin(np.abs(df.tc_1_median.values - centers[i])))]]
        ep = int(rows.transit_number.iloc[0])
        print(f"  ep{ep}: min_flux={mins[i]:.5f} (median of all minima {np.median(mins):.5f})")
        eps.append(ep)
    return eps


def main():
    for k, eps in PLOT_EPOCHS.items():
        if not os.path.exists(os.path.join(DIR, f"koi-{k}.npz")):
            print(f"[skip] {k}: segments not extracted yet")
            continue
        if eps is None:
            eps = deep_flux_epochs(k)
        plot_epochs(k, eps, "offenders")


if __name__ == "__main__":
    main()
