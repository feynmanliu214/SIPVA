#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot ROC, recovery rate, and recovery accuracy for the 2026-06-09 SNR TDV grid.

Reads the two consolidated run summaries (white-noise and realistic-Kepler-noise),
each with 50 systems at SNR 0/5/10/20/30/50 (SNR 0 = the db/dt=0 null), and emits,
per noise set, three ApJ-styled figures plus a printed numeric table:

  1. tdv_roc_<set>             -- ROC, SNR 0 as the null hypothesis.
  2. tdv_recovery_rate_<set>   -- detection fraction per SNR level (one-sided z>3).
  3. tdv_recovery_accuracy_<set> -- per-level percentage error of recovered db/dt (box).

Detection channels available in the summary: SIPVA global db/dt z-score and the
individual-fit impact-parameter t-score.

Run from the repo root under the venv:
    .venv/bin/python scripts/plot_tdv_grid_results.py
"""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from sklearn.metrics import auc, roc_curve

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SNR_DATA_DIR = _REPO_ROOT / "data" / "SNR_data"
_OUT_DIR = _REPO_ROOT / "data" / "Output_data"
_FIG_DIR = _REPO_ROOT / "outputs" / "figures"

_LEVELS = [0, 5, 10, 20, 30, 50]
_NULL = 0
_THRESH = 3.0   # one-sided detection threshold on the (signed) score
# SNR 5 is the hard, partially-recoverable regime. The ROC now keeps it in the positive pool so
# the AUC reflects separability over the full injected-signal range -- excluding it pinned the
# white SIPVA AUC to a misleading 1.00. The recovery-rate *bars* still drop it, to characterize
# detection against confidently-injected signal. (The accuracy box plot shows every signal level.)
_RR_EXCLUDE = {5}
_RR_LEVELS = [lv for lv in _LEVELS if lv not in _RR_EXCLUDE]

_SETS = [
    {"tag": "white", "csv": "tdv_recovery_summary_white.csv", "title": "White noise",
     "subdir": "white_noise", "prefix": "koi-"},
    {"tag": "real",  "csv": "tdv_recovery_summary_real.csv",  "title": "Realistic Kepler noise",
     "subdir": "real_kepler_noise", "prefix": "koi-"},
]

# Detection channels. db_dt_global_zscore / t_score_b come from the summary CSV; t_score_t14
# is pulled per-system from the metrics JSONs. The impact-parameter channels point positive for
# the synthetic (positive db/dt) signal; t14 points negative (rising b shortens the transit), so
# its signal direction is sign-flipped for the ROC and detected with score < -THRESH.
_SIGN_FLIP = {"t_score_t14"}   # negative is the signal direction

_TRACES = [
    {"key": "db_dt_global_zscore", "label": "SIPVA fit Impact Parameter",
     "color": "#0072B2", "ls": "-",  "marker": "o"},
    {"key": "t_score_b",           "label": r"$\it{Individual\ fit}$ Impact Parameter",
     "color": "#E69F00", "ls": "--", "marker": "s"},
    {"key": "t_score_t14",         "label": r"$\it{Individual\ fit}$ TDV",
     "color": "#009E73", "ls": "-.", "marker": "^"},
]


def _signed_score(df, key):
    """Score oriented so that the synthetic signal pushes it positive."""
    s = df[key].to_numpy()
    return -s if key in _SIGN_FLIP else s


def apj_rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax):
    ax.tick_params(direction="in", which="major", top=True, right=True, width=0.8, length=3.5)
    ax.tick_params(direction="in", which="minor", top=True, right=True, width=0.6, length=2.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(False)


_SINGLE_COL = 3.4
_DOUBLE_COL = 7.1


def _load_t14(subdir, prefix, name):
    """Pull t_score_t14 from a system's metrics JSON (folder = <prefix><name>)."""
    folder = _OUT_DIR / subdir / f"{prefix}{name}"
    files = glob.glob(str(folder / "tdv_metrics_*.json"))
    if not files:
        return np.nan
    with open(files[0]) as fh:
        return json.load(fh).get("t_score_t14", np.nan)


def load_summary(s):
    df = pd.read_csv(_SNR_DATA_DIR / s["csv"])
    df = df[df["status"] == "ok"].copy()
    df["level"] = df["snr_level"].astype(int)
    df["t_score_t14"] = df["name"].map(
        lambda n: _load_t14(s["subdir"], s["prefix"], n))
    return df.dropna(
        subset=["db_dt_global_zscore", "t_score_b", "t_score_t14"]).reset_index(drop=True)


def _save(fig, stem):
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(_FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote outputs/figures/{stem}.pdf (+ .png)")


def fig_roc(df, tag, title):
    """ROC with SNR 0 (db/dt=0) as the null; raw signed scores, no clamping.

    Negatives are the SNR-0 null; positives are every injected level (5/10/20/30/50), including
    the hard SNR-5 regime, so the AUC reflects separability over the full signal range.
    """
    y_true = (df["level"] != _NULL).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(_SINGLE_COL, 3.2))
    aucs = {}
    for t in _TRACES:
        score = _signed_score(df, t["key"])
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        aucs[t["key"]] = roc_auc
        ax.plot(fpr, tpr, color=t["color"], ls=t["ls"], drawstyle="steps-post",
                label=f"{t['label']} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, f"tdv_roc_{tag}")
    return aucs


def _recovery_fraction(sub, key):
    s = _signed_score(sub, key)
    return float((s > _THRESH).mean()) if len(s) else 0.0


def fig_recovery_curve(df, tag, title):
    """Detection-completeness curve: recovery fraction vs injected SNR, one line per method.

    Log SNR x-axis over the signal levels (5..50); a line plateaus cleanly at saturation so no
    level is dropped. The null (SNR 0) false-positive rate is reported in the figure caption
    (printed to stdout) rather than placed on the axis. The method gap (SIPVA vs the two
    individual-fit channels) is the intended visual payload.
    """
    signal_levels = [lv for lv in _LEVELS if lv != _NULL]
    xpos = np.arange(len(signal_levels))            # evenly spaced categorical positions
    fig, ax = plt.subplots(figsize=(_SINGLE_COL + 0.6, 3.2))
    table = {}
    for t in _TRACES:
        fracs = [_recovery_fraction(df[df["level"] == lv], t["key"]) for lv in signal_levels]
        table[t["key"]] = fracs
        ax.plot(xpos, fracs, color=t["color"], ls=t["ls"], marker=t["marker"],
                ms=5, mfc=t["color"], mec="black", mew=0.5, label=t["label"])

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(lv) for lv in signal_levels])
    ax.xaxis.set_minor_locator(plt.NullLocator())   # only the real SNR levels get ticks
    ax.set_xlim(xpos[0] - 0.4, xpos[-1] + 0.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("LLR Category")
    ax.set_ylabel("Recovery Fraction")
    ax.axhline(0.5, color="0.7", ls=":", lw=0.8, zorder=0)   # 50% completeness guide

    # Null false-positive rate per method (the SNR-0 systems). Printed for the figure caption
    # rather than annotated on the plot, to keep the legend area uncluttered.
    _short = {"db_dt_global_zscore": "SIPVA", "t_score_b": "individual-b",
              "t_score_t14": "individual-TDV"}
    fpr = {t["key"]: _recovery_fraction(df[df["level"] == _NULL], t["key"]) for t in _TRACES}
    caption_fpr = "Null false-positive rates: " + ", ".join(
        f"{_short[t['key']]} {fpr[t['key']]*100:.0f}%" for t in _TRACES)
    print(f"  [{tag}] caption — {caption_fpr}")

    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(0.98, 0.46))
    style_axes(ax)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    fig.tight_layout()
    _save(fig, f"tdv_recovery_curve_{tag}")
    return table


def fig_recovery_rate(df, tag, title):
    """Grouped bars: detection fraction per SNR level. SNR 0 bar = false-positive rate."""
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, 3.2))
    x = np.arange(len(_RR_LEVELS))
    bar_w = 0.26
    table = {}
    for i, t in enumerate(_TRACES):
        fracs = [_recovery_fraction(df[df["level"] == lv], t["key"]) for lv in _RR_LEVELS]
        table[t["key"]] = fracs
        ax.bar(x + (i - 1) * bar_w, fracs, width=bar_w, color=t["color"],
               edgecolor="black", linewidth=0.5, label=t["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([("Null" if lv == _NULL else f"SNR {lv}") for lv in _RR_LEVELS])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injected SNR")
    ax.set_ylabel("Recovery Fraction")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, f"tdv_recovery_rate_{tag}")
    return table


def fig_recovery_accuracy(df, tag, title):
    """Per-level box plot of percentage error of recovered db/dt, two estimators."""
    levels = [lv for lv in _LEVELS if lv != _NULL]   # true db/dt = 0 at null -> undefined percent
    methods = [
        {"col": "db_dt_global", "label": "SIPVA",
         "color": "#0072B2"},
        {"col": "db_dt_linreg", "label": r"$\it{Individual\ Fit}$ Impact Parameter",
         "color": "#E69F00"},
    ]
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, 3.2))
    x = np.arange(len(levels))
    box_w = 0.32
    legend_handles = []
    medians = {}
    for mi, m in enumerate(methods):
        data, med = [], []
        for lv in levels:
            sub = df[(df["level"] == lv) & df[m["col"]].notna() & df["true_db_dt"].notna()]
            true = sub["true_db_dt"].to_numpy()
            est = sub[m["col"]].to_numpy()
            pct = 100.0 * np.abs(true - est) / np.abs(true)
            data.append(pct)
            med.append(float(np.median(pct)) if len(pct) else float("nan"))
        medians[m["col"]] = med
        positions = x + (mi - 0.5) * box_w
        bp = ax.boxplot(data, positions=positions, widths=box_w * 0.9, showfliers=False,
                        patch_artist=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(m["color"])
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set_color("black")
                ln.set_linewidth(0.8)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=m["color"],
                                            alpha=0.65, edgecolor="black", label=m["label"]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"SNR {lv}" for lv in levels])
    ax.set_xlabel("Injected SNR")
    ax.set_ylabel(r"Percentage error of $\dot{b}$")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    _save(fig, f"tdv_recovery_accuracy_{tag}")
    return levels, medians


def main():
    apj_rcparams()
    for s in _SETS:
        df = load_summary(s)
        print(f"\n=== {s['title']} ({s['tag']}) ===")
        print("Loaded " + ", ".join(
            f"SNR{lv}={int((df['level'] == lv).sum())}" for lv in _LEVELS))

        aucs = fig_roc(df, s["tag"], s["title"])

        fig_recovery_curve(df, s["tag"], s["title"])     # new: completeness-curve form
        rate = fig_recovery_rate(df, s["tag"], s["title"])
        print("Recovery fraction (one-sided z>3):")
        for key, fracs in rate.items():
            print(f"  {key:>20s}: " +
                  ", ".join(f"{('Null' if lv == 0 else 'SNR'+str(lv))}={f*100:.0f}%"
                            for lv, f in zip(_RR_LEVELS, fracs)))

        levels, medians = fig_recovery_accuracy(df, s["tag"], s["title"])
        print("Median |db/dt| percentage error:")
        for col, med in medians.items():
            print(f"  {col:>14s}: " +
                  ", ".join(f"SNR{lv}={m:.0f}%" for lv, m in zip(levels, med)))
        print("AUC: " + ", ".join(f"{k}={v:.3f}" for k, v in aucs.items()))
    print("\nDone.")


if __name__ == "__main__":
    main()
