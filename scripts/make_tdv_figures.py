#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the three TDV result figures in ApJ / AAS-journal style.

Recomputes (does not copy) the figures sketched in ``notebooks/TDV ROC Analysis.ipynb`` from the
finished white-noise grid, restyled to ApJ conventions (STIX serif, column-appropriate sizing,
inward minor ticks, restrained line weights, colorblind-safe palette with redundant
linestyle/marker encoding).

Figures (vector PDF + PNG preview, written to outputs/figures/):
  1. tdv_roc                -- ROC with LLR 0 as the null hypothesis.
  2. tdv_recovery_rate      -- recovery fraction vs LLR, one-sided z>3 (z<-3 for t14).
  3. tdv_recovery_accuracy  -- per-level percentage error of recovered db/dt (box plot).

Data sources:
  data/Output_data/SNR artificial planets with white noise/koi-syn_snr{level}_{idx}/tdv_metrics_*.json
  data/SNR_data/SNR_{level}.csv   (true DB_OVER_DT, joined by the `name` column)

Run from the repo root under the venv:
    .venv/bin/python scripts/make_tdv_figures.py
"""
import glob
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from sklearn.metrics import auc, roc_curve

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "data" / "Output_data" / "SNR artificial planets with white noise"
_SNR_DATA_DIR = _REPO_ROOT / "data" / "SNR_data"
_FIG_DIR = _REPO_ROOT / "outputs" / "figures"

# Levels in fixed order; 0 is the null. Longest substring first so 'snr100'/'snr10'/'snr0'
# never collide when matched against a folder name.
_LEVELS = [0, 10, 20, 30, 50, 100]
_LEVELS_BY_LEN = sorted(_LEVELS, key=lambda v: -len(str(v)))

# --- ApJ trace styling: colorblind-safe (Okabe-Ito) + redundant linestyle/marker ---------------
#   SIPVA global db/dt z-score | individual-fit b t-score | individual-fit t14 t-score
_TRACES = [
    {"key": "db_dt_global_zscore", "label": "SIPVA Fit Impact Parameter",
     "color": "#0072B2", "ls": "-",   "marker": "o"},
    {"key": "t_score_b",           "label": r"$\it{Individual\ Fit}$ Impact Parameter",
     "color": "#E69F00", "ls": "--",  "marker": "s"},
    {"key": "t_score_t14",         "label": r"$\it{Individual\ Fit}$ TDV",
     "color": "#009E73", "ls": "-.",  "marker": "^"},
]


def apj_rcparams():
    """ApJ/AAS rcParams: STIX serif (the font AAS journals are typeset in), editable PDF text."""
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
        "pdf.fonttype": 42,   # embed editable TrueType text, not outlines
        "ps.fonttype": 42,
    })


def style_axes(ax):
    """Inward major+minor ticks on all four sides, no grid -- the ApJ look."""
    ax.tick_params(direction="in", which="major", top=True, right=True, width=0.8, length=3.5)
    ax.tick_params(direction="in", which="minor", top=True, right=True, width=0.6, length=2.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(False)


# --- column widths (inches) --------------------------------------------------------------------
_SINGLE_COL = 3.4
_DOUBLE_COL = 7.1


def _level_of(folder_name):
    for lv in _LEVELS_BY_LEN:
        if f"snr{lv}_" in folder_name:
            return lv
    return None


def load_results():
    """Return a DataFrame: one row per system with metrics + true db/dt, keyed by `name`/level."""
    # True db/dt per level, indexed by the system `name` (e.g. 'syn_snr10_000').
    truth = {}
    for lv in _LEVELS:
        csv = _SNR_DATA_DIR / f"SNR_{lv}.csv"
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            truth[str(r["name"])] = float(r["DB_OVER_DT"])

    rows = []
    for folder in sorted(_OUT_DIR.glob("koi-syn_snr*")):
        if not folder.is_dir():
            continue
        lv = _level_of(folder.name)
        if lv is None:
            continue
        metrics_files = glob.glob(os.path.join(folder, "tdv_metrics_*.json"))
        if not metrics_files:
            continue
        with open(metrics_files[0]) as fh:
            j = json.load(fh)
        name = f"syn_snr{lv}_{folder.name.rsplit('_', 1)[-1]}"
        rows.append({
            "name": name,
            "level": lv,
            "db_dt_global_zscore": j.get("db_dt_global_zscore"),
            "t_score_b": j.get("t_score_b"),
            "t_score_t14": j.get("t_score_t14"),
            "db_dt_global": j.get("db_dt_global"),
            "db_dt_linreg": j.get("db_dt_linreg"),
            "true_db_dt": truth.get(name),
        })
    df = pd.DataFrame(rows)
    # Drop any row missing a score needed downstream.
    return df.dropna(subset=["db_dt_global_zscore", "t_score_b", "t_score_t14"]).reset_index(drop=True)


def _save(fig, stem):
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(_FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote outputs/figures/{stem}.pdf (+ .png)")


def fig_roc(df):
    """ROC treating LLR 0 as the null. Raw signed scores (t14 sign-flipped); no clamping."""
    y_true = (df["level"] != 0).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(_SINGLE_COL, 3.2))
    for t in _TRACES:
        score = df[t["key"]].to_numpy()
        if t["key"] == "t_score_t14":
            score = -score          # negative t14 = signal direction
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=t["color"], ls=t["ls"], drawstyle="steps-post",
                label=f"{t['label']} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "tdv_roc")


def _recovery_fraction(sub, key):
    """One-sided, physical-sign recovery fraction over the systems in `sub`."""
    s = sub[key].to_numpy()
    hits = (s < -3) if key == "t_score_t14" else (s > 3)
    return hits.mean() if len(s) else 0.0


def fig_recovery_rate(df):
    """Grouped bars: recovery fraction per LLR level, one trace per detection channel."""
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, 3.2))
    x = np.arange(len(_LEVELS))
    bar_w = 0.26
    print("Recovery fractions (n per level shown):")
    for i, t in enumerate(_TRACES):
        fracs = []
        for lv in _LEVELS:
            sub = df[df["level"] == lv]
            fracs.append(_recovery_fraction(sub, t["key"]))
        ax.bar(x + (i - 1) * bar_w, fracs, width=bar_w, color=t["color"],
               edgecolor="black", linewidth=0.5, label=t["label"])
        print(f"  {t['key']:>20s}: " +
              ", ".join(f"LLR{lv}={f:.2f}" for lv, f in zip(_LEVELS, fracs)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"LLR{lv}" for lv in _LEVELS])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("LLR")
    ax.set_ylabel("Recovery Fraction")
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "tdv_recovery_rate")


def fig_recovery_accuracy(df):
    """Per-level box plot of percentage error of recovered db/dt, two estimators."""
    levels = [lv for lv in _LEVELS if lv != 0]   # true db/dt = 0 at LLR0 -> undefined percent
    methods = [
        {"col": "db_dt_global", "label": "SIPVA",                             "color": "#0072B2"},
        {"col": "db_dt_linreg", "label": r"$\it{Individual\ Fit}$ Impact Parameter",
         "color": "#E69F00"},
    ]
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, 3.2))
    x = np.arange(len(levels))
    box_w = 0.32
    legend_handles = []
    for mi, m in enumerate(methods):
        data = []
        for lv in levels:
            sub = df[(df["level"] == lv) & df[m["col"]].notna() & df["true_db_dt"].notna()]
            true = sub["true_db_dt"].to_numpy()
            est = sub[m["col"]].to_numpy()
            pct = 100.0 * np.abs(true - est) / np.abs(true)
            data.append(pct)
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
    ax.set_xticklabels([f"LLR{lv}" for lv in levels])
    ax.set_xlabel("LLR")
    ax.set_ylabel(r"Percentage error of $\dot{b}$")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "tdv_recovery_accuracy")


def main():
    apj_rcparams()
    df = load_results()
    print(f"Loaded {len(df)} systems across levels: "
          + ", ".join(f"LLR{lv}={int((df['level'] == lv).sum())}" for lv in _LEVELS))
    fig_roc(df)
    fig_recovery_rate(df)
    fig_recovery_accuracy(df)
    print("Done.")


if __name__ == "__main__":
    main()
