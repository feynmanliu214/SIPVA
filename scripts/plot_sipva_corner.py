#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corner plot of the SIPVA (global db/dt) posterior, formatted for an ApJ submission.

This is the *re-runnable* half of the workflow: it consumes the post-burn-in posterior samples
saved by the fit (``sipva_posterior_samples_koi_<koi>.npz``, written by
``fitting.save_posterior_samples``) and produces the figure + a display-unit summary table. It never
refits, so labels / units / formatting / styling can be iterated freely without rerunning the fit.

Run from the ``scripts/`` directory (the output root is ``../data/Output_data`` relative to cwd, the
same convention as the rest of the pipeline):

    ../.venv/bin/python plot_sipva_corner.py 377.02

Units policy
------------
Stored samples are in MODEL units. Display-unit conversions live *only* here and affect only the
plotted samples / labels / titles, never the stored posterior:
  * ``tc_1`` (zero-centered transit epoch): days -> seconds via x 86400, shown as Delta t_e [s].
  * ``k2_1`` ((Rp/Rstar)^2): shown as p = sqrt(k2) = Rp/Rstar.
All other parameters use their natural model units (rho in g/cm^3, P and the period in days,
db/dt already in 1/yr, the rest dimensionless).
"""

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless compute node
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner


# The orbital period is constrained to ~1 part in 10^6, so its full tick values (38.90788) are too
# wide for corner's rotated tick slots. We show ticks as the residual (P - P_BASE) in units of 1e-5 d
# instead, with the baseline + scale carried in a two-line axis label (see make_corner). P_BASE is a
# round value near the median.
P_BASE = 38.9079

# --- Display configuration, in the global-fit SAMPLER order ---------------------------------------
# Each entry: the internal sampler name, a short code used in the summary CSV, the display-unit
# transform applied to the stored (model-unit) samples, the axis label (with units), the title
# symbol (line 1, no units), the title unit fragment (math, appended to line 2), and the plain-text
# unit for the CSV. "q1"/"q2" match the passband-suffixed sampler names by prefix.
DISPLAY = [
    # The sampled density enters the model only through (a/R_star)^3 = G P^2 rho / (3 pi), i.e. it
    # is the total (photometric) density rho_eff = rho_star + p^3 rho_p, not the stellar density
    # alone -- label it as such. Same samples, corrected interpretation (the p^3 rho_p term is
    # ~1e-4 g/cm^3 for KOI 377.02, far below the posterior width).
    dict(key='rho',   code='rho_eff', transform=lambda x: x,
         axis=r'$\rho_{\rm eff}\equiv\rho_\star+p^3\rho_p$' + '\n' + r'$[{\rm g\,cm^{-3}}]$',
         sym=r'$\rho_{\rm eff}$', unit_tex=r'\,{\rm g\,cm^{-3}}', unit_txt='g/cm^3'),
    dict(key='tc_1',  code='te', transform=lambda x: x * 86400.0,
         axis=r'$\Delta t_{\rm e}\,[{\rm s}]$',
         sym=r'$\Delta t_{\rm e}$', unit_tex=r'\,{\rm s}', unit_txt='s'),
    dict(key='p_1',   code='P', transform=lambda x: x, three_line=True,
         axis=rf'$P-{P_BASE:g}$' + '\n' + r'$[10^{-5}\,{\rm d}]$',
         sym=r'$P$', unit_tex=r'\,{\rm d}', unit_txt='d'),
    dict(key='k2_1',  code='p', transform=lambda x: np.sqrt(x),
         axis=r'$p=R_p/R_\star$',
         sym=r'$p$', unit_tex='', unit_txt=''),
    dict(key='secw_1', transform=lambda x: x, code='secw',
         axis=r'$\sqrt{e}\cos\omega$',
         sym=r'$\sqrt{e}\cos\omega$', unit_tex='', unit_txt=''),
    dict(key='sesw_1', transform=lambda x: x, code='sesw',
         axis=r'$\sqrt{e}\sin\omega$',
         sym=r'$\sqrt{e}\sin\omega$', unit_tex='', unit_txt=''),
    dict(key='q1', transform=lambda x: x, code='q1',
         axis=r'$q_1$', sym=r'$q_1$', unit_tex='', unit_txt=''),
    dict(key='q2', transform=lambda x: x, code='q2',
         axis=r'$q_2$', sym=r'$q_2$', unit_tex='', unit_txt=''),
    dict(key='b_0', transform=lambda x: x, code='b0',
         axis=r'$b_0$', sym=r'$b_0$', unit_tex='', unit_txt=''),
    dict(key='db_dt', transform=lambda x: x, code='bdot',
         axis=r'$\dot b^{\rm S}\,[{\rm yr^{-1}}]$',
         sym=r'$\dot b^{\rm S}$', unit_tex=r'\,{\rm yr^{-1}}', unit_txt='1/yr'),
]

# Font sizes tuned so the master figure stays legible when scaled to a journal page.
LABEL_FS = 18      # axis labels (fits a unit-bearing label within one panel)
TICK_FS = 13       # tick numbers
TITLE_FS = 16      # diagonal titles
FIG_IN = 18.0      # master figure side, inches
LABELPAD = 0.22    # axis-label offset (axes fraction) added to corner's default -0.3, so unit-bearing
                   # labels clear the rotated multi-digit tick numbers instead of overprinting them


def _output_root():
    """Mirror of fitting.output_root() (kept inline so this plot-only script needs no heavy
    src/core import). Defaults to ../data/Output_data; honors TDV_OUTPUT_ROOT."""
    return os.environ.get("TDV_OUTPUT_ROOT", os.path.join('..', 'data', 'Output_data'))


def _koi_dir(koi):
    return os.path.join(_output_root(), f'koi-{koi}')


def load_posterior(koi):
    """Load the post-burn-in samples (model units, sampler order) and align them to DISPLAY."""
    npz_path = os.path.join(_koi_dir(koi), f'sipva_posterior_samples_koi_{koi}.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Run the fit first: ../.venv/bin/python fit_sipva_corner.py {koi}")
    data = np.load(npz_path, allow_pickle=True)
    samples = np.asarray(data['samples'], dtype=float)
    names = [str(n) for n in data['param_names']]
    if samples.shape[1] != len(DISPLAY):
        raise ValueError(f"expected {len(DISPLAY)} parameters, got {samples.shape[1]}")
    # Sanity-check that the stored sampler order matches DISPLAY (q1/q2 carry a passband suffix).
    for col, (name, cfg) in enumerate(zip(names, DISPLAY)):
        if not name.startswith(cfg['key']):
            raise ValueError(f"column {col}: stored '{name}' does not match expected '{cfg['key']}'")
    return samples, names


def to_display(samples):
    """Apply the per-parameter display-unit transforms. Returns a new (n_samples, n_param) array."""
    out = np.empty_like(samples)
    for col, cfg in enumerate(DISPLAY):
        out[:, col] = cfg['transform'](samples[:, col])
    return out


# --- Uncertainty-driven number formatting --------------------------------------------------------
def _round_2sig(x):
    """Round |x| to 2 significant figures. Returns (rounded_value, q) where q is the base-10
    exponent of the least-significant retained digit (e.g. 0.00084 -> (0.00084, -5))."""
    x = abs(float(x))
    if not np.isfinite(x) or x == 0.0:
        return 0.0, 0
    e = math.floor(math.log10(x))
    q = e - 1                       # 2 sig figs => digits at 10^e and 10^(e-1)
    val = round(x / 10.0**q) * 10.0**q
    return val, q


def _fixed(v, dec):
    """Fixed-point with `dec` decimals, but never render a signed zero ('-0.00' -> '0.00')."""
    s = f"{v:.{dec}f}"
    return s.lstrip('-') if float(s) == 0.0 else s


def _format_parts(q50, lo, hi):
    """Median with an asymmetric central-68% interval, formatted per the ApJ brief:
      * upper/lower uncertainties rounded to 2 sig figs (trailing zeros kept),
      * median rounded to the SAME final decimal place as the displayed uncertainties,
      * a shared x10^n factored out when plain fixed point would be awkward.
    Returns the formatted strings (median, upper, lower) and the shared exponent E (None when
    fixed-point), e.g. ('0.0098', '0.0012', '0.0012', None) or ('-0.1', '8.8', '8.6', -4)."""
    lo_r, q_lo = _round_2sig(lo)
    hi_r, q_hi = _round_2sig(hi)
    q = min(q_lo, q_hi)             # finest (smallest) decimal place among the two uncertainties

    # Scale decision: base it on the largest magnitude in play (median or either uncertainty), so a
    # median that happens to sit near zero does not force scientific notation on a healthy interval.
    # The lower bound is -4 (not -3) so a ~0.01-scale quantity such as db/dt stays in fixed point
    # (0.0098^{+0.0012}_{-0.0012}, per the brief) rather than collapsing to (9.8...)x10^-3.
    ref = max(abs(q50), lo_r, hi_r)
    E = math.floor(math.log10(ref)) if ref > 0 else 0
    use_sci = (E >= 4) or (E <= -4)

    if not use_sci:
        dec = max(0, -q)
        return _fixed(q50, dec), f"{hi_r:.{dec}f}", f"{lo_r:.{dec}f}", None

    f = 10.0**E
    dec = max(0, E - q)
    return _fixed(q50 / f, dec), f"{hi_r / f:.{dec}f}", f"{lo_r / f:.{dec}f}", E


def _join_body(m, u, d, E):
    """Combine the formatted parts into a single-line mathtext fragment (no '$', no unit), e.g.
    '0.0098^{+0.0012}_{-0.0012}' or '(-0.1^{+8.8}_{-8.6})\\times 10^{-4}'."""
    stack = f"{m}^{{+{u}}}_{{-{d}}}"
    return stack if E is None else f"({stack})\\times 10^{{{E}}}"


def make_titles_and_summary(disp_samples):
    """Compute q16/q50/q84 (on the DISPLAY samples), the asymmetric errors, the two-line LaTeX
    diagonal titles, and a tidy summary DataFrame."""
    q16, q50, q84 = np.percentile(disp_samples, [16, 50, 84], axis=0)
    lerr = q50 - q16
    uerr = q84 - q50

    titles, rows = [], []
    for i, cfg in enumerate(DISPLAY):
        m, u, d, E = _format_parts(q50[i], lerr[i], uerr[i])
        body = _join_body(m, u, d, E)
        if cfg.get('three_line') and E is None:
            # Three lines for a high-precision quantity (the period): symbol, then median+unit, then
            # the asymmetric-error stack on its own line. Keeps the title within one panel width
            # (a single-line "38.907876^{+...}_{-...} d" would spill into the neighbouring column).
            titles.append(f"{cfg['sym']}\n${m}{cfg['unit_tex']}$\n$^{{+{u}}}_{{-{d}}}$")
        else:
            # Two-line title: symbol on line 1, median+interval (+unit) on line 2.
            titles.append(f"{cfg['sym']}\n${body}{cfg['unit_tex']}$")
        rows.append(dict(parameter=cfg['code'], unit=cfg['unit_txt'],
                         q16=q16[i], q50=q50[i], q84=q84[i],
                         err_lower=lerr[i], err_upper=uerr[i],
                         title_tex=f"{body}{cfg['unit_tex']}"))  # exact line-2 math (no surrounding $)
    summary = pd.DataFrame(rows, columns=['parameter', 'unit', 'q16', 'q50', 'q84',
                                          'err_lower', 'err_upper', 'title_tex'])
    return titles, summary


def make_corner(koi, save=True, outdir=None):
    """Build (and optionally save) the SIPVA corner plot + the display-unit summary CSV for `koi`.

    ``outdir`` overrides the save directory (default: the KOI's output dir). Use a fresh,
    versioned directory to re-render with new styling without overwriting the manuscript asset."""
    samples, _ = load_posterior(koi)
    disp = to_display(samples)
    axis_labels = [cfg['axis'] for cfg in DISPLAY]
    titles, summary = make_titles_and_summary(disp)
    ndim = disp.shape[1]

    rc = {
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.formatter.use_mathtext': True,   # any 10^n renders as x10^n, not '1e-5'
        'axes.formatter.useoffset': False,     # show full tick values, never a '+3.89e1'-style offset
    }
    with mpl.rc_context(rc):
        fig = corner.corner(
            disp, labels=axis_labels,
            label_kwargs={'fontsize': LABEL_FS},
            labelpad=LABELPAD,                 # push axis labels clear of the rotated tick numbers
            quantiles=[0.16, 0.5, 0.84],
            show_titles=False,                 # custom titles below
            levels=(0.3935, 0.8647, 0.9889),   # 1,2,3 sigma (2D)
            max_n_ticks=3,                     # fewer ticks -> long P/p/te labels stop crowding
            plot_datapoints=False, fill_contours=True,
            smooth=1.0, smooth1d=1.0,
            hist_kwargs={'linewidth': 1.4}, color='#1f1f1f',
        )
        fig.set_size_inches(FIG_IN, FIG_IN)

        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            axes[i, i].set_title(titles[i], fontsize=TITLE_FS, pad=12, linespacing=1.35)

        for ax in fig.get_axes():
            ax.tick_params(labelsize=TICK_FS)
            ax.xaxis.label.set_size(LABEL_FS)
            ax.yaxis.label.set_size(LABEL_FS)

        # Period: relabel its ticks as the residual (P - P_BASE) x 1e5 so the numbers are short enough
        # for corner's rotated tick slots. FuncFormatter survives the savefig redraw; only the period
        # column's bottom x-axis and the period row's left y-axis carry visible tick labels.
        p_row = next(i for i, c in enumerate(DISPLAY) if c['key'] == 'p_1')
        p_resid = mpl.ticker.FuncFormatter(lambda v, pos: f"{(v - P_BASE) * 1e5:.1f}")
        axes[ndim - 1, p_row].xaxis.set_major_formatter(p_resid)
        axes[p_row, 0].yaxis.set_major_formatter(p_resid)

        # Open up the inter-panel gaps so neighbouring diagonal titles and rotated tick numbers do not
        # touch; headroom on top for the multi-line titles (bbox='tight' crops the outer margins).
        fig.subplots_adjust(hspace=0.16, wspace=0.14, top=0.96, right=0.985)

        if save:
            folder = outdir if outdir is not None else _koi_dir(koi)
            os.makedirs(folder, exist_ok=True)
            pdf = os.path.join(folder, f'sipva_corner_koi_{koi}.pdf')
            png = os.path.join(folder, f'sipva_corner_koi_{koi}.png')
            fig.savefig(pdf, bbox_inches='tight')                 # vector
            fig.savefig(png, dpi=600, bbox_inches='tight')        # high-res raster
            csv = os.path.join(folder, f'sipva_posterior_summary_koi_{koi}.csv')
            summary.to_csv(csv, index=False)
            print(f"Corner plot saved to {pdf}")
            print(f"Corner plot saved to {png}")
            print(f"Summary table saved to {csv}")
            print(summary.to_string(index=False))
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Make the ApJ SIPVA-posterior corner plot from saved "
                                             "samples (no refit).")
    ap.add_argument("koi", nargs="?", default="377.02", help="KOI number (default: 377.02).")
    ap.add_argument("--outdir", default=None,
                    help="Save directory override (default: the KOI's output dir). Point at a "
                         "fresh versioned directory to avoid overwriting the manuscript asset.")
    args = ap.parse_args()
    make_corner(args.koi, outdir=args.outdir)


if __name__ == "__main__":
    main()
