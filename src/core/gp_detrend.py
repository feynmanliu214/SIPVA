"""Gaussian-Process light-curve detrending (non-JAX celerite2 port of pyKepler's gpfit_with_mask).

Ported from wangxianyu7/pykepler (``detrend/gpfit.py`` + ``detrend/koi.py``): per Kepler quarter,
fit a Matern-3/2 GP baseline with the in-transit points down-weighted by *error inflation* (their
diagonal variance is blown up so they carry ~zero likelihood weight), then divide the flux by the
predicted baseline. The original uses celerite2's JAX backend + jaxopt; this port uses celerite2's
default (numpy/scipy) backend with ``scipy.optimize.minimize`` so the cluster venv needs no JAX. The
hyperparameter init/bounds match the original verbatim (Matern-3/2 amplitude ``sigma``,
length-scale ``rho``, a white ``jitter``, and a constant ``mean``).

``gp_baseline`` returns the predicted baseline (same length/order as its input) and **raises** on a
degenerate segment or a failed/insane fit, so the caller (``data.lightcurve_extract``) can fall back
to the Savitzky-Golay path for that product.
"""
import numpy as np
import celerite2
from celerite2 import terms
from scipy.optimize import minimize

# Multiplicative blow-up applied to the per-point error of in-transit cadences so the GP baseline
# ignores them (matches pykepler's 1e6 * f_std error inflation).
MASK_INFLATION = 1e6


def _sanitize_err(e, f):
    """Replace non-finite or non-positive flux errors with a robust successive-difference white-noise
    estimate (1.4826 * MAD(diff(f)) / sqrt(2)), so the GP jitter init/bounds are always well-defined.
    ``f`` must be time-ordered. Returns a finite, positive error array."""
    e = np.array(e, dtype=float)
    bad = ~np.isfinite(e) | (e <= 0)
    if np.any(bad):
        df = np.diff(f)
        sig = 0.0
        if df.size and np.any(np.isfinite(df)):
            mad = np.median(np.abs(df - np.median(df)))
            sig = 1.4826 * mad / np.sqrt(2.0)
        if not (np.isfinite(sig) and sig > 0):
            fstd = np.std(f)
            sig = fstd if (np.isfinite(fstd) and fstd > 0) else 1.0
        e[bad] = sig
    return e


def gp_baseline(t, f, e, mask):
    """Fit a Matern-3/2 GP baseline to one quarter, masking in-transit points by error inflation.

    Parameters
    ----------
    t, f, e : array-like
        Time, flux, and per-point flux error for one product/quarter (any order; sorted internally).
    mask : array-like of bool
        True for in-transit cadences (down-weighted in the fit, but still predicted across).

    Returns
    -------
    baseline : np.ndarray
        GP-predicted baseline at every input cadence, in the SAME order as ``t``. Flatten the light
        curve by dividing ``f / baseline``.

    Raises
    ------
    ValueError
        On a degenerate segment (too few points, duplicate timestamps, non-finite scale) or if the
        fitted baseline is non-finite or non-positive. The caller should fall back to savgol.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = t.size
    if n < 2:
        raise ValueError("gp_baseline: need >= 2 points")

    # celerite2 requires strictly-increasing time. Sort, fit, then map the baseline back to the
    # caller's original ordering.
    order = np.argsort(t)
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    ts, fs, es, ms = t[order], f[order], e[order], mask[order]
    if np.any(np.diff(ts) <= 0):
        raise ValueError("gp_baseline: non-increasing time after sort (duplicate timestamps)")

    es = _sanitize_err(es, fs)
    dt = np.median(np.diff(ts))
    T = ts[-1] - ts[0]
    f_mean, f_std, e_mean = np.mean(fs), np.std(fs), np.mean(es)
    if not (np.isfinite(f_std) and f_std > 0 and dt > 0 and T > 0):
        raise ValueError("gp_baseline: degenerate segment (std/dt/T not positive-finite)")

    # In-transit error inflation (added in quadrature to the diagonal below).
    e_inflate = ms * (MASK_INFLATION * f_std)

    # Params: [mean, ln_jitter, ln_sigma, ln_rho]; bounds verbatim from pykepler's gpfit_with_mask.
    p0 = np.array([f_mean, np.log(e_mean), np.log(f_std), np.log(T)])
    bounds = [
        (f_mean - f_std, f_mean + f_std),
        (np.log(e_mean / 100.0), np.log(f_std)),
        (np.log(f_std / 10.0), np.log(f_std * 10.0)),
        (np.log(dt), np.log(2.0 * T)),
    ]
    # Guard against inverted/degenerate bounds (e.g. e_mean > 100*f_std), which would make L-BFGS-B
    # raise; bail to the savgol fallback instead.
    for lo, hi in bounds:
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("gp_baseline: degenerate hyperparameter bounds")
    # Keep the start point strictly inside the box.
    p0 = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(p0, bounds)])

    def _build(p):
        mean, ln_jit, ln_sig, ln_rho = p
        kernel = terms.Matern32Term(sigma=np.exp(ln_sig), rho=np.exp(ln_rho))
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        gp.compute(ts, diag=es ** 2 + np.exp(2.0 * ln_jit) + e_inflate ** 2)
        return gp, mean

    def _nll(p):
        try:
            gp, mean = _build(p)
            return -gp.log_likelihood(fs - mean)
        except Exception:  # noqa: BLE001 -- treat any kernel/linalg failure as a bad point
            return 1e25

    res = minimize(_nll, p0, method="L-BFGS-B", bounds=bounds)
    gp, mean = _build(res.x)
    baseline_sorted = gp.predict(fs - mean, t=ts) + mean
    baseline = baseline_sorted[inv]
    if not (np.all(np.isfinite(baseline)) and np.all(baseline > 0)):
        raise ValueError("gp_baseline: non-finite or non-positive baseline")
    return baseline
