#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu

Theory-derived limb-darkening priors for real Kepler KOIs.

The fit samples the Kipping (2013) ``q1``/``q2`` limb-darkening parameters. Rather than leaving
them on a flat [0, 1] prior, this module pins them to the values expected from the host star's
atmosphere: it computes the quadratic coefficients ``u1``/``u2`` (with uncertainties) from the
star's Teff/logg/[Fe/H] using PyLDTk (Parviainen & Aigrain 2015) in the Kepler bandpass, then
transforms to ``q1``/``q2`` and propagates the uncertainties. The result is a per-coefficient
Normal prior ``(mu, sigma)``.

``koi_ld_prior`` returns ``None`` (so the caller keeps the uniform [0, 1] prior) whenever a real
uncertainty cannot be derived: PyLDTk is not installed, the model download fails, the catalog is
missing any of Teff/logg/[Fe/H] or their error bars, or the propagated sigma is non-finite/zero.

Results are cached to ``data/ld_cache/koi_<n>.json`` so the (slow, network-bound) PyLDTk model
download runs at most once per star.
"""

import json
import os
from pathlib import Path

import numpy as np

from data import get_koi


# Kepler bandpass in nm, used only for the BoxcarFilter fallback if PyLDTk's tabulated Kepler
# response is unavailable. The Kepler response is ~broad over this range.
_KEPLER_NM = (420.0, 900.0)


def _cache_dir():
    """data/ld_cache/, resolved relative to the repo root (robust to the caller's cwd)."""
    d = Path(__file__).resolve().parents[2] / 'data' / 'ld_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sym_err(err1, err2):
    """Symmetrized 1-sigma from the catalog's asymmetric +/- errors. Returns None if neither
    bound is a usable positive number (treated as 'no uncertainty available')."""
    vals = [abs(e) for e in (err1, err2) if e is not None and np.isfinite(e) and e != 0.0]
    if not vals:
        return None
    return 0.5 * sum(vals) if len(vals) == 2 else vals[0]


def u_to_q(u1, u2, su1, su2):
    """Quadratic LD coefficients (u1, u2) -> Kipping (2013) (q1, q2), with linearized error
    propagation (treating u1, u2 as independent):

        q1 = (u1 + u2)^2,                q2 = u1 / (2 (u1 + u2))

    Returns (q1, sig_q1, q2, sig_q2), or None if the transform is degenerate (u1 + u2 ~ 0) or any
    output is non-finite/non-positive sigma.
    """
    s = u1 + u2
    if not np.isfinite(s) or abs(s) < 1e-8:
        return None
    q1 = s ** 2
    q2 = u1 / (2.0 * s)
    # d q1/d u1 = d q1/d u2 = 2 s  ->  sig_q1 = 2|s| sqrt(su1^2 + su2^2)
    sig_q1 = 2.0 * abs(s) * np.sqrt(su1 ** 2 + su2 ** 2)
    # q2 = u1 / (2 s):  d q2/d u1 = u2 / (2 s^2),  d q2/d u2 = -u1 / (2 s^2)
    sig_q2 = np.sqrt((u2 * su1) ** 2 + (u1 * su2) ** 2) / (2.0 * s ** 2)
    if not (np.isfinite(q1) and np.isfinite(q2) and np.isfinite(sig_q1) and np.isfinite(sig_q2)):
        return None
    if sig_q1 <= 0.0 or sig_q2 <= 0.0:
        return None
    return float(q1), float(sig_q1), float(q2), float(sig_q2)


def _kepler_filter():
    """PyLDTk's tabulated Kepler response if available, else a Boxcar approximation."""
    try:
        from ldtk.filters import kepler
        return kepler
    except Exception:
        from ldtk import BoxcarFilter
        return BoxcarFilter('Kepler', *_KEPLER_NM)


def _compute_uq_from_stellar(teff, e_teff, logg, e_logg, feh, e_feh,
                             nsamples=1500, n_mc=12000):
    """Run PyLDTk for one star and return quadratic (u1, u2, su1, su2). Raises on any PyLDTk
    failure (import, download, fit) -- the caller decides whether to cache/fall back."""
    from ldtk import LDPSetCreator
    sc = LDPSetCreator(teff=(teff, e_teff), logg=(logg, e_logg), z=(feh, e_feh),
                       filters=[_kepler_filter()])
    ps = sc.create_profiles(nsamples=nsamples)
    coeffs, errors = ps.coeffs_qd(do_mc=True, n_mc_samples=n_mc)
    u1, u2 = float(coeffs[0][0]), float(coeffs[0][1])
    su1, su2 = float(errors[0][0]), float(errors[0][1])
    return u1, u2, su1, su2


def koi_ld_prior(koi_number, force=False):
    """Normal-prior (mu, sigma) on Kipping q1 and q2 for a real KOI, derived from the host star's
    Teff/logg/[Fe/H] via PyLDTk. Returns (q1_mu, q1_sig, q2_mu, q2_sig), or None to signal the
    caller should keep the uniform [0, 1] prior.

    Cached to data/ld_cache/koi_<n>.json. Deterministic "no prior" outcomes (missing stellar
    params) are cached; transient PyLDTk failures (import/download) are NOT cached, so a later run
    can retry. Pass force=True to recompute and overwrite the cache.
    """
    cache_file = _cache_dir() / f"koi_{koi_number}.json"
    if cache_file.exists() and not force:
        try:
            data = json.loads(cache_file.read_text())
            qp = data.get('q_prior')
            return tuple(qp) if qp is not None else None
        except (ValueError, OSError):
            pass  # corrupt cache -> recompute

    koi = get_koi(koi_number)
    teff, logg, feh = koi.koi_steff, koi.koi_slogg, koi.koi_smet
    e_teff = _sym_err(koi.koi_steff_err1, koi.koi_steff_err2)
    e_logg = _sym_err(koi.koi_slogg_err1, koi.koi_slogg_err2)
    e_feh = _sym_err(koi.koi_smet_err1, koi.koi_smet_err2)

    # Require all three stellar params AND their error bars: the prior std comes only from
    # propagating the stellar-parameter uncertainties (no fixed-width fallback by design).
    if any(v is None or not np.isfinite(float(v))
           for v in (teff, logg, feh, e_teff, e_logg, e_feh)):
        cache_file.write_text(json.dumps({
            'koi': koi_number, 'q_prior': None,
            'reason': 'missing stellar params or error bars'}, indent=2))
        return None

    teff, logg, feh = float(teff), float(logg), float(feh)
    e_teff, e_logg, e_feh = float(e_teff), float(e_logg), float(e_feh)

    try:
        u1, u2, su1, su2 = _compute_uq_from_stellar(teff, e_teff, logg, e_logg, feh, e_feh)
    except Exception as exc:  # PyLDTk missing / download failed -> fall back, do NOT cache
        print(f"[limb_darkening] PyLDTk unavailable for KOI {koi_number} "
              f"({type(exc).__name__}: {exc}); leaving q1/q2 on the uniform prior.")
        return None

    q = u_to_q(u1, u2, su1, su2)
    cache_file.write_text(json.dumps({
        'koi': koi_number,
        'stellar': {'teff': teff, 'e_teff': e_teff, 'logg': logg, 'e_logg': e_logg,
                    'feh': feh, 'e_feh': e_feh},
        'u': {'u1': u1, 'u2': u2, 'su1': su1, 'su2': su2},
        'q_prior': list(q) if q is not None else None,
    }, indent=2))
    return q
