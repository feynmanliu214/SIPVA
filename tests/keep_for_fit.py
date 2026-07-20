#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit checks for the rho-consistency rejection of non-physical per-transit fits.

Covers fitting.compute_keep_for_fit (the single keep_for_fit mask) and the priors.py rho-width cap.
See docs/2026-06-10_reject_nonphysical_transits_plan.md.

Run with the project venv and src/core on the path:

    PYTHONPATH=src/core .venv/bin/python tests/keep_for_fit.py
"""
import math
import sys

import numpy as np

from fitting import compute_keep_for_fit
import priors


def _triple(v):
    """A [median, lerr, uerr] posterior row as the pipeline stores it."""
    return [v, 0.01, 0.01]


def _param_arrays(rhos, t14s):
    """Build a minimal param_arrays_0 with one row per (rho, t14). A None t14 marks a failed fit."""
    return {
        'rho': [_triple(r) for r in rhos],
        't14_1': [(_triple(t) if t is not None else None) for t in t14s],
    }


def _check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    return cond


def test_rho_band_and_lockstep():
    # rho_star = 3.7, F = 3 -> keep band [1.2333, 11.1]. Rows: normal, degenerate-low, normal,
    # too-high, NaN-rho.
    pa = _param_arrays(rhos=[3.4, 0.39, 3.6, 50.0, float('nan')],
                       t14s=[0.14, 0.42, 0.13, 0.10, 0.20])
    keep, rho_ok, factor = compute_keep_for_fit(pa, rho_star=3.7, reject_factor=3.0)
    ok = True
    ok &= _check("factor passed through", factor == 3.0)
    ok &= _check("normal rows kept", keep[0] and keep[2])
    ok &= _check("degenerate low-rho dropped", not keep[1])
    ok &= _check("too-high rho dropped", not keep[3])
    ok &= _check("non-finite rho dropped", not keep[4])
    ok &= _check("rho_consistent matches the rho check", rho_ok == [True, False, True, False, False])
    # Lockstep: deriving masked arrays from keep_idx keeps rows aligned and excludes the bad ones.
    keep_idx = [i for i, k in enumerate(keep) if k]
    masked_rho = [pa['rho'][i][0] for i in keep_idx]
    ok &= _check("masked rho rows are exactly the in-band ones", masked_rho == [3.4, 3.6])
    return ok


def test_nan_t14_with_physical_rho_is_kept():
    # A NaN-t14 row with finite, in-band rho must be KEPT (duration is not a rejection criterion);
    # a None-t14 row (failed fit) must be dropped.
    pa = _param_arrays(rhos=[3.5, 3.5], t14s=[float('nan'), None])
    keep, rho_ok, _ = compute_keep_for_fit(pa, rho_star=3.7, reject_factor=3.0)
    ok = True
    ok &= _check("NaN-t14 with physical rho kept", keep[0] is True)
    ok &= _check("None-t14 (failed fit) dropped", keep[1] is False)
    ok &= _check("rho_consistent True for both (rho in band)", rho_ok == [True, True])
    return ok


def test_first_row_flagged_reference_epoch():
    # If row 0 is rho-inconsistent, keep_idx[0] must point at the first KEPT transit (row 1),
    # so delta_t / b0_seed downstream anchor on a real transit, not an excluded one.
    pa = _param_arrays(rhos=[0.3, 3.4, 3.5], t14s=[0.5, 0.14, 0.13])
    keep, _, _ = compute_keep_for_fit(pa, rho_star=3.7, reject_factor=3.0)
    keep_idx = [i for i, k in enumerate(keep) if k]
    return _check("first kept index is row 1, not the flagged row 0", keep_idx[0] == 1)


def test_synthetic_path_is_noop():
    # rho_star=None (synthetic): no rho cut; only None-t14 rows are dropped.
    pa = _param_arrays(rhos=[0.01, 99.0, 3.0], t14s=[0.5, 0.1, None])
    keep, rho_ok, factor = compute_keep_for_fit(pa, rho_star=None)
    ok = True
    ok &= _check("no factor when synthetic", factor is None)
    ok &= _check("extreme rho kept when synthetic", keep[0] and keep[1])
    ok &= _check("None-t14 still dropped when synthetic", not keep[2])
    ok &= _check("rho_consistent all True when synthetic", rho_ok == [True, True, True])
    return ok


def test_rho_prior_width_cap():
    # The 12% cap narrows an over-wide catalog rho width; an already-tight one is untouched.
    class FakeKOI:
        def __init__(self, **f): self.__dict__.update(f)
        def __getattr__(self, n): return None

    base = dict(koi_period=12.0, koi_period_err1=1e-4, koi_period_err2=-1e-4,
                koi_impact=0.3, koi_impact_err1=0.05, koi_impact_err2=-0.05,
                koi_ror=0.04, koi_ror_err1=5e-4, koi_ror_err2=-5e-4,
                koi_longp=None, koi_eccen=None)
    orig = priors.get_koi
    ok = True
    try:
        # Widths use MEAN symmetrization of the +/- errors (2026-07 model corrections; was
        # quadrature). Wide catalog error (sigma 0.5*(0.20+0.20)=0.20 > 0.12*1.234=0.14808)
        # -> capped to 0.14808.
        priors.get_koi = lambda k: FakeKOI(koi_srho=1.234, koi_srho_err1=0.20, koi_srho_err2=-0.20, **base)
        spec = dict((n, (d, a, b)) for n, d, a, b in priors.koi_prior_spec(1.01))
        ok &= _check("wide rho width capped to 0.12*srho",
                     math.isclose(spec['rho'][2], 0.14808, rel_tol=1e-12))
        # Already-tight catalog error (sigma 0.007 < cap) -> unchanged.
        priors.get_koi = lambda k: FakeKOI(koi_srho=1.234, koi_srho_err1=0.007, koi_srho_err2=-0.007, **base)
        spec = dict((n, (d, a, b)) for n, d, a, b in priors.koi_prior_spec(1.01))
        ok &= _check("already-tight rho width untouched",
                     math.isclose(spec['rho'][2], 0.007, rel_tol=1e-12))
    finally:
        priors.get_koi = orig
    return ok


def main():
    tests = [test_rho_band_and_lockstep, test_nan_t14_with_physical_rho_is_kept,
             test_first_row_flagged_reference_epoch, test_synthetic_path_is_noop,
             test_rho_prior_width_cap]
    all_ok = True
    for t in tests:
        print(t.__name__)
        all_ok &= bool(t())
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
