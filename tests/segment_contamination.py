#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit checks for the segment-contamination fix
(docs/2026-06-12_fix_segment_contamination_plan.md):

  - Component 1: sibling-cadence masking + O-C provenance classification
  - Component 2: near-center coverage requirement
  - Component 3: segment baseline-quality guard
  - Component 4: NaN-safe derived-duration posterior estimate

Run with the project venv and src/core on the path:

    PYTHONPATH=src/core .venv/bin/python tests/segment_contamination.py
"""
import sys

import numpy as np

from data import (TransitEphemeris, segment_coverage_ok, segment_baseline_ok)
from fitting import param_posterior_est
from pipeline import _mask_sibling_cadences


def _check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    return bool(cond)


# --- Component 2: coverage --------------------------------------------------

def test_coverage():
    t14 = 0.2  # days; h = 0.5 * 0.2 = 0.1
    ok = True
    # Fully sampled transit, points on both sides, >= 3 within +/-h -> kept.
    full = 0.0 + np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
    ok &= _check("full transit kept", segment_coverage_ok(full, 0.0, t14)[0] is True)
    # One-sided (only after center) -> dropped (no before point).
    one_sided = 0.0 + np.array([0.02, 0.04, 0.06, 0.08])
    keep, reason = segment_coverage_ok(one_sided, 0.0, t14)
    ok &= _check("one-sided dropped", keep is False and reason == "no_coverage")
    # Two points straddling the center but < min_in_transit (3) -> dropped.
    n2 = 0.0 + np.array([-0.05, 0.05])
    ok &= _check("n=2 straddling dropped", segment_coverage_ok(n2, 0.0, t14)[0] is False)
    return ok


# --- Component 3: baseline guard --------------------------------------------

def test_baseline_guard():
    t14 = 0.2          # OOT = |dt| > 0.12
    ferr = 0.001
    dt = np.linspace(-0.2, 0.2, 21)
    seg_t = dt.copy()  # center 0
    fe = np.full_like(dt, ferr)
    ok = True

    # Flat baseline at 1.0 -> passes.
    flat = np.ones_like(dt)
    ok &= _check("flat baseline passes", segment_baseline_ok(seg_t, flat, fe, 0.0, t14)[0] is True)

    # 1% ramp on the whole segment -> OOT median far from 1 -> median test fails.
    ramp = np.full_like(dt, 1.01)
    keep, reason, _ = segment_baseline_ok(seg_t, ramp, fe, 0.0, t14)
    ok &= _check("OOT-median ramp dropped", keep is False and reason == "bad_baseline")

    # A run of 3 consecutive OOT cadences dipping below 1 - 5*ferr -> run test fails,
    # while the OOT median stays ~1 (so it is the run test, not the median test, that fires).
    dip = np.ones_like(dt)
    run_idx = np.where(np.isclose(dt, 0.14) | np.isclose(dt, 0.16) | np.isclose(dt, 0.18))[0]
    dip[run_idx] = 0.99   # below 1 - 5*0.001 = 0.995
    keep, reason, _ = segment_baseline_ok(seg_t, dip, fe, 0.0, t14)
    ok &= _check("consecutive OOT dip dropped", keep is False and reason == "bad_baseline")

    # Too few OOT points -> guard skipped (kept), reason None.
    near = np.array([-0.05, -0.02, 0.0, 0.03, 0.06])  # all within |dt|<=0.12 -> 0 OOT
    keep, reason, detail = segment_baseline_ok(near, np.ones(5), np.full(5, ferr), 0.0, t14)
    ok &= _check("too-few-OOT guard skipped", keep is True and reason is None and "skipped" in detail)
    return ok


# --- Component 1: sibling masking + provenance ------------------------------

def test_oc_provenance():
    ok = True
    lin = TransitEphemeris(0.0, 10.0, source="linear")
    ok &= _check("linear -> 'linear'", lin.oc_provenance(3) == "linear")

    hol = TransitEphemeris(0.0, 10.0, oc_epochs=[5, 10], oc_minutes=[1.0, 2.0],
                           source="holczer2016")
    ok &= _check("measured epoch -> holczer_measured", hol.oc_provenance(5) == "holczer_measured")
    ok &= _check("between -> holczer_interpolated", hol.oc_provenance(7) == "holczer_interpolated")
    ok &= _check("below range -> outside_holczer_range", hol.oc_provenance(2) == "outside_holczer_range")
    ok &= _check("above range -> outside_holczer_range", hol.oc_provenance(99) == "outside_holczer_range")

    fit = TransitEphemeris(0.0, 10.0, oc_epochs=[5, 10], oc_minutes=[1.0, 2.0],
                           source="pytransit_fit")
    ok &= _check("pytransit_fit -> 'pytransit_fit'", fit.oc_provenance(7) == "pytransit_fit")
    return ok


def test_sibling_masking():
    # Target: t0=0, P=10 (centers ..., 10, ...), T14_target=0.2 -> window +/-0.2.
    # Sibling: t0=0.1, P=10 (center 10.1), T14_sib=0.1 -> mask +/-0.075 around 10.1.
    target = TransitEphemeris(0.0, 10.0, source="linear")
    sib_eph = TransitEphemeris(0.1, 10.0, source="holczer2016")
    siblings = [{"koi": "999.01", "eph": sib_eph, "t14_days": 0.1, "source": "holczer2016"}]

    t = np.array([9.85, 9.90, 10.00, 10.05, 10.10, 10.15, 10.30])
    f = np.arange(len(t), dtype=float)        # distinct values to verify lockstep removal
    e = np.full(len(t), 0.001)
    times, fluxs, errs = [t.copy()], [f.copy()], [e.copy()]

    audit = _mask_sibling_cadences("999.target", target, 0.2, times, fluxs, errs, siblings)

    ok = True
    # Sibling window 10.025..10.175 removes 10.05, 10.10, 10.15 (indices 3,4,5).
    expected_keep = np.array([9.85, 9.90, 10.00, 10.30])
    ok &= _check("3 sibling cadences removed", audit["n_cadences_sibling_masked"] == 3)
    ok &= _check("times pruned to clean cadences", np.allclose(times[0], expected_keep))
    ok &= _check("flux stays in lockstep", np.allclose(fluxs[0], np.array([0., 1., 2., 6.])))
    ok &= _check("errs stay in lockstep", len(errs[0]) == 4)
    ok &= _check("one target epoch affected", audit["epochs_sibling_affected"] == 1)
    # 6 in-window points, 3 removed -> 50% > 30% -> epoch flagged for review.
    ok &= _check(">30%-loss epoch on review list", audit["sibling_review_epochs"] == [1])
    return ok


# --- Component 4: NaN-safe derived durations --------------------------------

class _Var:
    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=float)

    @property
    def values(self):
        return self._a


class _DF:
    def __init__(self, derived=None, posterior=None):
        self.derived_parameters = derived or {}
        self.posterior = posterior or {}


def test_nan_safe_duration():
    ok = True
    # All-finite samples: equals np.percentile, returns a triple.
    clean = np.linspace(0.1, 0.2, 1001)
    df = _DF(derived={'t14_1': _Var(clean)})
    res = param_posterior_est(df, 't14_1', 'derived_parameters')
    q16, q50, q84 = np.percentile(clean, [16, 50, 84])
    ok &= _check("clean samples -> finite triple",
                 res is not None and np.isclose(res[0], q50))

    # 10% non-transiting (NaN) tail: still measurable -> median over the transiting samples.
    tail = clean.copy()
    tail[:100] = np.nan
    df = _DF(derived={'t14_1': _Var(tail)})
    res = param_posterior_est(df, 't14_1', 'derived_parameters')
    ok &= _check("10% NaN -> finite (not None)",
                 res is not None and np.isfinite(res[0])
                 and np.isclose(res[0], np.nanpercentile(tail, 50)))

    # >50% non-transiting -> genuine non-measurement -> None.
    mostly = clean.copy()
    mostly[:600] = np.nan
    df = _DF(derived={'t14_1': _Var(mostly)})
    ok &= _check(">50% NaN -> None", param_posterior_est(df, 't14_1', 'derived_parameters') is None)
    return ok


def main():
    tests = [test_coverage, test_baseline_guard, test_oc_provenance, test_sibling_masking,
             test_nan_safe_duration]
    all_ok = True
    for t in tests:
        print(t.__name__)
        all_ok &= bool(t())
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
