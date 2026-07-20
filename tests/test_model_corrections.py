#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation suite for the four real-Kepler model corrections (2026-07-19 plan):

  SS1  catalog-only SIPVA priors (T1-T3)
  SS2  db/dt boundary removal + broad safety support (T4-T6)
  SS3  consistent eccentric transit geometry (T7-T12)
  SS4  finite-exposure integration (T13-T18)

Standalone script (no pytest), matching the repo's tests/ convention. No network: the
catalog is faked and PyLDTk is monkeypatched. Run with:

    PYTHONPATH=src/core .venv/bin/python tests/test_model_corrections.py
"""
import hashlib
import inspect
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "core"))

import data
import fitting
import model
import priors
from data import (LC_EXPTIME_D, SC_EXPTIME_D, TransitEphemeris, _choose_cadence_subset,
                  exposure_config, get_transit_arrays, product_exptimes,
                  select_cadence_per_transit)
from fitting import (DB_DT_SUPPORT, _transit_center_and_depth, build_de_bounds, log_prior,
                     logprob_dbdt, neg_log_likelihood)
from model import InvalidGeometryError, build_transit_models, evaluate_transit_flux
from pipeline import align_to_selection
from priors import sipva_prior_spec
from pytransit import QuadraticModel, TransitAnalysis
from pytransit.orbits import as_from_rhop, d_from_pkaiews, i_from_ba, i_from_baew
from ta_eccentric import EccentricTransitAnalysis

SRC_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "core")


def _check(msg, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {msg}")
    return bool(cond)


# --- Fixtures ---------------------------------------------------------------

class FakeKOI:
    def __init__(self, **f): self.__dict__.update(f)
    def __getattr__(self, n): return None


FAKE_FIELDS = dict(koi_srho=1.4, koi_srho_err1=0.10, koi_srho_err2=-0.12,
                   koi_period=10.0, koi_period_err1=1e-5, koi_period_err2=-1e-5,
                   koi_impact=0.3, koi_impact_err1=0.05, koi_impact_err2=-0.05,
                   koi_ror=0.08, koi_ror_err1=1e-3, koi_ror_err2=-1e-3,
                   koi_eccen=0.1, koi_longp=90.0)
FAKE_LD = (0.45, 0.03, 0.35, 0.04)


class _fake_catalog:
    """Monkeypatch priors.get_koi / priors.koi_ld_prior inside a with-block (no network)."""
    def __init__(self, ld=FAKE_LD, fields=None):
        self.ld, self.fields = ld, (fields or FAKE_FIELDS)
    def __enter__(self):
        self._gk, self._ld = priors.get_koi, priors.koi_ld_prior
        priors.get_koi = lambda k: FakeKOI(**self.fields)
        priors.koi_ld_prior = lambda k: self.ld
        return self
    def __exit__(self, *a):
        priors.get_koi, priors.koi_ld_prior = self._gk, self._ld


# Canonical transit pv for the geometry/exposure tests:
# [rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2]
PV0 = np.array([1.4, 0.0, 10.0, 0.3, 0.01, 0.0, 0.0, 0.3, 0.3])


def _pv(**kw):
    names = ['rho', 'tc', 'p', 'b', 'k2', 'secw', 'sesw', 'q1', 'q2']
    pv = PV0.copy()
    for k, v in kw.items():
        pv[names.index(k)] = v
    return pv


def _spec_arrays(spec):
    kinds = [d for _n, d, _a, _b in spec]
    means = np.array([a for _n, _d, a, _b in spec], dtype=float)
    stds = np.array([b for _n, _d, _a, b in spec], dtype=float)
    return kinds, means, stds


THETA0 = np.array([1.4, 0.0, 10.0, 0.0064, 0.22, 0.22, 0.45, 0.35, 0.5, -0.05])


# --- SS1: catalog-only SIPVA priors ------------------------------------------

def test_T1_prior_independent_of_individual_fits():
    ok = True
    with _fake_catalog():
        spec1 = sipva_prior_spec(999.99)
        kinds, means, stds = _spec_arrays(spec1)
        lp1 = log_prior(THETA0, means, stds, kinds=kinds)

        # "Change or remove the saved individual-fit uncertainties": the construction takes
        # no such input at all, so mutating a stand-in individual-fit result dict cannot
        # change the spec or the log prior at a fixed parameter vector.
        fake_fit_results = {'rho': np.random.rand(30, 3), 'b_1': np.random.rand(30, 3)}
        fake_fit_results['rho'] *= 100.0
        del fake_fit_results['b_1']
        spec2 = sipva_prior_spec(999.99)
        kinds2, means2, stds2 = _spec_arrays(spec2)
        lp2 = log_prior(THETA0, means2, stds2, kinds=kinds2)

    ok &= _check("spec identical on rebuild", spec1 == spec2)
    ok &= _check("log prior bit-identical", lp1 == lp2 and np.isfinite(lp1))
    ok &= _check("b_0 entry is UP(0,1)", spec1[8] == ('b_0', 'UP', 0.0, 1.0))
    ok &= _check("db_dt entry is NP(0,0.2)", spec1[9] == ('db_dt', 'NP', 0.0, 0.2))
    ok &= _check("tc_1 (Delta t_e) stays NP(0,1e-8)", spec1[1] == ('tc_1', 'NP', 0.0, 1e-8))
    ok &= _check("rho width uses mean symmetrization (0.11 < 0.12*srho cap)",
                 math.isclose(spec1[0][3], 0.11, rel_tol=1e-12))
    # Hand-computed prior value: sum of the NP log-densities + 0 for the U(0,1) b_0.
    expect = 0.0
    for i, (nm, kd, a, b) in enumerate(spec1):
        if kd == 'NP':
            expect += -0.5 * ((THETA0[i] - a) / b) ** 2 - np.log(np.sqrt(2 * np.pi) * b)
    ok &= _check("log prior equals hand-computed catalog density", np.isclose(lp1, expect, rtol=0, atol=1e-12))
    return ok


def test_T2_b0_uniform_and_q_fallback():
    ok = True
    with _fake_catalog():
        kinds, means, stds = _spec_arrays(sipva_prior_spec(999.99))
    lo_ = log_prior(np.r_[THETA0[:8], 0.2, THETA0[9]], means, stds, kinds=kinds)
    hi_ = log_prior(np.r_[THETA0[:8], 0.8, THETA0[9]], means, stds, kinds=kinds)
    ok &= _check("b_0 prior flat inside [0,1]", np.isclose(lo_, hi_, rtol=0, atol=1e-12))
    for bad in (-0.01, 1.01):
        lp = log_prior(np.r_[THETA0[:8], bad, THETA0[9]], means, stds, kinds=kinds)
        ok &= _check(f"b_0={bad} rejected by prior", lp == -np.inf)

    with _fake_catalog(ld=None):   # PyLDTk unavailable -> q1/q2 fall back to UP(0,1)
        spec = sipva_prior_spec(999.99)
        kinds, means, stds = _spec_arrays(spec)
    ok &= _check("q1/q2 fallback entries are UP(0,1)",
                 spec[6][1] == 'UP' and spec[7][1] == 'UP')
    a_ = log_prior(THETA0, means, stds, kinds=kinds)
    theta_q = THETA0.copy(); theta_q[6], theta_q[7] = 0.9, 0.1
    b_ = log_prior(theta_q, means, stds, kinds=kinds)
    ok &= _check("fallback q prior flat inside [0,1]", np.isclose(a_, b_, rtol=0, atol=1e-12))
    # DE bounds and walker-init inputs derive from the same spec (never from fits):
    bounds = build_de_bounds(means, stds, kinds)
    ok &= _check("b_0 DE box is the full [0,1]", bounds[8] == (0.0, 1.0))
    ok &= _check("db_dt DE box is the 3-sigma prior box",
                 np.allclose(bounds[9], (-0.6, 0.6)))
    ok &= _check("secw/sesw DE boxes reach 3-sigma (no 1e-4 relic)",
                 bounds[4][1] > 0.01 and bounds[5][1] > 0.01)
    return ok


def test_T3_combine_errors_convention():
    ok = True
    ok &= _check("mean of two errors", math.isclose(priors._combine_errors(0.10, -0.12), 0.11))
    ok &= _check("single error passes through", priors._combine_errors(0.10, None) == 0.10)
    ok &= _check("no errors -> 0.0", priors._combine_errors(None, None) == 0.0)
    ok &= _check("zeros treated as absent", priors._combine_errors(0.0, 0.0) == 0.0)
    return ok


# --- SS2: db/dt boundary removal ----------------------------------------------

def _two_segment_data(b0=0.5, db_dt=-0.10, delta_t_yr=(0.0, 2.0)):
    theta = np.r_[PV0[[0, 1, 2]], PV0[4], PV0[5], PV0[6], PV0[7], PV0[8], b0, db_dt]
    # theta order: rho, tc, p, k2, secw, sesw, q1, q2, b_0, db_dt
    t_out = [np.linspace(-0.12, 0.12, 40), np.linspace(-0.12, 0.12, 40)]
    f_out = []
    for dt in delta_t_yr:
        b_j = b0 + db_dt * dt
        f_out.append(evaluate_transit_flux(_pv(b=min(max(b_j, 0.0), 0.999)), t_out[0]))
    ferr = [2e-4, 2e-4]
    means = np.zeros(10); stds = np.ones(10)
    return theta, np.array(delta_t_yr), f_out, t_out, ferr, means, stds


def test_T4_valid_negative_slope_is_finite():
    theta, dts, f_out, t_out, ferr, means, stds = _two_segment_data(b0=0.5, db_dt=-0.10)
    lp = logprob_dbdt(theta, dts, f_out, t_out, ferr, means, stds)
    ok = _check("db_dt=-0.10 with all b_j in [0,1] has finite posterior density",
                np.isfinite(lp))
    nll = neg_log_likelihood(theta, dts, f_out, t_out, ferr, means, stds)
    ok &= _check("neg_log_likelihood finite at the same point", np.isfinite(nll))
    return ok


def test_T5_invalid_slopes_rejected():
    ok = True
    theta, dts, f_out, t_out, ferr, means, stds = _two_segment_data(b0=0.5, db_dt=0.30)
    ok &= _check("slope pushing b_j > 1 rejected",
                 logprob_dbdt(theta, dts, f_out, t_out, ferr, means, stds) == -np.inf)
    theta, dts, f_out, t_out, ferr, means, stds = _two_segment_data(
        b0=0.5, db_dt=-1.5, delta_t_yr=(0.0, 0.1))
    ok &= _check("|db_dt| > 1 rejected by the broad safety support",
                 logprob_dbdt(theta, dts, f_out, t_out, ferr, means, stds) == -np.inf)
    ok &= _check("safety support is +/-1.0 exactly", DB_DT_SUPPORT == 1.0)
    return ok


def test_T6_no_old_boundary_literal():
    hits = []
    for fn in sorted(os.listdir(SRC_CORE)):
        if not fn.endswith(".py"):
            continue
        text = open(os.path.join(SRC_CORE, fn)).read()
        if "0.075" in text:
            hits.append(fn)
    return _check(f"no 0.075 literal under src/core (hits: {hits})", not hits)


# --- SS3: eccentric geometry ----------------------------------------------------

def test_T7_e_zero_reproduces_circular():
    ok = True
    for b in (0.0, 0.3, 0.7, 0.95):
        for a in (5.0, 15.0, 40.0):
            ok &= (i_from_baew(b, a, 0.0, 0.0) == i_from_ba(b, a))
    ok = _check("i_from_baew(e=0) bit-identical to i_from_ba over grid", ok)

    t = np.linspace(-0.15, 0.15, 400)
    f_new = evaluate_transit_flux(_pv(), t)
    # Manual circular reference (the pre-change construction at e = 0).
    aor = as_from_rhop(PV0[0], PV0[2])
    qm = QuadraticModel(); qm.set_data(t)
    u1 = 2 * np.sqrt(PV0[7]) * PV0[8]; u2 = np.sqrt(PV0[7]) * (1 - 2 * PV0[8])
    f_ref = qm.evaluate(np.sqrt(PV0[4]), [u1, u2], PV0[1], PV0[2], aor,
                        i_from_ba(PV0[3], aor), 0.0, 0.0)
    ok &= _check("flux at e=0 matches circular reference to < 1e-12",
                 float(np.max(np.abs(f_new - f_ref))) < 1e-12)
    return ok


def test_T8_geometry_roundtrip():
    worst = 0.0
    for b in (0.05, 0.3, 0.6, 0.9):
        for a in (5.0, 15.0, 40.0):
            for e in (0.0, 0.2, 0.5):
                for w in (-2.5, -np.pi / 2, 0.0, np.pi / 2, 2.0):
                    arg = (b / a) * (1 + e * np.sin(w)) / (1 - e ** 2)
                    if abs(arg) >= 1:
                        continue
                    i = i_from_baew(b, a, e, w)
                    b_back = a * np.cos(i) * (1 - e ** 2) / (1 + e * np.sin(w))
                    worst = max(worst, abs(b_back - b))
    return _check(f"(b,a,e,w)->i->b roundtrip, worst |db|={worst:.2e} < 1e-10", worst < 1e-10)


def test_T9_duration_direction():
    p, k, a, b = 10.0, 0.1, 15.0, 0.3
    def t14(e, w):
        i = i_from_baew(b, a, e, w)
        return d_from_pkaiews(p, k, a, i, e, w, 1, kind=14)
    circ = t14(0.0, 0.0)
    peri = t14(0.3, np.pi / 2)     # transit near periastron: faster -> shorter
    apo = t14(0.3, -np.pi / 2)     # transit near apastron: slower -> longer
    ok = _check(f"T14(e=0.3, w=+90deg)={peri:.4f} < T14(0)={circ:.4f}", peri < circ)
    ok &= _check(f"T14(e=0.3, w=-90deg)={apo:.4f} > T14(0)={circ:.4f}", apo > circ)
    return ok


def _numeric_t14(pv, span=0.25, step=1e-5):
    t = np.arange(-span, span, step)
    f = evaluate_transit_flux(pv, t)
    in_tr = np.where(f < 1.0 - 1e-12)[0]
    return (t[in_tr[-1]] - t[in_tr[0]]) + step


def test_T10_flux_duration_consistency():
    ok = True
    tol = 5.0 / 86400.0                       # 5 s (Taylor-z vs analytic approximation)
    aor = as_from_rhop(PV0[0], PV0[2])
    cases = [(0.0, 0.0, 0.0, 0.0), (0.3, np.pi / 2, 0.0, np.sqrt(0.3)),
             (0.3, -np.pi / 2, 0.0, -np.sqrt(0.3))]
    t14n = {}
    for e, w, secw, sesw in cases:
        pv = _pv(secw=secw, sesw=sesw)
        i = i_from_baew(PV0[3], aor, e, w)
        ana = d_from_pkaiews(PV0[2], np.sqrt(PV0[4]), aor, i, e, w, 1, kind=14)
        num = _numeric_t14(pv)
        t14n[(e, w)] = num
        ok &= _check(f"e={e}, w={w:+.2f}: |T14_num - T14_ana| = "
                     f"{abs(num - ana) * 86400:.2f} s < 5 s", abs(num - ana) < tol)
    shift = t14n[(0.3, np.pi / 2)] - t14n[(0.0, 0.0)]
    ok &= _check(f"e-induced shift {shift * 86400:.0f} s is large vs tolerance (>10x)",
                 abs(shift) > 10 * tol and shift < 0)
    return ok


def test_T11_no_hardcoded_circular_durations_and_pinned_upstream():
    ok = True
    pat = re.compile(r"d_from_pkaiews\([^)]*\b0\.\s*,\s*0\.")
    hits = []
    for fn in sorted(os.listdir(SRC_CORE)):
        if fn.endswith(".py") and pat.search(open(os.path.join(SRC_CORE, fn)).read()):
            hits.append(fn)
    ok &= _check(f"no d_from_pkaiews(..., 0., 0., ...) in src/core (hits: {hits})", not hits)

    # Pin the upstream 2.7.1 methods our subclass copies: a PyTransit upgrade must fail here.
    pins = {'transit_model': '9185b6fdd1fe8204b0e2368a0a58886a1078bb9ed0d31800b1c8ddb9d03120d8',
            'posterior_samples': 'a92f578f800994d9e94fc0c266d1bb962a31b2a72ec991e899f895ebf13b79d8'}
    for m, want in pins.items():
        got = hashlib.sha256(inspect.getsource(getattr(TransitAnalysis, m)).encode()).hexdigest()
        ok &= _check(f"upstream TransitAnalysis.{m} source unchanged (2.7.1 pin)", got == want)
    ok &= _check("EccentricTransitAnalysis overrides both methods",
                 EccentricTransitAnalysis.transit_model is not TransitAnalysis.transit_model
                 and EccentricTransitAnalysis.posterior_samples is not TransitAnalysis.posterior_samples)

    # Functional: the subclass flux model responds to eccentricity consistently (differs from
    # the mixed-geometry base for e != 0; identical at e = 0).
    rng = np.random.default_rng(1)
    t = np.linspace(-0.12, 0.12, 200)
    f_data = 1.0 + 1e-4 * rng.standard_normal(t.size)
    ta = TransitAnalysis(name="t11_base", passbands='Kepler', times=t, fluxes=f_data)
    eta = EccentricTransitAnalysis(name="t11_ecc", passbands='Kepler', times=t, fluxes=f_data)
    pv = ta.ps.sample_from_prior(1)[0]
    for nm, val in (('rho', 1.4), ('tc_1', 0.0), ('p_1', 10.0), ('b_1', 0.3), ('k2_1', 0.01),
                    ('secw_1', 0.0), ('sesw_1', 0.0)):
        pv[ta.ps.names.index(nm)] = val
    f_base0 = np.asarray(ta.transit_model(pv)); f_ecc0 = np.asarray(eta.transit_model(pv))
    ok &= _check("subclass == base at e = 0", float(np.max(np.abs(f_base0 - f_ecc0))) < 1e-14)
    pv[ta.ps.names.index('sesw_1')] = np.sqrt(0.3)
    f_base = np.asarray(ta.transit_model(pv)); f_ecc = np.asarray(eta.transit_model(pv))
    ok &= _check("subclass != base at e = 0.3 (mid-transit b now eccentric-consistent)",
                 float(np.max(np.abs(f_base - f_ecc))) > 1e-6)
    return ok


def test_T12_synthetic_identity_drift_bound():
    # Synthetic runs sample secw/sesw at the ~1e-5 level; the geometry change may drift the
    # flux only below 1e-9 there (Q3-A acceptance).
    t = np.linspace(-0.15, 0.15, 500)
    pv = _pv(secw=1e-5, sesw=1e-5)
    f_new = evaluate_transit_flux(pv, t)
    aor = as_from_rhop(pv[0], pv[2])
    ecc = pv[5] ** 2 + pv[6] ** 2
    w = np.arctan2(pv[6], pv[5])
    u1 = 2 * np.sqrt(pv[7]) * pv[8]; u2 = np.sqrt(pv[7]) * (1 - 2 * pv[8])
    qm = QuadraticModel(); qm.set_data(t)
    f_old = qm.evaluate(np.sqrt(pv[4]), [u1, u2], pv[1], pv[2], aor,
                        i_from_ba(pv[3], aor), ecc, w)   # old mixed-geometry behavior
    drift = float(np.max(np.abs(f_new - f_old)))
    ok = _check(f"drift at secw=sesw=1e-5 is {drift:.2e} < 1e-9", drift < 1e-9)
    ok &= _check("invalid geometry raises InvalidGeometryError (ArithmeticError subclass)",
                 issubclass(InvalidGeometryError, ArithmeticError))
    try:
        evaluate_transit_flux(_pv(secw=0.8, sesw=0.7), t)   # e = 1.13 >= 1
        ok &= _check("e >= 1 rejected", False)
    except ArithmeticError:
        ok &= _check("e >= 1 rejected", True)
    return ok


# --- SS4: finite-exposure integration ---------------------------------------------

def test_T13_sc_unchanged():
    t = np.linspace(-0.15, 0.15, 300)
    f_inst = evaluate_transit_flux(_pv(), t)
    f_sc = evaluate_transit_flux(_pv(), t, exptime=SC_EXPTIME_D, nsamples=1)
    ok = _check("nsamples=1 output exactly equals instantaneous", np.array_equal(f_inst, f_sc))
    m_none = build_transit_models([t])[0]
    m_sc = build_transit_models([t], exp_list=[(1, SC_EXPTIME_D)])[0]
    f1 = evaluate_transit_flux(_pv(), t, model=m_none)
    f2 = evaluate_transit_flux(_pv(), t, model=m_sc)
    ok &= _check("build_transit_models SC config identical to no-exposure", np.array_equal(f1, f2))
    return ok


def test_T14_lc_differs_at_ingress():
    t = np.linspace(-0.15, 0.15, 2000)
    f_inst = evaluate_transit_flux(_pv(), t)
    f_lc = evaluate_transit_flux(_pv(), t, exptime=LC_EXPTIME_D, nsamples=15)
    d = np.abs(f_lc - f_inst)
    imax = int(np.argmax(d))
    mid = int(np.argmin(np.abs(t)))
    ok = _check(f"LC integration changes the model (max |df| = {d[imax]:.2e} > 1e-4)",
                d[imax] > 1e-4)
    in_tr = f_inst < 1 - 1e-8
    t1, t4 = t[in_tr][0], t[in_tr][-1]
    near_contact = min(abs(t[imax] - t1), abs(t[imax] - t4)) < 0.02
    ok &= _check("largest deviation sits at ingress/egress", near_contact)
    ok &= _check(f"mid-transit deviation ({d[mid]:.2e}) << ingress deviation",
                 d[mid] < 0.2 * d[imax])
    return ok


def test_T15_brute_force_supersampling():
    t = np.linspace(-0.15, 0.15, 700)
    f_lc = evaluate_transit_flux(_pv(), t, exptime=LC_EXPTIME_D, nsamples=15)
    # Independent 99-point mean over each exposure window through the INSTANTANEOUS path
    # (PyTransit sample positions: exptime * ((s - 0.5)/n - 0.5), s = 1..n).
    n_ref = 99
    acc = np.zeros_like(t)
    for s in range(1, n_ref + 1):
        off = LC_EXPTIME_D * ((s - 0.5) / n_ref - 0.5)
        acc += evaluate_transit_flux(_pv(), t + off)
    f_ref = acc / n_ref
    d = float(np.max(np.abs(f_lc - f_ref)))
    f_inst = evaluate_transit_flux(_pv(), t)
    effect = float(np.max(np.abs(f_lc - f_inst)))
    ok = _check(f"15-pt vs 99-pt brute force: max |df| = {d:.2e} < 1e-5", d < 1e-5)
    ok &= _check(f"integration effect ({effect:.2e}) >= 10x that bound", effect >= 1e-4)
    return ok


def test_T16_mixed_cadence_target():
    t_sc = np.arange(-0.15, 0.15, SC_EXPTIME_D)
    t_lc = 10.0 + np.arange(-0.15, 0.15, LC_EXPTIME_D)
    cfgs = [exposure_config(SC_EXPTIME_D), exposure_config(LC_EXPTIME_D)]
    ok = _check("exposure_config classes: SC->(1,.), LC->(15,.)",
                cfgs[0][0] == 1 and cfgs[1][0] == 15)
    models = build_transit_models([t_sc, t_lc], exp_list=cfgs)
    pv_lc = _pv(tc=10.0)
    f_sc = evaluate_transit_flux(_pv(), t_sc, model=models[0])
    f_lc = evaluate_transit_flux(pv_lc, t_lc, model=models[1])
    f_sc_ref = evaluate_transit_flux(_pv(), t_sc)
    f_lc_ref = evaluate_transit_flux(pv_lc, t_lc, exptime=LC_EXPTIME_D, nsamples=15)
    ok &= _check("SC segment == independent instantaneous eval (shape+order)",
                 f_sc.shape == t_sc.shape and np.array_equal(f_sc, f_sc_ref))
    ok &= _check("LC segment == independent integrated eval (shape+order)",
                 f_lc.shape == t_lc.shape and np.array_equal(f_lc, f_lc_ref))
    # Scalar-time evaluation (the t_c,j locator path) with the LC config:
    c_int, fmin_int = _transit_center_and_depth(pv_lc, t_lc, exp_cfg=cfgs[1])
    c_inst, fmin_inst = _transit_center_and_depth(pv_lc, t_lc)
    ok &= _check("t_c,j locator runs on the integrated model and finds the center",
                 abs(c_int - 10.0) < 5e-3)
    ok &= _check("integrated minimum is shallower than instantaneous",
                 fmin_int > fmin_inst)
    return ok


def test_T17_likelihood_uses_integration():
    t_lc = np.arange(-0.15, 0.15, LC_EXPTIME_D)
    f_data = evaluate_transit_flux(_pv(), t_lc)          # data made WITHOUT integration
    theta = np.r_[PV0[[0, 1, 2]], PV0[4], PV0[5], PV0[6], PV0[7], PV0[8], PV0[3], 0.0]
    m_int = build_transit_models([t_lc], exp_list=[exposure_config(LC_EXPTIME_D)])
    m_inst = build_transit_models([t_lc])
    args = (np.array([0.0]), [f_data], [t_lc], [2e-4], np.zeros(10), np.ones(10))
    lp_int = logprob_dbdt(theta, *args, models=m_int)
    lp_inst = logprob_dbdt(theta, *args, models=m_inst)
    return _check(f"logprob_dbdt differs with exposure models "
                  f"(|dlogp| = {abs(lp_int - lp_inst):.3g})",
                  np.isfinite(lp_int) and np.isfinite(lp_inst) and
                  abs(lp_int - lp_inst) > 1.0)


def test_T18_extraction_split_dedup_fallback():
    ok = True
    koi = FakeKOI(koi_time0bk=0.0, koi_period=10.0, koi_duration=4.8)
    eph = TransitEphemeris(0.0, 10.0)
    # Product 0 (LC): epochs 0 and 1. Product 1 (SC): epoch 1 again -> the concatenated
    # array has a same-epoch LC->SC boundary that the class split must break.
    t_lc0 = np.arange(-0.15, 0.15, LC_EXPTIME_D)
    t_lc1 = 10.0 + np.arange(-0.15, 0.15, LC_EXPTIME_D)
    t_sc1 = 10.0 + np.arange(-0.15, 0.15, SC_EXPTIME_D)
    times = [np.concatenate([t_lc0, t_lc1]), t_sc1]
    rng = np.random.default_rng(2)
    fluxs = [1 + 1e-4 * rng.standard_normal(times[0].size),
             1 + 1e-4 * rng.standard_normal(times[1].size)]
    t_out, f_out, ferr_out, exp_out = get_transit_arrays(
        times, fluxs, None, [True, False], [0, 1], koi, eph,
        exptimes=[LC_EXPTIME_D, SC_EXPTIME_D])
    ok &= _check(f"class split yields 3 single-cadence segments (got {len(t_out)})",
                 len(t_out) == 3)
    ok &= _check("per-segment scalars carry the right provenance",
                 np.allclose(exp_out, [LC_EXPTIME_D, LC_EXPTIME_D, SC_EXPTIME_D]))
    # (c) dedup keeps the SC observation of epoch 1:
    t_sel, f_sel, fe_sel, sel = select_cadence_per_transit(None, t_out, f_out, ferr_out, eph)
    exp_sel = [exp_out[i] for i in sel]
    ok &= _check("dedup keeps SC for the doubly-observed epoch",
                 len(t_sel) == 2 and np.allclose(sorted(exp_sel), [SC_EXPTIME_D, LC_EXPTIME_D]))
    # (d) SC->LC quality fallback keeps the LC scalar through the production mapping:
    #     candidate 2 (the SC one) fails the quality cuts -> surv drops it; the survivor
    #     selection then picks the LC candidate for epoch 1.
    cand_exp = exp_out                       # [LC(e0), LC(e1), SC(e1)]
    surv = [0, 1]                            # SC candidate failed a cut
    t_s, f_s, fe_s, sel_s = select_cadence_per_transit(
        None, [t_out[i] for i in surv], [f_out[i] for i in surv],
        [ferr_out[i] for i in surv], eph)
    exp_aligned = align_to_selection(cand_exp, surv, sel_s)
    ok &= _check("SC->LC fallback keeps LC exposure provenance via align_to_selection",
                 len(exp_aligned) == 2 and np.allclose(exp_aligned, [LC_EXPTIME_D, LC_EXPTIME_D]))
    # (e) fallback-fitter subset rule: shortest FIT-WORTHY cadence class.
    mk = lambda nsc, nlc: (np.arange(nsc + nlc, dtype=float),
                           np.ones(nsc + nlc),
                           np.r_[np.full(nsc, SC_EXPTIME_D), np.full(nlc, LC_EXPTIME_D)])
    ts, fs, ex = mk(6, 8)
    ok &= _check("adequate SC subset wins", _choose_cadence_subset(ts, fs, ex)[2] == SC_EXPTIME_D)
    ts, fs, ex = mk(3, 8)
    ok &= _check("SC below the 5-point floor falls back to LC",
                 _choose_cadence_subset(ts, fs, ex)[2] == LC_EXPTIME_D)
    ts, fs, ex = mk(3, 4)
    ok &= _check("neither class fit-worthy -> epoch skipped (None)",
                 _choose_cadence_subset(ts, fs, ex) is None)
    # product_exptimes: metadata first, median-diff fallback.
    class _L:
        def __init__(self, meta): self.meta = meta
    exps = product_exptimes([_L({'TIMEDEL': 0.02043}), _L({})],
                            [t_lc0, t_sc1])
    ok &= _check("product_exptimes: TIMEDEL preferred, median-diff fallback",
                 math.isclose(exps[0], 0.02043) and math.isclose(exps[1], SC_EXPTIME_D))
    return ok


def main():
    tests = [test_T1_prior_independent_of_individual_fits, test_T2_b0_uniform_and_q_fallback,
             test_T3_combine_errors_convention, test_T4_valid_negative_slope_is_finite,
             test_T5_invalid_slopes_rejected, test_T6_no_old_boundary_literal,
             test_T7_e_zero_reproduces_circular, test_T8_geometry_roundtrip,
             test_T9_duration_direction, test_T10_flux_duration_consistency,
             test_T11_no_hardcoded_circular_durations_and_pinned_upstream,
             test_T12_synthetic_identity_drift_bound, test_T13_sc_unchanged,
             test_T14_lc_differs_at_ingress, test_T15_brute_force_supersampling,
             test_T16_mixed_cadence_target, test_T17_likelihood_uses_integration,
             test_T18_extraction_split_dedup_fallback]
    all_ok = True
    for t in tests:
        print(t.__name__)
        all_ok &= bool(t())
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
