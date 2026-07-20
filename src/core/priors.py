#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu

Prior construction for transit fits. Four independent strategies:

1. Catalog-derived  -- widths from the NASA Exoplanet Archive KOI error bars.
   This is the canonical path for the live TDV pipeline.
   ``koi_prior_spec`` -> ``apply_prior_spec`` / ``set_koi_priors``.
2. Synthetic        -- hardcoded widths for injection / SNR studies (NOT derived
   from any catalog; see ``set_synthetic_priors``). The hardcoded-vs-catalog width
   split is intentional, not an oversight.
3. SIPVA catalog-only -- the 10-parameter global db/dt fit prior
   (``sipva_prior_spec``): catalog widths + Uniform b_0 + Normal db_dt. Deliberately
   independent of the individual-fit posteriors, which may seed optimizers/walkers
   but never enter the prior density.
4. Posterior-derived -- widths from the medians of an earlier per-transit fit
   (``set_posterior_priors``); retained for the synthetic/injection global fit.

PyTransit's ``set_prior`` resolves parameters by name, so the order in which
priors are applied carries no behavior.
"""

import os
import numpy as np
from data import get_koi
from limb_darkening import koi_ld_prior
from model import calculate_uncertainty, Q1_KEY, Q2_KEY


# Canonical parameter order for the Normal core priors, shared by the catalog and
# synthetic paths so both read the same way. (Order is cosmetic: set_prior is
# name-keyed.)
CORE_PARAMS = ('rho', 'p_1', 'k2_1', 'b_1', 'secw_1', 'sesw_1')


# --- Shared per-parameter helpers ------------------------------------------

def _combine_errors(err1, err2):
    """Symmetrize the catalog's asymmetric +/- errors: mean of the available magnitudes
    (the single |err| when only one is present, 0.0 when neither). Replaces the previous
    quadrature combination, which inflated the width by a factor of sqrt(2) (symmetric
    errors) up to ~2 (strongly asymmetric) and disagreed with the mean-symmetrization
    already used for koi_ror (sigma_p below) and in limb_darkening._sym_err."""
    vals = [abs(e) for e in (err1, err2) if e]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _k2_from_ratio(p, sigma_p):
    """Map radius ratio p (and its sigma) to k2 = p**2 with propagated sigma.

    Returns (mu, sigma_raw) where sigma_raw = |dk2/dp| * sigma_p = |2 p sigma_p|.
    Callers apply their own scale factor and floor.
    """
    return p ** 2, abs(2.0 * p * sigma_p)


def _ecc_to_secw_sesw(ecc, omega_deg):
    """Catalog (e, omega[deg]) -> (sqrt(e) cos w, sqrt(e) sin w), clamping e >= 0."""
    omega = np.deg2rad(float(omega_deg))
    s = np.sqrt(max(float(ecc), 0.0))
    return s * np.cos(omega), s * np.sin(omega)


# --- 1. Catalog-derived priors ---------------------------------------------

def koi_prior_spec(koi_number, num_transits=None, factor=1.0, ecc_sigma=0.1):
    """Compute the KOI-derived Normal priors as a plain, picklable list of
    (param_name, 'NP', mu, sigma) tuples. These are identical for every transit of a target,
    so they can be computed once and shipped to parallel workers (which apply them via
    apply_prior_spec) without pickling a TransitAnalysis object.

    ``num_transits`` is accepted for signature compatibility but currently unused
    (the catalog widths are not sqrt(N)-scaled).
    """
    koi = get_koi(koi_number)

    # --- period prior ---
    per_sigma = _combine_errors(koi.koi_period_err1, koi.koi_period_err2) * factor

    # --- impact parameter prior ---
    b_sigma = _combine_errors(koi.koi_impact_err1, koi.koi_impact_err2) * factor

    # --- stellar density prior (g/cm^3 in KOI table) ---
    rho_sigma = _combine_errors(koi.koi_srho_err1, koi.koi_srho_err2) * factor
    # Cap the rho prior width. Catalog stellar-density errors are large and wildly asymmetric (e.g.
    # KOI 103.01: +0.013/-0.887), and _combine_errors symmetrizes them in quadrature into an almost
    # flat prior. Since per-transit duration scales as T14 ~ rho^(-1/3), that lets a GP-detrend-
    # corrupted likelihood pull rho to a spuriously low value and balloon the duration. Capping the
    # width at RHO_PRIOR_FRAC * srho keeps the prior informative-but-generous (>=+/-3 sigma still
    # spans %-level real TDV) while putting the ~9x-low degenerate mode at >7 sigma. The cap can only
    # narrow the width, never widen it. See docs/2026-06-10_reject_nonphysical_transits_plan.md.
    rho_prior_frac = float(os.environ.get("TDV_RHO_PRIOR_FRAC", "0.12"))
    rho_sigma = min(rho_sigma, rho_prior_frac * float(koi.koi_srho))

    # --- radius ratio squared k2 = p^2 ---
    # Propagate via dk2/dp = 2p. Use symmetrized sigma_p from KOI ror errors.
    p = koi.koi_ror
    if p is None:
        # fall back to a weak width if missing
        p = 0.05
        sigma_p = 0.02
    else:
        sigma_p = 0.5 * (abs(koi.koi_ror_err1 or 0.0) + abs(koi.koi_ror_err2 or 0.0))
        if sigma_p == 0.0:
            sigma_p = max(1e-4, 0.02 * p)  # tiny floor
    k2_mu, k2_sigma_raw = _k2_from_ratio(p, sigma_p)
    k2_sigma = k2_sigma_raw * factor

    # --- eccentricity parameters ---
    if (koi.koi_longp is None) or (koi.koi_eccen is None):
        secw_1_value = 0.0
        sesw_1_value = 0.0
    else:
        # koi_longp is in degrees in the NASA archive
        secw_1_value, sesw_1_value = _ecc_to_secw_sesw(koi.koi_eccen, koi.koi_longp)

    # Normal priors (NP = Normal(mu, sigma)), in canonical CORE_PARAMS order.
    spec = [
        ('rho',    'NP', float(koi.koi_srho),   max(rho_sigma, 1e-6)),
        ('p_1',    'NP', float(koi.koi_period), max(per_sigma, 1e-10)),
        ('k2_1',   'NP', k2_mu,                 max(k2_sigma, 1e-8)),
        ('b_1',    'NP', float(koi.koi_impact), max(b_sigma, 1e-4)),
        # Eccentricity parameters: weak normals around catalog (or 0 if missing)
        ('secw_1', 'NP', float(secw_1_value),   float(ecc_sigma)),
        ('sesw_1', 'NP', float(sesw_1_value),   float(ecc_sigma)),
    ]

    # Limb darkening: Normal priors on the Kipping q1/q2, centered on the values expected from
    # the host star's Teff/logg/[Fe/H] (via PyLDTk) with the propagated stellar-parameter
    # uncertainty as the width. None -> leave q1/q2 on PyTransit's default uniform [0,1].
    ld = koi_ld_prior(koi_number)
    if ld is not None:
        q1_mu, q1_sig, q2_mu, q2_sig = ld
        spec += [(Q1_KEY, 'NP', q1_mu, q1_sig), (Q2_KEY, 'NP', q2_mu, q2_sig)]
    return spec


def apply_prior_spec(ta, spec):
    """Apply a list of (param_name, dist, a, b) prior tuples (from koi_prior_spec) to a TA."""
    for name, dist, a, b in spec:
        ta.set_prior(name, dist, a, b)
    return ta


def set_koi_priors(ta_input, koi_number, num_transits, factor=1.0, ecc_sigma=0.1):
    """Apply the catalog-derived priors to a TA (or list of TAs). Convenience wrapper
    over koi_prior_spec + apply_prior_spec for the in-process (non-parallel) path."""
    spec = koi_prior_spec(koi_number, num_transits, factor=factor, ecc_sigma=ecc_sigma)
    if isinstance(ta_input, list):
        return [apply_prior_spec(ta, spec) for ta in ta_input]
    return apply_prior_spec(ta_input, spec)


# --- 2. Synthetic / injection priors ---------------------------------------

def synthetic_prior_spec(period, impact_param, rho, planet_star_ratio,
                         secw_1_value=0.0, sesw_1_value=0.0,
                         ld_uniform=True, noise_uniform=True):
    """Hardcoded-width synthetic-injection priors as a plain, picklable list of
    (param_name, dist, a, b) tuples, in canonical CORE_PARAMS order.

    The injection-path analogue of ``koi_prior_spec``: the same spec can be shipped to the
    parallel TDV workers (which rebuild a TransitAnalysis locally and apply it via
    ``apply_prior_spec``) without pickling a TransitAnalysis. Widths are fixed by design,
    NOT catalog-derived. The centers are typically caller-perturbed off the true simulation
    inputs to mimic literature-based estimates.
    """
    k2 = planet_star_ratio ** 2

    # Normal priors, in canonical CORE_PARAMS order.
    spec = [
        ('rho',    'NP', rho,           rho / 10.0),
        ('p_1',    'NP', period,        period * 1e-5),
        ('k2_1',   'NP', k2,            k2 / 50.0),
        ('b_1',    'NP', impact_param,  0.2),
        ('secw_1', 'NP', secw_1_value,  1e-5),
        ('sesw_1', 'NP', sesw_1_value,  1e-5),
    ]
    # Uniform priors
    if ld_uniform:
        spec += [(Q1_KEY, 'UP', 0.0, 1.0), (Q2_KEY, 'UP', 0.0, 1.0)]
    if noise_uniform:
        spec += [('wn_loge_0', 'UP', -4.0, 0.0)]
    return spec


def _spec_to_prior_dict(spec):
    """Describe a prior spec as a dict (NP -> Normal mean/std, UP -> Uniform lower/upper)."""
    out = {}
    for name, dist, a, b in spec:
        if dist == 'NP':
            out[name] = {"type": "Normal", "mean": a, "std": b}
        elif dist == 'UP':
            out[name] = {"type": "Uniform", "lower": a, "upper": b}
        else:
            out[name] = {"type": dist, "a": a, "b": b}
    return out


def set_synthetic_priors(ta_input, num_transits, period, impact_param, rho, planet_star_ratio,
                         secw_1_value=0.0, sesw_1_value=0.0,
                         ld_uniform=True, noise_uniform=True):
    """Apply hardcoded-width synthetic-injection priors to a single TA or a list of them,
    and also return a dict describing the priors.

    ``num_transits`` is accepted for signature compatibility but unused. For the parallel
    TDV path, prefer ``synthetic_prior_spec`` (a picklable spec) over a prebuilt TA.
    """
    spec = synthetic_prior_spec(period, impact_param, rho, planet_star_ratio,
                                secw_1_value, sesw_1_value, ld_uniform, noise_uniform)
    prior_dict = _spec_to_prior_dict(spec)
    if isinstance(ta_input, list):
        return [apply_prior_spec(ta, spec) for ta in ta_input], prior_dict
    return apply_prior_spec(ta_input, spec), prior_dict


# --- 3. SIPVA global-fit priors (catalog-only) -------------------------------

# Canonical SIPVA global-fit parameter order (10 parameters): tc_1 is the tightly
# constrained shared timing offset Delta t_e; b_0/db_dt close the vector.
SIPVA_PARAMS = ('rho', 'tc_1', 'p_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY,
                'b_0', 'db_dt')


def sipva_prior_spec(koi_number, ecc_sigma=0.1):
    """Catalog-only priors for the 10 SIPVA global-fit parameters: an ordered list of
    (param_name, kind, a, b) tuples in SIPVA_PARAMS order, with kind 'NP' (Normal(a=mu,
    b=sigma)) or 'UP' (Uniform(a=lower, b=upper)).

    By construction this takes NO input from the individual-fit posteriors:
      - rho / p_1 / k2_1 / secw_1 / sesw_1 reuse the catalog math of ``koi_prior_spec``
        (single source of truth; the rho width keeps the 0.12*srho cap -- the rho_eff
        density-prior approximation is unchanged);
      - q1/q2 come from the PyLDTk stellar-atmosphere prior, with a Uniform(0, 1)
        fallback when no PyLDTk prior is derivable;
      - b_0 is a broad Uniform(0, 1) -- the individual-fit regression prediction may seed
        the optimizer/walkers but never this prior;
      - db_dt is Normal(0, 0.2 / yr);
      - tc_1 is the tight shared timing offset Normal(0, 1e-8 d), exactly as before.
    """
    base = {name: (dist, a, b)
            for name, dist, a, b in koi_prior_spec(koi_number, ecc_sigma=ecc_sigma)}
    spec = [
        ('rho',) + base['rho'],
        ('tc_1', 'NP', 0.0, 1e-8),
        ('p_1',) + base['p_1'],
        ('k2_1',) + base['k2_1'],
        ('secw_1',) + base['secw_1'],
        ('sesw_1',) + base['sesw_1'],
    ]
    for qkey in (Q1_KEY, Q2_KEY):
        spec.append(((qkey,) + base[qkey]) if qkey in base else (qkey, 'UP', 0.0, 1.0))
    spec += [
        ('b_0', 'UP', 0.0, 1.0),
        ('db_dt', 'NP', 0.0, 0.2),
    ]
    return spec


# --- 4. Posterior-derived priors --------------------------------------------

def set_posterior_priors(ta, priors, allowed_params, factor=2):
    """Seed Normal priors from the medians/uncertainties of an earlier per-transit fit.
    Only parameters in ``allowed_params`` are set; ``priors`` maps name -> array of
    [median, err_high, err_low] rows."""
    for param, values in priors.items():
        if param not in allowed_params:
            continue  # Skip this iteration if the parameter isn't in the allowed list
        values = np.array(values)  # Ensure that 'values' is a NumPy array
        median_value = np.median(values[:, 0])
        uncen_value = np.median(calculate_uncertainty(values))
        ta.set_prior(param, 'NP', median_value, uncen_value * factor)
