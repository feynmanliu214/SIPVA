#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""


import math
import numpy as np
from numpy import sqrt, arctan2, squeeze, ones
from pytransit import QuadraticModel
from pytransit.orbits import as_from_rhop, i_from_baew


# Single source of truth for the photometric passband. The live pipeline is Kepler-only; the
# limb-darkening dict keys PyTransit emits are derived from this name (q1_<PASSBAND>, ...), so
# everything that reads those keys imports Q1_KEY/Q2_KEY rather than hardcoding the band.
PASSBAND = 'Kepler'
Q1_KEY, Q2_KEY = f'q1_{PASSBAND}', f'q2_{PASSBAND}'


class InvalidGeometryError(ArithmeticError):
    """A parameter draw implies no valid transit geometry (e >= 1, |cos i| > 1, or a
    non-finite orbit scale). Subclasses ArithmeticError so the existing
    ``except ArithmeticError`` rejection paths in the likelihoods treat such a draw as
    out-of-support (-inf / +inf) instead of crashing."""


def evaluate_transit_flux(pv, time_array, model=None, exptime=0.0, nsamples=1):
    """Model flux for one transit. ``model``, if given, is a prebuilt QuadraticModel whose
    time grid (and exposure configuration) must match; otherwise a model is built here with
    the scalar per-segment exposure config (``nsamples`` > 1 -> ``nsamples``-point
    finite-exposure integration over ``exptime`` days; the default is the legacy
    instantaneous evaluation, byte-identical for short-cadence/synthetic callers)."""
    #pv = atleast_2d(pv)
    flux = ones(len(time_array))

    # Extract parameters from pv based on the default sequence (q1, q2 are positional here)
    rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2 = pv

    # Convert q1 and q2 to ldc
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    ldc = [u1, u2]

    # Convert other parameters
    k = sqrt(k2_1)
    aor = as_from_rhop(rho, p_1)
    ecc = secw_1**2 + sesw_1**2
    w = arctan2(sesw_1, secw_1)

    # Eccentric mid-transit geometry: cos i = (b / aor) * (1 + e sin w) / (1 - e^2).
    # i_from_baew implements exactly this (verified against pytransit 2.7.1); at e = 0 it is
    # bit-identical to the circular i_from_ba conversion. Draws with no valid geometry are
    # rejected, never clamped.
    if not (np.isfinite(aor) and aor > 0.0) or not (0.0 <= ecc < 1.0):
        raise InvalidGeometryError(f"invalid orbit: a/R*={aor}, e={ecc}")
    cosi_arg = (b_1 / aor) * (1.0 + ecc * np.sin(w)) / (1.0 - ecc ** 2)
    if not np.isfinite(cosi_arg) or abs(cosi_arg) > 1.0:
        raise InvalidGeometryError(f"no transit geometry: |cos i| = {abs(cosi_arg)} > 1")
    inc = i_from_baew(b_1, aor, ecc, w)

    # Evaluate the model. If a prebuilt model is supplied its time grid must match
    # `time_array`; reusing it avoids the expensive QuadraticModel construction + set_data
    # on every call (numerically identical output).
    if model is None:
        model = QuadraticModel()
        if nsamples and int(nsamples) > 1:
            model.set_data(time_array, nsamples=int(nsamples), exptimes=float(exptime))
        else:
            model.set_data(time_array)
    flux += model.evaluate(k, ldc, tc_1, p_1, aor, inc, ecc, w) - 1.

    return squeeze(flux)


def build_transit_models(t_out_list, exp_list=None):
    """Build one QuadraticModel per transit, with set_data done once, for reuse across
    the many likelihood evaluations of the global fit. Index-aligned with t_out_list.

    ``exp_list``, if given, is an index-aligned list of per-segment scalar exposure configs
    ``(nsamples, exptime_days)`` (or None entries): long-cadence segments get
    nsamples-point finite-exposure integration; ``None`` / nsamples <= 1 keeps the legacy
    instantaneous evaluation (short cadence and the synthetic path).

    Construction + set_data is the dominant per-call cost; doing it once per fixed transit
    time grid (instead of on every evaluate_transit_flux call) is numerically identical.
    """
    models = []
    for j, time_array in enumerate(t_out_list):
        tm = QuadraticModel()
        cfg = exp_list[j] if exp_list is not None else None
        if cfg is not None and int(cfg[0]) > 1:
            tm.set_data(time_array, nsamples=int(cfg[0]), exptimes=float(cfg[1]))
        else:
            tm.set_data(time_array)
        models.append(tm)
    return models


def calculate_stellar_density(rs_over_a, period_days):
    """
    Calculate the stellar density given the inverse scaled semi-major axis (R_star/a, NOT
    a/R_star) and the orbital period (P) in days.

    Parameters:
    rs_over_a (float): The star's radius scaled by the semi-major axis (R_star/a, dimensionless).
    period_days (float): The orbital period in days.

    Returns:
    float: The stellar density in g/cm^3.
    """

    a_over_rs = 1/rs_over_a
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    period_seconds = period_days*24*3600  # convert period from days to seconds

    # Calculate the density using the given formula
    density = (3 * math.pi) / (G * period_seconds ** 2) * (a_over_rs) ** 3

    return density/1000


def calculate_uncertainty(param_array):
    param_array = np.array(param_array)
    uncen_value = np.maximum(param_array[:, 1], param_array[:, 2])
    return uncen_value
