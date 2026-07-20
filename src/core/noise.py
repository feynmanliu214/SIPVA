#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realistic-noise components for synthetic light curves.

Adds correlated (red) noise and observational gaps on top of the white-noise
model, per docs/2026-06-07_realistic_noise_lightcurves_plan.md (Codex-reviewed,
approved round 6). Granulation correlation time (hours) is much shorter than the
inter-transit spacing (days), so the GP is sampled independently per transit
window — exact, cheap (N=300), and dependency-light (numpy Cholesky; no
celerite2, which is absent).
"""

import numpy as np

# Bounded retry cap for the gap mask (plan: MAX_GAP_RETRIES).
MAX_GAP_RETRIES = 100

_SQRT3 = np.sqrt(3.0)


def matern32_cov(t, sigma, tau):
    """Matern-3/2 covariance matrix on time array ``t``.

    k(d) = sigma**2 * (1 + sqrt(3) d / tau) * exp(-sqrt(3) d / tau), with
    d = |t_i - t_j|. ``t`` and ``tau`` must share units (days here).
    """
    t = np.asarray(t, dtype=float)
    d = np.abs(t[:, None] - t[None, :])
    scaled = _SQRT3 * d / tau
    return sigma ** 2 * (1.0 + scaled) * np.exp(-scaled)


def averaging_factor(n, cadence_days, tau):
    """g = Var(mean of n unit-variance Matern-3/2 points) = (1/n^2) sum_ij corr_ij.

    This is the factor by which correlated noise averages down over an n-point
    window: g -> 1/n for white noise (tau -> 0), g -> 1 for fully correlated
    noise. Used to express CDPP (variance of the windowed mean) for the red GP.
    """
    t = np.arange(n) * cadence_days
    return float(matern32_cov(t, 1.0, tau).mean())


def sample_red_noise(t, sigma_r, tau, rng):
    """Draw one correlated red-noise realization on ``t`` (exact GP, Cholesky).

    Returns zeros when ``sigma_r == 0`` (a valid "gaps-only, no red noise"
    config): the covariance and its jitter would both be zero, so Cholesky would
    otherwise fail.
    """
    t = np.asarray(t, dtype=float)
    if sigma_r == 0:
        return np.zeros(t.size)
    cov = matern32_cov(t, sigma_r, tau)
    cov[np.diag_indices_from(cov)] += 1e-12 * sigma_r ** 2  # Cholesky jitter
    chol = np.linalg.cholesky(cov)
    return chol @ rng.standard_normal(t.size)


def apply_gaps(num_transits, gap_prob, min_keep, rng):
    """Bernoulli per-transit dropout with a kept-count floor.

    Each transit is dropped independently with probability ``gap_prob``. Redraw
    the whole mask up to ``MAX_GAP_RETRIES`` times until at least
    ``min(min_keep, num_transits)`` transits survive; if still infeasible, keep
    all transits and flag it.

    Returns ``(keep_mask, gap_retry_exhausted)``: the bool flag lets the caller
    record whether the retry cap was hit (a bare mask cannot carry that).
    """
    floor = min(min_keep, num_transits)
    for _ in range(MAX_GAP_RETRIES):
        keep_mask = rng.random(num_transits) >= gap_prob
        if int(keep_mask.sum()) >= floor:
            return keep_mask, False
    return np.ones(num_transits, dtype=bool), True
