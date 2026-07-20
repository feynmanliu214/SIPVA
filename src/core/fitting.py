#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""


import numpy as np
from model import evaluate_transit_flux, PASSBAND, Q1_KEY, Q2_KEY


def log_prior(theta, means, stds, friction=1, ld_normal=False, kinds=None):
    """
    Calculate the log-prior for a set of parameters with mixed prior types.
    The limb-darkening q1/q2 (indices 6,7) are always confined to [0,1]; within that support
    they use a Normal(means,stds) prior when ``ld_normal`` is True (real-KOI fits, where the
    center/width come from the PyLDTk stellar-atmosphere prior) and a flat prior otherwise
    (synthetic/injection runs). All other parameters use Normal priors.

    ``kinds`` (optional): per-parameter prior kinds for the catalog-driven real-KOI path,
    'NP' (Normal(means[i], stds[i])) or 'UP' (Uniform: means[i]=lower, stds[i]=upper).
    None reproduces the legacy all-Normal behavior exactly (the synthetic path relies on
    that). The kinds vector comes from ``priors.sipva_prior_spec`` and NEVER from
    individual-fit posteriors.
    """
    if len(theta) != len(means) or len(theta) != len(stds):
        raise ValueError("The lengths of theta, means, and stds must be the same.")

    # Parameter order: ['rho', 'tc_1', 'p_1', 'k2_1', 'secw_1', 'sesw_1', 'q1', 'q2', 'b_0', 'db_dt']
    log_prior_value = 0.0

    for i, param_value in enumerate(theta):
        kind = kinds[i] if kinds is not None else 'NP'
        if i == 6 or i == 7:  # q1 (index 6), q2 (index 7) -- limb darkening
            if not (0.0 <= param_value <= 1.0):
                return -np.inf
            # Normal prior within [0,1] when PyLDTk-derived (legacy flag or kinds=='NP');
            # else flat within [0,1] (adds 0, both legacy-synthetic and UP fallback).
            if (kinds is None and ld_normal) or (kinds is not None and kind == 'NP'):
                log_prior_value += -0.5 * ((param_value - means[i]) / stds[i]) ** 2 - np.log(np.sqrt(2 * np.pi) * stds[i])
        elif kind == 'UP':
            lo, hi = means[i], stds[i]
            if not (lo <= param_value <= hi):
                return -np.inf
            log_prior_value += -np.log(hi - lo)
        else:
            # Normal prior for other parameters
            log_prior_value += -0.5 * ((param_value - means[i]) / stds[i]) ** 2 - np.log(np.sqrt(2 * np.pi) * stds[i])

    # The Gaussian likelihood normalization belongs in the likelihood (logprob_dbdt), not the
    # prior -- it was previously added here, double-counting it. Removed.
    return log_prior_value * friction


# Broad numerical safety support for the db/dt slope (1/yr). The old restrictive hard
# boundary is gone (it truncated real posteriors, e.g. KOI 103.01); the OPERATIVE constraint
# is the per-epoch 0 <= b_j = b_0 + db_dt * x_j <= 1 check below. This bound only guards the
# numerics far outside the Normal(0, 0.2^2) prior and must be verified inactive on all
# validation posteriors.
DB_DT_SUPPORT = 1.0


def logprob_dbdt(pars, delta_t_values, f_out, t_out, ferr_out, means, stds, models=None):
    ''' Returns scalar chi-squared value for transit model given by vector pars and empirical
     transit fluxes f_out and times t_out. `models`, if given, is an index-aligned list of
     prebuilt QuadraticModels (one per transit) reused to skip per-call model construction.'''

    # Ensure means and stds are numpy arrays
    means = np.array(means)
    stds = np.array(stds)
    
    # Check if any parameter is outside 4 stds of its mean
    #if any(np.abs(pars[:-1] - means[:-1]) > 4 * stds[:-1]):
        #return -np.inf
    #if any(np.abs(pars - means) > 4 * stds):
          #return -np.inf
    rho, tc_1, p_1, k2_1, secw_1, sesw_1, q1, q2, b_0, db_dt = pars
    N_data = len(np.concatenate(t_out))
    logprob = 0.0

    if not (-1e-5 <= tc_1 <= 1e-5):    # Bounds for 2nd parameter (index 1)
        return -np.inf
    if not (-DB_DT_SUPPORT <= db_dt <= DB_DT_SUPPORT):   # broad numerical safety only;
        return -np.inf                                   # per-epoch b_j check is operative
    if not (0 <= q1 <= 1):        # Bounds for 7th parameter (index 6)
        return -np.inf
    if not (0 <= q2 <= 1):        # Bounds for 8th parameter (index 7)
        return -np.inf

    if not (0 <= b_0 <= 1):
        return -np.inf

    if k2_1 < 5e-6:
        return -np.inf

    if rho < 0:
        return -np.inf

    if rho > 100:
        return -np.inf

    for i in range(len(t_out)):
        # Compute b_1 for this transit
        b_1 = b_0 + db_dt * delta_t_values[i]
        b_1 = b_1[0] if isinstance(b_1, np.ndarray) and b_1.size == 1 else b_1

        # Create parameter vector for this transit
        pv = np.array([rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2])

        if b_1 < 0:
            return -np.inf
        
        if b_1 > 1:
            return -np.inf


        # Compute the model flux. PyTransit's Mandel-Agol quadratic model can raise an arithmetic
        # error (e.g. ZeroDivisionError) at singular orbital geometries; treat such a draw as
        # out-of-support and reject it, mirroring the finite-value guards below.
        try:
            fmod = evaluate_transit_flux(pv, t_out[i], model=None if models is None else models[i])
        except ArithmeticError:
            return -np.inf

        if np.shape(fmod) != np.shape(f_out[i]):
            return -np.inf                                               # check shape BEFORE subtracting (else raises)
        resid = fmod - f_out[i]
        if not np.all(np.isfinite(resid)):
            return -np.inf                                               # invalid model for these params -> reject
        sig2 = np.broadcast_to(ferr_out[i], np.shape(f_out[i])) ** 2
        if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
            return -np.inf                                               # invalid noise -> reject
        logprob -= 0.5 * np.sum((resid ** 2) / sig2)                     # chi-squared term
        logprob -= 0.5 * np.sum(np.log(2 * np.pi * sig2))                # one term per point (was per transit)

    return logprob


def neg_log_likelihood(params, delta_t_values, f_out, t_out, ferr_out, means, stds, models=None):
    """
    The negative log-likelihood function that needs to be minimized. `models`, if given, is an
    index-aligned list of prebuilt QuadraticModels reused to skip per-call model construction.
    """
    # Ensure means and stds are numpy arrays
    means = np.array(means)
    stds = np.array(stds)

    rho, tc_1, p_1, k2_1, secw_1, sesw_1, q1, q2, b_0, db_dt = params
    logprob = 0.0

    # Mirror the FULL posterior hard support (logprob_dbdt) so the MLE seed cannot land where
    # the posterior has zero probability. find_mle_and_save MINIMIZES, so out-of-support -> +inf
    # (a NaN/invalid point used to set logprob = -inf, which the minimizer would happily pick).
    if not (-1e-5 <= tc_1 <= 1e-5):     return np.inf
    if not (-DB_DT_SUPPORT <= db_dt <= DB_DT_SUPPORT):  return np.inf
    if not (0 <= q1 <= 1):              return np.inf
    if not (0 <= q2 <= 1):              return np.inf
    if not (0 <= b_0 <= 1):             return np.inf
    if k2_1 < 5e-6:                     return np.inf
    if not (0 <= rho <= 100):           return np.inf

    for i in range(len(t_out)):
        # Create parameter vector for this transit
        b_1 = b_0 + db_dt * delta_t_values[i]
        b_1 = b_1[0] if isinstance(b_1, np.ndarray) and b_1.size == 1 else b_1
        if not (0 <= b_1 <= 1):
            return np.inf                                       # per-transit b_1 support (matches logprob_dbdt)

        pv = np.array([rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2])

        # Compute the model flux. Reject draws where the MA-quadratic model raises an arithmetic
        # error at singular geometry (DE minimizes, so out-of-support -> +inf).
        try:
            fmod = evaluate_transit_flux(pv, t_out[i], model=None if models is None else models[i])
        except ArithmeticError:
            return np.inf

        n_i = len(f_out[i])
        if np.shape(fmod) != np.shape(f_out[i]):
            return np.inf                                       # check shape BEFORE subtracting (else raises)
        resid = fmod - f_out[i]
        if not np.all(np.isfinite(resid)) or not np.isfinite(np.sum(resid ** 2)):
            return np.inf                                       # penalize (DE minimizes)
        rss = np.sum(resid ** 2)
        # Profiled Gaussian neg-log-likelihood with sigma_hat^2 = RSS/N: 0.5*N*log(RSS/N) + const.
        # The N factor (was missing) is the profiling prefactor, not a dof correction.
        logprob += 0.5 * n_i * np.log(max(rss, 1e-300) / n_i)   # floor avoids log(0)

    return logprob


def logprob_with_prior(theta, delta_t_values, f_out, t_out, ferr_out, means, stds, models=None,
                       ld_normal=False, kinds=None):
    lp = log_prior(theta, means, stds, friction = 1, ld_normal=ld_normal, kinds=kinds)
    N_data = len(np.concatenate(t_out))
    #lp = lp/len(t_out)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logprob_dbdt(theta, delta_t_values, f_out, t_out, ferr_out, means, stds, models)


import numpy as np
from collections import defaultdict
from model import calculate_uncertainty


def run_transit_analysis(ta, plot=False):
    # Run the MCMC
    ta.optimize_global(niter=750, npop=100)
    
    if plot:
        ta.plot_light_curves(method='fit')
        
    ta.sample_mcmc(niter=1500, thin=20, repeats=3, save=False)

    # Print the rejection rate
    print_rejection_rate(ta.sampler)

    if plot:
        ta.plot_light_curves(method='posterior')

    # Sample the posterior
    df = ta.posterior_samples()

    return df


def param_posterior_extract(df, param_keys, context=""):
    param_values = {}
    for key in param_keys:
        param_type = 'derived_parameters' if 't14' in key or 't23' in key else 'posterior'
        param_values[key] = param_posterior_est(df, key, param_type, context=context)

    return param_values


def _fit_one_transit(task):
    """Worker for the parallel individual fit. Builds the per-transit TransitAnalysis locally
    (a prebuilt one cannot be pickled across the process boundary -- its ParameterSet loses its
    'frozen' attribute), runs the MCMC, and returns the small, picklable param dict (extracted
    in-worker so large posterior DataFrames don't cross the boundary).

    task = (index, times_i, fluxes_i, koi_number, prior_spec, seed_base, param_keys, center,
            exp_cfg)
    where ``exp_cfg`` is the segment's scalar exposure config (nsamples, exptime_days) or
    None for the legacy instantaneous evaluation (synthetic path / short cadence).
    """
    import random
    from ta_eccentric import EccentricTransitAnalysis
    from priors import apply_prior_spec

    index, times_i, fluxes_i, koi_number, prior_spec, seed_base, param_keys, center, exp_cfg = task

    # Strip the astropy MaskedNDArray wrapper. Its mask is silently dropped to None when the
    # array is pickled across the process boundary, which then breaks PyTransit's
    # nanstd(diff(f)) in TransitAnalysis. The flux carries no masked points, so plain float
    # arrays are numerically identical to the in-process (serial) path.
    times_i = np.ascontiguousarray(np.ma.getdata(times_i), dtype=float)
    fluxes_i = np.ascontiguousarray(np.ma.getdata(fluxes_i), dtype=float)

    # Deterministic per-worker reseed. Forked workers inherit the parent's NumPy/random state ->
    # correlated/identical streams across transits. Seeding from (seed_base + index) gives each
    # transit an independent, reproducible stream.
    seed = (seed_base + index) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

    # Rebuild the TA with the same priors the serial path used: a unique name (so PyTransit save
    # artifacts can't collide if saving is ever re-enabled), the per-transit tc_1 prior, then the
    # shared KOI-derived priors. EccentricTransitAnalysis = TransitAnalysis with consistent
    # eccentric mid-transit geometry in flux model AND derived t14/t23; exp_cfg switches on
    # long-cadence finite-exposure integration (None -> instantaneous, as before).
    nsm = int(exp_cfg[0]) if exp_cfg is not None else 1
    ta = EccentricTransitAnalysis(name=f"koi{koi_number}_t{index}", passbands=PASSBAND,
                                  times=times_i, fluxes=fluxes_i,
                                  nsamples=nsm,
                                  exptimes=float(exp_cfg[1]) if (exp_cfg is not None and nsm > 1) else 0.0)
    # Seed the (free) tc_1 prior at the TTV-aware predicted center when available, else the
    # segment median (today's behavior). The prior stays wide so the fit still floats tc_1.
    tc_seed = float(center) if center is not None else float(np.median(times_i))
    ta.set_prior('tc_1', 'NP', tc_seed, 0.5)
    apply_prior_spec(ta, prior_spec)

    df = run_transit_analysis(ta)
    # Thread KOI + segment index into the NaN-fraction audit log so a non-transiting-sample
    # warning emitted from a process-pool worker is attributable to a specific transit.
    return param_posterior_extract(df, param_keys, context=f"KOI-{koi_number} seg{index}")


def analyze_single_transit(ta):
    """Run the MCMC analysis on a single TransitAnalysis instance and return the derived parameters."""
    
    # Use the run_transit_analysis function from earlier to run the analysis
    df = run_transit_analysis(ta)
    
    # Extract derived parameters
    t14_1 = param_posterior_est(df, 't14_1', 'derived_parameters')
    t23_1 = param_posterior_est(df, 't23_1', 'derived_parameters')
    b_1 =  param_posterior_est(df, 'b_1', 'posterior')
    
    # Create a dictionary to hold all the parameters
    param_arrays = {}
    derived_parameters = ['t14_1', 't23_1']
    posterior_parameters = ['rho', 'tc_1', 'p_1', 'b_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY, 'wn_loge_0']

    all_parameters = derived_parameters + posterior_parameters
    
    for param in all_parameters:
        param_arrays[param] = param_posterior_est(df, param, 'derived_parameters' if param in derived_parameters else 'posterior')
    
    return t14_1, t23_1, b_1, param_arrays


def analyze_multiple_transits(ta_array):
    """Run the MCMC analysis on multiple TransitAnalysis instances and collect the derived parameters."""
    
    duration_array_14 = []
    duration_array_23 = []
    b_1_array = []
    
    # Initialize a dictionary to hold all parameter arrays
    aggregated_param_arrays = defaultdict(list)
    
    for ta in ta_array:
        t14_1, t23_1, b_1, param_arrays = analyze_single_transit(ta)
        duration_array_14.append(t14_1)
        duration_array_23.append(t23_1)
        b_1_array.append(b_1)
        
        # Aggregate parameter arrays
        for key, value in param_arrays.items():
            aggregated_param_arrays[key].append(value)
            
    # Convert lists to NumPy arrays for easier manipulation
    for key in aggregated_param_arrays:
        aggregated_param_arrays[key] = np.array(aggregated_param_arrays[key])
        
    return duration_array_14, duration_array_23, b_1_array, dict(aggregated_param_arrays)


def compute_keep_for_fit(param_arrays_0, rho_star=None, reject_factor=None):
    """Full-length boolean mask over the per-transit fit rows: True = keep for the regressions and the
    global db/dt fit.

    A row is dropped if its individual fit yielded no t14 estimate (``t14_1`` is None), or -- when
    ``rho_star`` is given (the real-KOI path) -- if its fitted stellar density ``rho`` median is
    non-finite or outside ``[rho_star / F, rho_star * F]`` with ``F = reject_factor`` (env
    ``TDV_RHO_REJECT_FACTOR``, default 3). Stellar density is a property of the star, identical for
    every transit, so a per-transit fit that lands far from the catalog value is degenerate (low rho
    -> over-long, flat-bottomed transit). Transit DURATION is deliberately NOT a criterion: it is the
    TDV signal we measure, so a duration-based cut could suppress real signal.

    Returns ``(keep_for_fit, rho_consistent, factor)``:
      - ``keep_for_fit``  -- list[bool], the combined mask (rho-consistency AND not-None).
      - ``rho_consistent`` -- list[bool], the rho-only audit flag written to the per-transit CSV
        (a None-t14 row whose rho is in band is still rho_consistent=True).
      - ``factor`` -- the F actually used (None when rho_star is None, e.g. synthetic runs).

    See docs/2026-06-10_reject_nonphysical_transits_plan.md.
    """
    n = len(param_arrays_0['t14_1'])
    not_none = [param_arrays_0['t14_1'][i] is not None for i in range(n)]

    if rho_star is None:
        rho_consistent = [True] * n
        factor = None
    else:
        if reject_factor is None:
            reject_factor = float(os.environ.get("TDV_RHO_REJECT_FACTOR", "3"))
        factor = float(reject_factor)
        lo, hi = rho_star / factor, rho_star * factor
        rho_consistent = []
        for i in range(n):
            row = param_arrays_0['rho'][i]
            rho_med = float(row[0]) if row is not None else float('nan')
            rho_consistent.append(bool(np.isfinite(rho_med) and lo <= rho_med <= hi))

    keep_for_fit = [bool(rc and nn) for rc, nn in zip(rho_consistent, not_none)]
    return keep_for_fit, rho_consistent, factor


def remove_none_entries(param_arrays, times, fluxes, ferr_out):
    none_indices = [i for i, val in enumerate(param_arrays['t14_1']) if val is None]
    
    if not none_indices:
        return times, fluxes, ferr_out  # Add ferr_out here to return 3 values
    
    new_times = [v for i, v in enumerate(times) if i not in none_indices]
    new_fluxes = [v for i, v in enumerate(fluxes) if i not in none_indices]
    new_ferr_out = [v for i, v in enumerate(ferr_out) if i not in none_indices]
    
    return new_times, new_fluxes, new_ferr_out


def fold_transits(times, param_arrays):
    folded_transits = {}
    tc_1_values = np.array(param_arrays['tc_1'])[:, 0]  # Assuming tc_1_values are stored as the first column
    
    for i, (time_array, tc_1) in enumerate(zip(times, tc_1_values)):
        folded_time = time_array - tc_1
        folded_transits[tc_1] = folded_time  # Use tc_1 as a unique key to identify each transit
        
    return folded_transits


import os
import pickle
import numpy as np
import pandas as pd
import emcee
from scipy.optimize import differential_evolution, minimize_scalar
from model import evaluate_transit_flux
from model import calculate_uncertainty


def fold_transits_and_calculate_delta_t(times, transit_mid_times_list):
    folded_transits = []
    tc_1_values = transit_mid_times_list # Assuming tc_1_values are stored as the first column
    delta_t_values = []
    
    first_tc_1 = tc_1_values[0]  # Store the first tc_1 value
    for i, (time_array, tc_1) in enumerate(zip(times, tc_1_values)):
        #print(np.median(time_array))
        folded_time = time_array - tc_1
        folded_transits.append(folded_time)  # Append to list
        
        delta_t = tc_1 - first_tc_1  # Calculate Delta t
        delta_t_values.append(delta_t)
        
    return folded_transits, np.array(delta_t_values)


def extract_median(param_arrays, key):
    """Extracts the median of the first column of a parameter array."""
    return np.median(np.array(param_arrays[key])[:, 0])


def calculate_all_uncertainties(param_arrays, keys):
    """Calculates uncertainties for all given keys."""
    return [calculate_uncertainty(np.array(param_arrays[key])) for key in keys]


def calculate_median_stds(stds):
    """Calculate the median of each array in stds."""
    return np.array([np.median(std) if isinstance(std, np.ndarray) else std for std in stds])


def update_ferr_out(best_params, delta_t_values, f_out, t_out, ferr_out, models=None,
                    nu0=10.0, k_loc=0, rel_floor=1e-3):
    """Per-transit white-noise sigma, shrunk toward the pooled residual scatter.

    Replaces the raw per-transit RMS sqrt(RSS_j / N_j), which is a very noisy variance
    estimate for short (N_j ~ 5-15) transits and over/under-states the db/dt uncertainty when
    fed as a fixed MCMC weight.

    For each transit j: RSS_j = sum (model - flux)^2, N_j points, local dof
    d_j = max(N_j - k_loc, 1). With a pooled scatter s0^2 = sum RSS / sum N and a shrinkage
    pseudo-count nu0:
        s_j^2 = (nu0 * s0^2 + RSS_j) / (nu0 + d_j)
    so few-point transits are pulled toward s0 (stable) while well-sampled transits keep their
    own scatter. k_loc=0 by default (no per-transit baseline is fit); use 1 only if a local
    offset is ever added. nu0/k_loc/rel_floor are exposed for sensitivity checks.
    """
    rho, tc_1, p_1, k2_1, secw_1, sesw_1, q1, q2, b_0, db_dt = best_params
    rss = np.empty(len(t_out)); n = np.empty(len(t_out))
    for i in range(len(t_out)):
        b_1 = b_0 + db_dt * delta_t_values[i]
        b_1 = b_1[0] if isinstance(b_1, np.ndarray) and b_1.size == 1 else b_1
        pv = np.array([rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2])
        fmod = evaluate_transit_flux(pv, t_out[i], model=None if models is None else models[i])
        if np.shape(fmod) != np.shape(f_out[i]):
            raise ValueError(f"update_ferr_out: shape mismatch for transit {i}")  # BEFORE subtracting
        resid = fmod - f_out[i]
        if not np.all(np.isfinite(resid)):
            raise ValueError(f"update_ferr_out: nonfinite residual for transit {i}")  # don't poison s0
        rss[i] = np.sum(resid ** 2)
        n[i]   = len(f_out[i])
    s0_sq = rss.sum() / n.sum()                      # pooled scatter
    if not np.isfinite(s0_sq) or s0_sq <= 0:
        raise ValueError("update_ferr_out: nonfinite/zero pooled scatter s0")
    d = np.maximum(n - k_loc, 1.0)
    s_sq = (nu0 * s0_sq + rss) / (nu0 + d)
    s_sq = np.maximum(s_sq, (rel_floor * np.sqrt(s0_sq)) ** 2)   # floor vs pathological weights
    return list(np.sqrt(s_sq))


def optimize_global(log_prob_func, bounds, popsize=15, maxiter=5000):
    """
    Perform global optimization using Differential Evolution.
    
    Returns:
    - result: The optimization result containing the best parameters found.
    """
    result = differential_evolution(log_prob_func, bounds, popsize=popsize, maxiter=maxiter)
    return result


def perturb_initial_guess(pars, scale=1e-4):
    """
    Add small Gaussian noise to the initial parameter guesses.
    
    Parameters:
    - pars: Original parameter array
    - scale: Standard deviation of the Gaussian noise (default is 0.01)
    
    Returns:
    - perturbed_pars: New parameter array with added noise
    """
    noise = np.random.normal(0, scale, size=len(pars))
    perturbed_pars = pars * (1+noise)
    return perturbed_pars


def calculate_inverse_gamma_parameters(current_state, delta_t_values, f_out, t_out, means, stds):
    """
    Calculate the parameters of the inverse gamma distribution for ferr_out
    based on the residuals of the current model predictions and observed data.
    """
    # Prior parameters for the inverse gamma distribution
    alpha_prior = 1.0  # Example prior shape parameter
    beta_prior = 1.0  # Example prior scale parameter
    
    N = len(f_out)  # Number of observations for this transit

    # Calculate the sum of squared deviations (residuals)
    # Assuming evaluate_transit_flux function is available and returns the model predictions
    predicted_flux = evaluate_transit_flux(current_state, t_out)
    sum_squared_residuals = np.sum((f_out - predicted_flux) ** 2)

    # Update the parameters
    alpha_updated = alpha_prior + N / 2
    beta_updated = beta_prior + 0.5 * sum_squared_residuals

    return alpha_updated, beta_updated


def _legacy_de_bounds(flattened_means, flattened_stds):
    """DE search boxes for the synthetic / posterior-derived global-fit path -- kept verbatim
    (byte-identical) except the db_dt cap, which follows the posterior support widening
    (the old narrow cap literal is gone; see DB_DT_SUPPORT). Real KOIs use build_de_bounds."""
    bounds = []
    for i, (mean, std) in enumerate(zip(flattened_means, flattened_stds)):
        default_bound = (mean - 3*std, mean + 3*std)

        if i == 1:  # Second parameter (index 1)
            bounds.append((-1e-5, 1e-5))
        elif i == 4 or i == 5:  # Fifth or Sixth parameter
            # Use (-1e-4, 1e-4) if default bound is outside this range
            bounds.append((max(min(default_bound[0], -1e-4), -1e-4), min(max(default_bound[1], 1e-4), 1e-4)))
        elif i == 9:  # db_dt -- 3-sigma prior box capped to the (broad) posterior support;
                      # fall back to the support on an empty intersection.
            lo, hi = max(default_bound[0], -DB_DT_SUPPORT), min(default_bound[1], DB_DT_SUPPORT)
            bounds.append((lo, hi) if lo < hi else (-DB_DT_SUPPORT, DB_DT_SUPPORT))
        elif i == 8:  # b_0 -- cap to [0, 1] (posterior support), same intersection-with-fallback.
            lo, hi = max(default_bound[0], 0.0), min(default_bound[1], 1.0)
            bounds.append((lo, hi) if lo < hi else (0.0, 1.0))
        elif i in [6, 7]:  # Seventh or Eighth parameter
            # Ensure default bound does not outlarge (0.02, 0.98)
            bounds.append((max(min(default_bound[0], 0.02), 0.02), min(max(default_bound[1], 0.98), 0.98)))
        else:
            bounds.append(default_bound)
    return bounds


def build_de_bounds(means, stds, kinds):
    """DE search boxes for the catalog-driven (real-KOI) global fit, derived ONLY from the
    prior spec -- never from individual-fit posteriors: Normal -> mean +/- 3 sigma with the
    physical caps below; Uniform -> its full support (b_0 -> [0, 1], no data-dependence).
    Index map: 0 rho, 1 tc_1, 2 p_1, 3 k2_1, 4 secw_1, 5 sesw_1, 6 q1, 7 q2, 8 b_0, 9 db_dt.
    Unlike the legacy path there is no +/-1e-4 secw/sesw box: the DE likelihood must be able
    to explore eccentric geometry (3-sigma of the catalog prior, within the physical (-1,1))."""
    bounds = []
    for i, (mean, std, kind) in enumerate(zip(means, stds, kinds)):
        if kind == 'UP':
            bounds.append((float(mean), float(std)))   # (lower, upper)
            continue
        lo, hi = mean - 3 * std, mean + 3 * std
        if i == 1:                      # tc_1: hard support
            lo, hi = -1e-5, 1e-5
        elif i in (4, 5):               # secw/sesw: physical support
            lo, hi = max(lo, -1.0 + 1e-6), min(hi, 1.0 - 1e-6)
        elif i in (6, 7):               # q1/q2 (PyLDTk Normal): existing (0.02, 0.98) cap
            lo, hi = max(lo, 0.02), min(hi, 0.98)
            if lo >= hi:
                lo, hi = 0.02, 0.98
        elif i == 3:                    # k2: keep above the likelihood support floor
            lo = max(lo, 5e-6)
        elif i == 9:                    # db_dt: 3-sigma prior box inside the safety support
            lo, hi = max(lo, -DB_DT_SUPPORT), min(hi, DB_DT_SUPPORT)
        bounds.append((float(lo), float(hi)))
    return bounds


def find_mle_and_save(initial_guess, delta_t_values, f_out, t_out, ferr_out, means, stds, save_path=None, models=None, kinds=None):
    """
    Finds the Maximum Likelihood Estimate using the BFGS method and saves the results to a CSV file.
    `models`, if given, is an index-aligned list of prebuilt QuadraticModels reused across evaluations.
    ``kinds`` (from sipva_prior_spec) selects the catalog-derived DE boxes; None keeps the
    legacy synthetic-path boxes.
    """


    # Use scipy.optimize.minimize to find the MLE

    # Flatten means and stds if they are lists of arrays
    flattened_means = np.concatenate([np.array(mean).flatten() for mean in means])
    flattened_stds = np.concatenate([np.array(std).flatten() for std in stds])

    if kinds is not None:
        bounds = build_de_bounds(flattened_means, flattened_stds, kinds)
    else:
        bounds = _legacy_de_bounds(flattened_means, flattened_stds)


    result = differential_evolution(
        neg_log_likelihood,
        bounds=bounds,  # Ensure bounds are correctly defined
        args=(delta_t_values, f_out, t_out, ferr_out, means, stds, models),
        strategy='best1bin',
        maxiter=10000, 
        popsize=15,
        tol= 1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )

        
    if result.success:
        best_params = result.x
        print("Best Parameters:", best_params)
        
        # Differential Evolution does not provide uncertainties directly, so uncertainties may need to be estimated separately.
        # One common approach is to use bootstrapping or other resampling methods to estimate parameter uncertainties.
        
        # Saving results to CSV
        if save_path is not None:
            param_names = ['rho', 'tc_1', 'p_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY, 'b_0', 'db_dt']  # Updated parameter names
            mle_value = -neg_log_likelihood(best_params, delta_t_values, f_out, t_out, ferr_out, means, stds, models)
            # Since we don't have uncertainties directly from Differential Evolution, we'll save the parameters without uncertainties.
            df = pd.DataFrame({'Parameter': param_names + ['MLE'], 'Best Estimate': list(best_params) + [mle_value]})
            df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
    else:
        print("Optimization failed.")
        return None
    
    # Since uncertainties are not derived directly from Differential Evolution, you may return best_params without uncertainties here.
    return best_params


def sample_posterior(nwalkers, ndim, initial_guess, delta_t_values, f_out, t_out, ferr_out, means, stds, nsteps=20_0000, save_path=None, db_over_dt_enabled=True, models=None, ld_normal=False, kinds=None):
    """
    Sample from the posterior distribution using MCMC. ``ld_normal`` selects the q1/q2 prior
    (Normal within [0,1] for real KOIs, flat for synthetic runs); see ``log_prior``.
    ``kinds`` (from sipva_prior_spec, real KOIs) switches the DE boxes and the walker
    initialization to the catalog-derived path; None keeps the legacy behavior byte-identical.

    Returns:
    - sampler: The emcee sampler object containing the MCMC chain.

    """
    def log_prob_fn(params, delta_t_values, f_out, t_out, ferr_out, means, stds, models=None):
        # Compute the prior once. On an invalid prior, short-circuit but still return a
        # well-defined blob so emcee's blob array (saved as logprob_dbdt_values) stays intact.
        lp = log_prior(params, means, stds, friction=1, ld_normal=ld_normal, kinds=kinds)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        # Compute the db/dt log-likelihood exactly once and reuse it as the saved blob,
        # instead of evaluating it twice (once via logprob_with_prior, once for the blob).
        ll = logprob_dbdt(params, delta_t_values, f_out, t_out, ferr_out, means, stds, models)
        return lp + ll, ll  # (posterior log-prob, logprob_dbdt blob)

    # Flatten means and stds if they are lists of arrays
    flattened_means = np.concatenate([np.array(mean).flatten() for mean in means])
    flattened_stds = np.concatenate([np.array(std).flatten() for std in stds])

    if kinds is not None:
        bounds = build_de_bounds(flattened_means, flattened_stds, kinds)
    else:
        bounds = _legacy_de_bounds(flattened_means, flattened_stds)

    # If db_over_dt is not enabled, set its bounds to a very small range around 0
    if not db_over_dt_enabled:
        bounds[-1] = (-1e-9, 1e-9)

    # Repeat optimize_global a few times to find the best of the best_params.
    # Reduced from 5 to 3: combined with the cached transit models each restart is far
    # cheaper, and 3 independent DE runs still adequately seed the walkers.
    num_repeats = 3  # Number of times to repeat the optimization
    best_of_best_params = None
    best_of_best_score = np.inf

    for _ in range(num_repeats):
        result = optimize_global(lambda x: -logprob_with_prior(x, delta_t_values, f_out, t_out, ferr_out, means, stds, models, ld_normal=ld_normal, kinds=kinds), bounds)
        if result.fun < best_of_best_score:
            best_of_best_score = result.fun
            best_of_best_params = result.x

    # If every DE candidate was out of support, all runs return fun=inf and best_of_best_params
    # stays None -> the walker init below would dereference None. Fail with a clear message.
    if best_of_best_params is None or not np.isfinite(best_of_best_score):
        raise ValueError("sample_posterior: global optimization found no in-support point "
                         "(check b_0/db_dt seed and bounds)")

    # Initialize an empty array to hold the positions
    pos = np.zeros((nwalkers, ndim))

    if kinds is None:
        # Legacy (synthetic / posterior-derived) walker init, byte-identical.
        # Loop over each walker to initialize their positions
        for i in range(nwalkers):
            # For all but the last parameter, add random perturbation based on the prior stds
            pos[i, :-1] = best_of_best_params[:-1] + np.random.normal(0, stds[:-1], ndim - 1)
            # For the last parameter, use a fixed std of 1e-4 for the perturbation
            pos[i, -1] = best_of_best_params[-1] + np.random.normal(0, 1e-3)
    else:
        # Catalog-driven init: deterministic scales derived from the PRIOR SPEC only.
        # Initialization sets starting points; it cannot alter the posterior density.
        # NP -> 0.1 sigma; UP -> 0.1 (hi - lo) (covers b_0 and the q1/q2 PyLDTk-None
        # fallback); tc_1 -> 1e-8; db_dt -> 1e-3 and NEVER clipped. Positions are clipped
        # into hard supports only (UP dims and q1/q2), and any walker starting at -inf is
        # resampled around the DE best.
        init_scale = np.empty(ndim)
        for j in range(ndim):
            if kinds[j] == 'UP':
                init_scale[j] = 0.1 * (flattened_stds[j] - flattened_means[j])
            else:
                init_scale[j] = 0.1 * flattened_stds[j]
        init_scale[1] = 1e-8   # tc_1 (tight shared timing offset)
        init_scale[9] = 1e-3   # db_dt (matches the legacy jitter)

        for i in range(nwalkers):
            for _attempt in range(100):
                cand = best_of_best_params + np.random.normal(0, init_scale)
                for j in (6, 7):                      # q1/q2 hard support
                    cand[j] = np.clip(cand[j], 1e-3, 1.0 - 1e-3)
                for j, kd in enumerate(kinds):        # UP supports (b_0, fallback q1/q2)
                    if kd == 'UP':
                        cand[j] = np.clip(cand[j], flattened_means[j] + 1e-3,
                                          flattened_stds[j] - 1e-3)
                lp0, _ = log_prob_fn(cand, delta_t_values, f_out, t_out, ferr_out,
                                     means, stds, models)
                if np.isfinite(lp0):
                    pos[i] = cand
                    break
            else:
                raise ValueError(f"sample_posterior: walker {i} found no finite-probability "
                                 f"start in 100 tries around the DE best")

    # Ensure the last parameter is centered around zero with a std of 1e-8 for all chains
    #pos[:, -1] = np.random.normal(0, 1e-8, nwalkers)

    print(pos)
    # Initialize the sampler with the new log_prob_fn
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=(delta_t_values, f_out, t_out, ferr_out, means, stds, models))

    # Run MCMC
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Extract the logprob_dbdt values from the sampler's 'blobs'
    logprob_dbdt_values = sampler.get_blobs()

    # Save essential results if a save path is provided
    if save_path:
        results = {
            'chains': sampler.get_chain(),  # The chain for each walker
            'log_prob': sampler.get_log_prob(),  # Log probability values for each sample
            'logprob_dbdt_values': logprob_dbdt_values,  # Assuming you've calculated this separately
            'initial_positions': pos,  # Initial positions of the walkers
        }
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    return sampler


def analyze_mcmc_results(sampler, burn_in=2500):
    """
    Analyze the MCMC results to get the best-fitting parameters and errors.
    
    Parameters:
    - sampler: emcee sampler object
    - burn_in: Number of steps to discard as "burn-in" (default is 100_000)
    
    Returns:
    - samples: Flattened MCMC samples after burn-in
    - q50: Best-fitting parameters (medians)
    - final_err: Final errors for each parameter
    """
    
    # Get the number of dimensions from the sampler
    ndim = sampler.chain.shape[-1]
    
    # Discard the burn-in steps and reshape
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    
    # Calculate percentiles for each parameter
    q16 = np.percentile(samples, 16, axis=0)
    q50 = np.percentile(samples, 50, axis=0)
    q84 = np.percentile(samples, 84, axis=0)
    
    # Calculate errors
    lerr = q50 - q16
    uerr = q84 - q50
    
    # Determine the final error for each parameter
    final_err = np.maximum(lerr, uerr)
    
    return samples, q50, lerr, uerr


def construct_pv_list(param_arrays_2):
    pv_list = []
    num_transits = len(param_arrays_2['rho'])
    
    for i in range(num_transits):
        rho = param_arrays_2['rho'][i, 0]
        tc_1 = param_arrays_2['tc_1'][i, 0]
        p_1 = param_arrays_2['p_1'][i, 0]
        b_1 = param_arrays_2['b_1'][i, 0]
        k2_1 = param_arrays_2['k2_1'][i, 0]
        secw_1 = param_arrays_2['secw_1'][i, 0]
        sesw_1 = param_arrays_2['sesw_1'][i, 0]
        q1 = param_arrays_2[Q1_KEY][i, 0]
        q2 = param_arrays_2[Q2_KEY][i, 0]

        pv = [rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2]
        pv_list.append(pv)
        
    return pv_list


def _transit_center_and_depth(pv, segment_times, exp_cfg=None):
    """Mid-transit time and minimum model flux for one transit segment.

    The model flux is minimized over the segment window [min(t), max(t)] with a bounded scalar
    optimizer, which reproduces the former dense-grid argmin (formerly a 1e6-point np.linspace per
    transit -- ~8 GB resident for the largest synthetic systems) to better-than-grid precision at
    O(1) memory. evaluate_transit_flux squeezes single-point input to a 0-d array, hence the
    atleast_1d unwrap. ``exp_cfg`` = (nsamples, exptime_days) applies the same finite-exposure
    integration as the fitting likelihood, so the t_c,j located here belongs to the SAME model
    (None -> instantaneous, the synthetic/short-cadence path)."""
    tmin, tmax = float(np.min(segment_times)), float(np.max(segment_times))
    nsm = int(exp_cfg[0]) if exp_cfg is not None else 1
    expt = float(exp_cfg[1]) if exp_cfg is not None else 0.0
    res = minimize_scalar(
        lambda x: float(np.atleast_1d(evaluate_transit_flux(pv, np.array([x]),
                                                            exptime=expt, nsamples=nsm))[0]),
        bounds=(tmin, tmax), method='bounded')
    return float(res.x), float(res.fun)


def process_and_filter_data(param_arrays_2, times, fluxes, ferr_out1, threshold=0.5,
                            exp_list=None):
    """... ``exp_list``: optional index-aligned per-segment (nsamples, exptime_days) configs for
    the t_c,j locator; the caller filters its own copy with the returned outlier_indices."""
    pv_list = construct_pv_list(param_arrays_2)
    # Grid index kept only to preserve the return signature; it is unpacked but never read
    # downstream (pipeline.py), so a placeholder of zeros suffices.
    transit_mid_times_index_list = []
    transit_mid_times_list = []
    min_fluxes = []

    for k in range(len(times)):
        center_time, min_flux = _transit_center_and_depth(
            pv_list[k], times[k], exp_cfg=None if exp_list is None else exp_list[k])
        min_fluxes.append(min_flux)
        transit_mid_times_index_list.append(0)
        transit_mid_times_list.append(center_time)

    outlier_measure = np.ones(len(min_fluxes)) - min_fluxes
    max_measure = np.max(outlier_measure)
    outlier_threshold = threshold * max_measure
    
    outlier_indices = [idx for idx, val in enumerate(outlier_measure) if val < outlier_threshold]
    
    def filter_list_by_indices(original_list, indices_to_remove):
        return [item for idx, item in enumerate(original_list) if idx not in indices_to_remove]
        
    transit_mid_times_list = filter_list_by_indices(transit_mid_times_list, outlier_indices)
    transit_mid_times_index_list = filter_list_by_indices(transit_mid_times_index_list, outlier_indices)
    times = filter_list_by_indices(times, outlier_indices)
    fluxes = filter_list_by_indices(fluxes, outlier_indices)
    ferr_out1 = filter_list_by_indices(ferr_out1, outlier_indices)

    indices_with_multiple_values = []      # (previous tuple-based guard never matched; removed)

    return transit_mid_times_list, transit_mid_times_index_list, times, fluxes, ferr_out1, outlier_indices, indices_with_multiple_values


def filter_outliers_by_residual(param_arrays_2, times, fluxes, ferr_out1, exp_list=None):
    """10-sigma point-level residual clip against the individual-fit median model.
    ``exp_list``: optional per-segment (nsamples, exptime_days) so the reference model is the
    same finite-exposure-integrated one the likelihood uses (None -> instantaneous)."""
    pv_list = construct_pv_list(param_arrays_2)

    for k in range(len(times)):
        cfg = exp_list[k] if exp_list is not None else None
        simulate_flux = evaluate_transit_flux(
            pv_list[k], times[k],
            exptime=float(cfg[1]) if cfg is not None else 0.0,
            nsamples=int(cfg[0]) if cfg is not None else 1)
        residual = fluxes[k] - simulate_flux
        sigma = np.std(residual)
        
        mask = np.abs(residual) <= 10 * sigma
        
        times[k] = times[k][mask]
        fluxes[k] = fluxes[k][mask]
        ferr_out1[k] = ferr_out1[k][mask]
        
    return times, fluxes, ferr_out1


import os
import json
import numpy as np
import pandas as pd


def param_posterior_est(df, param, category, context=""):
    if str(category) == 'posterior':
        variable = df.posterior[param]
    if str(category) == 'derived_parameters':
         variable = df.derived_parameters[param]

    # Extract the variable values as a numpy array and flatten across chains
    samples = variable.values.flatten()

    # PyTransit's derived durations (t14/t23 via d_from_pkaiews) are NaN for any posterior sample
    # with b > 1 + k (a grazing/non-transiting draw). np.percentile propagates a single NaN to the
    # whole median, so an otherwise healthy transit silently vanishes from the duration regression.
    # Use nanpercentile, and read the NaN fraction only to decide whether the parameter is a genuine
    # non-measurement: > 50% non-transiting samples -> return None (excluded via the not_none path),
    # otherwise report the median over the transiting samples. The fraction is logged (not persisted)
    # so the audit trail survives without any schema change. context (KOI + segment index) makes the
    # line attributable from a process-pool worker.
    n_total = int(samples.size)
    n_nan = int(np.count_nonzero(~np.isfinite(samples)))
    if n_nan:
        frac = (n_nan / n_total) if n_total else 1.0
        label = f"{context} " if context else ""
        print(f"[param_posterior_est] {label}{param}: {100 * frac:.0f}% non-transiting samples "
              f"({n_nan}/{n_total}).")
        if frac > 0.5:
            return None

    # nanpercentile ignores the NaN tail above; for fully-finite samples it equals np.percentile.
    q16, q50, q84 = np.nanpercentile(samples, [16, 50, 84])
    lerr, uerr = q50 - q16, q84 - q50

    #print(f'{param} = {q50:.3f} +{uerr:.3f} -{lerr:.3f}')
    return q50, lerr, uerr


def print_rejection_rate(sampler):
    acceptance_fraction = sampler.acceptance_fraction  # Average acceptance fraction for each walker
    mean_acceptance = np.mean(acceptance_fraction)
    rejection_rate = 100 * (1.0 - mean_acceptance)
    print(f"Rejection Rate: {rejection_rate:.2f}%")




def calculate_and_print_uncertainty(params, special_params=None):
    uncen_dict = {}
    for key, array in params.items():
        if key in special_params:  # Handle special params differently
            array = np.array(array)
            uncen_value = calculate_uncertainty(array)
            print(f"The mean of {key}", np.mean(array[:, 0]))
        else:
            array = np.array(array)
            uncen_value = np.median(calculate_uncertainty(array))
            print(f"The mean of {key} uncen", uncen_value)
            print(f"The mean of {key}", np.median(array[:, 0]))

        uncen_dict[f"{key}_uncen"] = uncen_value

    return uncen_dict


def save_dict_to_csv(param_dict, koi, file_suffix):
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(param_dict)
    
    # Create an empty DataFrame to hold the new columns
    new_df = pd.DataFrame()
    
    # Loop through each column to split it into three new columns
    for col in df.columns:
        new_df[f"{col}_median"] = df[col].apply(lambda x: x[0])
        new_df[f"{col}_lerr"] = df[col].apply(lambda x: x[1])
        new_df[f"{col}_uerr"] = df[col].apply(lambda x: x[2])
    
    # Define the folder and file paths
    folder_name = _koi_output_folder(koi)
    csv_file_path = os.path.join(folder_name, f'param_arrays_{file_suffix}_koi_{koi}.csv')
    
    # Save the new DataFrame to a CSV file
    new_df.to_csv(csv_file_path, index=False)
    print(f"Dictionary saved to {csv_file_path}")


# Map internal global-fit parameter labels -> standard field symbols, and symbol -> unit.
# Used by save_parameters_csv. (`b` and `db_dt` come from best_params; `db_dt` is per-year.)
_PARAM_RENAME = {
    'rho': 'rho', 'tc_1': 'tc', 'p_1': 'p', 'k2_1': 'k2', 'secw_1': 'secw',
    'sesw_1': 'sesw', Q1_KEY: 'q1', Q2_KEY: 'q2', 'b_estim': 'b',
    'db_over_dt_estim': 'db_dt',
}
_PARAM_UNITS = {
    'rho': 'g/cm3', 'tc': 'day', 'p': 'day', 'k2': '', 'secw': '', 'sesw': '',
    'q1': '', 'q2': '', 'b': '', 'db_dt': '1/yr',
}


def output_root():
    """Root directory for per-KOI outputs. Defaults to ``../data/Output_data`` (cwd=scripts/);
    ``TDV_OUTPUT_ROOT`` overrides it so separate runs (e.g. GP-vs-savgol validation) can write to
    disjoint roots at write time instead of relying on a post-run ``mv``. Single source of truth --
    every per-KOI output path in this module and analysis.py goes through it."""
    return os.environ.get("TDV_OUTPUT_ROOT", os.path.join('..', 'data', 'Output_data'))


def _koi_output_folder(koi):
    folder_name = os.path.join(output_root(), f'koi-{koi}')
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_per_transit_csv(param_arrays_0, koi, transit_numbers, rho_consistent=None,
                         sibling_overlap_h=None, b_err_ratio=None):
    """Per-transit individual-fit results: one row per transit, a leading transit_number (physical
    orbital epoch) column, then each parameter's [median, lerr, uerr] triple expanded into
    <param>_median/_lerr/_uerr columns.

    ``rho_consistent`` (optional, in lockstep with the rows) is appended as a plain boolean column --
    passed as a separate aligned vector rather than a param_arrays_0 column, since every param_arrays
    column is expanded as a [median, lerr, uerr] triple above. False marks a transit excluded from the
    regressions and global fit for failing the rho-consistency check. All transits are written
    (nothing is dropped from the CSV).

    ``sibling_overlap_h`` and ``b_err_ratio`` (optional, aligned vectors) are real-KOI audit columns
    (see Components 1/5); synthetic callers leave them None so the synthetic CSV schema is unchanged.

    A parameter value may be ``None`` (a genuine non-measurement, e.g. a > 50%-non-transiting t14 from
    ``param_posterior_est``); its three cells are written blank rather than crashing the triple
    subscription."""
    df = pd.DataFrame.from_dict(param_arrays_0)
    new_df = pd.DataFrame()
    new_df['transit_number'] = list(transit_numbers)
    for col in df.columns:
        new_df[f"{col}_median"] = df[col].apply(lambda x: None if x is None else x[0])
        new_df[f"{col}_lerr"] = df[col].apply(lambda x: None if x is None else x[1])
        new_df[f"{col}_uerr"] = df[col].apply(lambda x: None if x is None else x[2])
    if rho_consistent is not None:
        new_df['rho_consistent'] = list(rho_consistent)
    if sibling_overlap_h is not None:
        new_df['sibling_overlap_h'] = list(sibling_overlap_h)
    if b_err_ratio is not None:
        new_df['b_err_ratio'] = list(b_err_ratio)

    csv_file_path = os.path.join(_koi_output_folder(koi), f'per_transit_fits_koi_{koi}.csv')
    new_df.to_csv(csv_file_path, index=False)
    print(f"Per-transit fits saved to {csv_file_path}")


def save_parameters_csv(koi, param_keys, values, lerrs, uerrs):
    """Global-fit parameters in clean long format: parameter,value,err_lower,err_upper,unit.
    param_keys are the internal labels (keys_extended); they are renamed to standard symbols and
    annotated with units."""
    rows = []
    for key, value, lerr, uerr in zip(param_keys, values, lerrs, uerrs):
        name = _PARAM_RENAME.get(key, key)
        rows.append({'parameter': name, 'value': value, 'err_lower': lerr,
                     'err_upper': uerr, 'unit': _PARAM_UNITS.get(name, '')})

    csv_file_path = os.path.join(_koi_output_folder(koi), f'parameters_koi_{koi}.csv')
    pd.DataFrame(rows, columns=['parameter', 'value', 'err_lower', 'err_upper', 'unit']).to_csv(
        csv_file_path, index=False)
    print(f"Parameters saved to {csv_file_path}")


def save_tdv_metrics_json(koi, metrics):
    """Scalar TDV metrics as JSON. All db/dt quantities are in 1/yr."""
    json_file_path = os.path.join(_koi_output_folder(koi), f'tdv_metrics_koi_{koi}.json')
    with open(json_file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"TDV metrics saved to {json_file_path}")


def save_posterior_samples(koi, samples, param_names, model_units, meta=None):
    """Persist the post-burn-in global (SIPVA) fit posterior so a corner plot can be remade later
    without rerunning the fit. Two artifacts under the per-KOI output dir:

      sipva_posterior_samples_koi_<koi>.npz -- the flattened post-burn-in chains (n_samples, n_param)
          stored VERBATIM in model units, in sampler order; plus ``param_names`` and ``model_units``.
          Any display-unit conversion (e.g. tc->seconds, k2->p) is the plotting layer's job, never
          applied here.
      sipva_posterior_meta_koi_<koi>.json -- human-readable metadata: parameter names/units, sampler
          settings, and the model-unit 16/50/84 percentiles for every parameter.

    Default-off in the pipeline (only the corner-plot driver passes save_posterior=True), so
    production/synthetic runs are byte-for-byte unaffected."""
    folder = _koi_output_folder(koi)
    samples = np.asarray(samples, dtype=float)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)

    npz_path = os.path.join(folder, f'sipva_posterior_samples_koi_{koi}.npz')
    np.savez_compressed(npz_path, samples=samples,
                        param_names=np.array(list(param_names)),
                        model_units=np.array(list(model_units)))

    meta_out = {
        'koi': koi,
        'param_names': list(param_names),
        'model_units': list(model_units),
        'n_samples': int(samples.shape[0]),
        'n_params': int(samples.shape[1]),
        'percentiles_model_units': {
            name: {'q16': float(a), 'q50': float(b), 'q84': float(c)}
            for name, a, b, c in zip(param_names, q16, q50, q84)
        },
    }
    if meta:
        meta_out.update(meta)
    json_path = os.path.join(folder, f'sipva_posterior_meta_koi_{koi}.json')
    with open(json_path, 'w') as f:
        json.dump(meta_out, f, indent=2)

    print(f"Posterior samples saved to {npz_path}")
    print(f"Posterior metadata saved to {json_path}")
    return npz_path


def save_param_arrays_2_to_csv(param_arrays_2, koi, file_suffix):
    # Convert the dictionary of NumPy arrays to a pandas DataFrame
    df = pd.DataFrame({key: list(value) for key, value in param_arrays_2.items()})
    
    # Create an empty DataFrame to hold the new columns
    new_df = pd.DataFrame()
    
    # Loop through each column to split it into three new columns
    for col in df.columns:
        new_df[f"{col}_median"] = df[col].apply(lambda x: x[0])
        new_df[f"{col}_lerr"] = df[col].apply(lambda x: x[1])
        new_df[f"{col}_uerr"] = df[col].apply(lambda x: x[2])
    
    # Define the folder and file paths
    folder_name = _koi_output_folder(koi)
    csv_file_path = os.path.join(folder_name, f'param_arrays_{file_suffix}_koi_{koi}.csv')
    
    # Save the new DataFrame to a CSV file
    new_df.to_csv(csv_file_path, index=False)
    print(f"Dictionary saved to {csv_file_path}")
