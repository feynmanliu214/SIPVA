#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""

import numpy as np
from oblate_model import compute_spherical_transit_lightcurve


def logprob(pars, f_out, t_out, ferr_out, cad_out, ttv_array):
    ''' Returns scalar chi-squared value for transit model given by vector pars and empirical
     transit fluxs f_out and times t_out. TTV_array should be a vector with same dimension
     as the number of transits. If fit_ttv = 1, this applies a shift in transit timing
     as given by TTV_array. If fit_ttv = 0, no TTV shift is applied'''
    t_0, b_0, period, Rx, f, obliquity, u1, u2, log10_rho_star  = pars
    N_data = len(np.concatenate(t_out))
    logprob = 0.

    if np.abs(b_0) > 1.0:
        return 1e6
    if np.min([u1, u2]) < 0:
        return 1e6
    if u1+u2 > 1:
        return 1e6
    if Rx < 0.:
        return 1e6
    
    t_oot = np.array([(t_out[i] - t_0 + period/2.) % period - period/2. for i in range(len(t_out))])

    for i in range(len(t_out)):
        fmod_pars = [0.0, b_0, period, Rx, f, obliquity, u1, u2, log10_rho_star]
        fmod = compute_spherical_transit_lightcurve(fmod_pars, t_oot[i] - ttv_array[i], exp_time=cad_out[i][0]/60./24.)
        if np.max(fmod) == np.nan:
            logprob = np.inf
        else:
            logprob += 0.5*np.sum(((fmod-f_out[i])/(ferr_out[i]))**2)
    #print(logprob/N_data)
    return logprob/N_data


def logprob_TTV(ttv_array, pars, f_out, t_out, ferr_out, cad_out):
    ''' Returns scalar chi-squared value for transit model given by vector pars and empirical
     transit fluxs f_out and times t_out. TTV_array should be a vector with same dimension
     as the number of transits. For each transit, the time is off-set by the corresponding number
     in ttv_array.'''
    t_0, b_0, period, Rx, f, obliquity, u1, u2, log10_rho_star  = pars
    N_data = len(np.concatenate(t_out))
    logprob = 0.

    if np.abs(b_0) > 1.0:
        return 1e6
    if np.min([u1, u2]) < 0:
        return 1e6
    if u1+u2 > 1:
        return 1e6
    if Rx < 0.:
        return 1e6

    t_oot = np.array([(t_out[i] - t_0 + period/2.) % period - period/2. for i in range(len(t_out))])

    for i in range(len(t_out)):
        fmod_pars = [0.0, b_0, period, Rx, f, obliquity, u1, u2, log10_rho_star]
        fmod = compute_spherical_transit_lightcurve(fmod_pars, t_oot[i] - ttv_array[i], exp_time=cad_out[i][0]/60./24.)
        if np.max(fmod) == np.nan:
            logprob = np.inf
        else:
            logprob += 0.5*np.sum(((fmod-f_out[i])/(ferr_out[i]))**2)
    #print(logprob/N_data)
    return logprob/N_data


def logprob_dbdt(pars_8, f_out, t_out, ferr_out, cad_out, ttv_array):
    ''' Returns scalar chi-squared value for transit model given by vector pars and empirical
     transit fluxs f_out and times t_out. TTV_array should be a vector with same dimension
     as the number of transits. If fit_ttv = 1, this applies a shift in transit timing
     as given by TTV_array. If fit_ttv = 0, no TTV shift is applied'''
    t_0, b_0, b_dot, period, Rx, f, obliquity, u1, u2, log10_rho_star  = pars_8
    N_data = len(np.concatenate(t_out))
    logprob = 0.

    t_oot = np.array([(t_out[i] - t_0 + period/2.) % period - period/2. for i in range(len(t_out))])
    if np.abs(b_0) > 1.0:
        return 1e6
    if np.min([u1, u2]) < 0:
        return 1e6
    if u1+u2 > 1:
        return 1e6
    if Rx < 0.:
        return 1e6

    for i in range(len(t_out)):
        b_t = b_0 + b_dot*(np.mean(t_out[i]) - np.mean(t_out[0]))/365.
        fmod_pars = [0.0, b_t, period, Rx, f, obliquity, u1, u2, log10_rho_star]
        fmod = compute_spherical_transit_lightcurve(fmod_pars, t_oot[i] - ttv_array[i], exp_time=cad_out[i][0]/60./24.)
        if np.max(fmod) == np.nan:
            logprob = np.inf
        else:
            logprob += 0.5*np.sum(((fmod-f_out[i])/(ferr_out[i]))**2)
    #print(logprob/N_data)
    return logprob/N_data


def hess_diag(best, func, h, f_out, t_out, ferr_out, cad_out, ttv_array):
## [f(x+h) - 2f(x) + f(x-h)]/h^2
    dim = len(best)
    res = np.zeros(dim)
    for ii in range(dim):
        best[ii] = best[ii] + h
        A = func(best, f_out, t_out, ferr_out, cad_out, ttv_array)
        best[ii] = best[ii] - h
        B = func(best, f_out, t_out, ferr_out, cad_out, ttv_array)
        best[ii] = best[ii] - h
        C = func(best, f_out, t_out, ferr_out, cad_out, ttv_array)
        best[ii] = best[ii] + h
        res[ii] = (A-2*B+C)/h**2.

    return res
