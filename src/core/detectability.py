#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""


import os
import csv
import math
import numpy as np
import pandas as pd
from scipy.constants import pi
from pytransit.utils.mocklc import create_mock_light_curve
from model import calculate_stellar_density
from priors import set_synthetic_priors


def f(i, b, p, period, db_over_dt):
    """Function to calculate (dF/dt*Period)**2"""
    # Normalized i to map into [lower_bound, upper_bound]
    lower_bound = (1 - p) ** 2 / b
    upper_bound = (1 + p) ** 2 / b
    center = (lower_bound + upper_bound) / 2
    length = upper_bound - lower_bound
    i = i / 2 * length + center

    z = np.sqrt(np.abs(b * i))  # Express z in terms of d and p

    if (1 - p) < z < (1 + p) and ((z ** 2 - 2 * z - p ** 2 + 1) * (z ** 2 + 2 * z - p ** 2 + 1) * z) != 0:
        term1 = -z * (z ** 2 - p ** 2 - 1) / (2 * np.sqrt(-z ** 4 + 2 * (p ** 2 + 1) * z ** 2 - (p ** 2 - 1) ** 2))
        term2 = -((p ** 2 / z) * (z ** 2 - p ** 2 + 1)) / np.sqrt(-(z ** 2 - 2 * p * z + p ** 2 - 1) * (z ** 2 + 2 * p * z + p ** 2 - 1))
        term3 = -(z ** 2 + p ** 2 - 1) / ((z ** 2 - 2 * z - p ** 2 + 1) * (z ** 2 + 2 * z - p ** 2 + 1) * z)
        k = p ** 2 / pi * (term1 + term2 + term3)
    else:
        k = 0

    return (k * period * db_over_dt) ** 2


f_vectorized = np.vectorize(f)


def integral_func(array, b, p, PERIOD, DB_OVER_DT):
    """Calculate integral of function f over the given array"""
    return np.sum(f_vectorized(array, b, p, PERIOD, DB_OVER_DT) * np.gradient(array)) / np.sum(np.gradient(array))


def snr_square(b, p, PERIOD, DB_OVER_DT, RS_OVER_A, NUM_TRANSITS, CADENCE_SC = 60, SIGMA_SC =1e-4 * 2 ):
    """Calculate square of Signal to Noise Ratio (SNR)"""
    TRANSIT_DUR = RS_OVER_A * PERIOD
    x = np.linspace(-1 - p, 1 + p, 100)
    integral = integral_func(x, b, p, PERIOD, DB_OVER_DT)
    sigma_f_squared = integral / (2 * (1 + p))
    return sigma_f_squared * (TRANSIT_DUR / CADENCE_SC) * NUM_TRANSITS ** 2 / (2 * SIGMA_SC ** 2)


def generate_light_curves(period, num_transit, db_over_dt, b, p, RS_OVER_A):
    rho = calculate_stellar_density(RS_OVER_A, period)
    times, fluxes = [], []
    
    for i in range(num_transit):
        time, flux, true_pars = create_mock_light_curve(texp=60,
                                                        passband='Kepler',
                                                        noise=1e-4 * 2,
                                                        transit_pars={'period': period,
                                                                      't0': 0 + period * i,
                                                                      'ror': p,
                                                                      'rho': rho,
                                                                      'b': b + db_over_dt * period * i / 365})
        times.append(time)
        fluxes.append(flux)
    
    return times, fluxes


def get_light_curves_from_csv(row_number, csv_filename):
    # Construct the full path to the CSV file
    parent_dir = os.path.abspath('..')
    data_dir = os.path.join(parent_dir, 'data', 'SNR_data')
    csv_path = os.path.join(data_dir, csv_filename)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract the parameters for the specified row
    params = df.iloc[row_number]
    period = params['PERIOD']
    num_transit = int(params['NUM_TRANSITS'])
    db_over_dt = params['DB_OVER_DT']
    b = params['b']
    p = params['p']
    RS_OVER_A = params['RS_OVER_A']
    
    # Use the previously defined generate_light_curves function to generate the light curves
    times, fluxes = generate_light_curves(period, num_transit, db_over_dt, b, p, RS_OVER_A)
    
    return times, fluxes


def get_planet_name(row_number, csv_filename):
    full_path = f'../data/SNR_data/{csv_filename}'  # Concatenate the base path with the filename
    try:
        with open(full_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            for i, row in enumerate(csvreader):
                if i == row_number:
                    return row[-1]  # Return the last column (planet name)
        return "Row number not found in the file."
    except FileNotFoundError:
        return "File not found."


def process_row_with_prior_model(ta_input, row_number, csv_filename):
    # Construct the full path to the CSV file
    parent_dir = os.path.abspath('..')
    data_dir = os.path.join(parent_dir, 'data', 'SNR_data')
    csv_path = os.path.join(data_dir, csv_filename)
    
    # Read the CSV file and get the specific row
    df = pd.read_csv(csv_path)
    params = df.iloc[row_number]
    
    # Extract parameters and calculate rho
    period = params['PERIOD']
    impact_param = params['b']
    planet_star_ratio = params['p']
    RS_OVER_A = params['RS_OVER_A']
    num_transits = int(params['NUM_TRANSITS'])
    rho = calculate_stellar_density(RS_OVER_A, period)

    # Add perturbations to the parameters
    period += np.random.normal(0, 0.01 * period)
    impact_param += np.random.normal(0, 0.01 * impact_param)
    planet_star_ratio += np.random.normal(0, 0.01 * planet_star_ratio)
    rho += np.random.normal(0, 0.01 * rho)
    
    # Call the set_synthetic_priors function with the perturbed parameters
    ta_output = set_synthetic_priors(ta_input, num_transits, period, impact_param, rho, planet_star_ratio)
    
    return ta_output


def _generate_model_light_curves(period, num_transit, b0, p, RS_OVER_A,
                                 db_over_dt, cadence_sec, passband='Kepler'):
    """
    Returns time, flux lists (one array per transit) using PyTransit model with limb darkening.
    Uses noise=0.0 so LLR compares pure models (noise level enters via SIGMA_SC).
    """
    rho = calculate_stellar_density(RS_OVER_A, period)  # g/cm^3 (as your function returns)
    times, fluxes = [], []
    for i in range(num_transit):
        bi = b0 + (db_over_dt * period * i / 365.0)  # same convention you used
        t0 = period * i
        t, f, _ = create_mock_light_curve(
            texp=cadence_sec,
            passband=passband,
            noise=0.0,  # <<< IMPORTANT: model-only flux
            transit_pars={
                'period': period,
                't0': t0,
                'ror': p,
                'rho': rho,
                'b': bi
            }
        )
        times.append(t)
        fluxes.append(f)
    return times, fluxes


def generate_light_curves_constant_b(period, num_transit, b, p, RS_OVER_A, cadence_sec):
    """Noise-free, constant-b version for F_{M0}."""
    return _generate_model_light_curves(period, num_transit, b, p, RS_OVER_A,
                                        db_over_dt=0.0, cadence_sec=cadence_sec)


def SNR_square_LD(b, p, PERIOD, DB_OVER_DT, RS_OVER_A, NUM_TRANSITS,
                  CADENCE_SC=60, SIGMA_SC=1e-4 * 2):
    """
    Limb-darkened version of SNR^2 (a.k.a. LLR^2):
        LLR^2 = sum_i ( (F_M1[i] - F_M0[i])^2 / SIGMA_SC^2 )

    - F_M1: model flux with an impact-parameter trend db/dt = DB_OVER_DT
    - F_M0: model flux with constant b
    - SIGMA_SC: per-sample photometric uncertainty (same units as flux), e.g., Kepler SC.
    - CADENCE_SC: sampling cadence (sec); also used as exposure time in the model.

    Returns
    -------
    float : LLR^2 (to get LLR, take np.sqrt of the return value)
    """
    # Model with trend (M1)
    _, fluxes_M1 = _generate_model_light_curves(PERIOD, NUM_TRANSITS, b, p, RS_OVER_A,
                                                db_over_dt=DB_OVER_DT, cadence_sec=CADENCE_SC)
    # Model without trend (M0)
    _, fluxes_M0 = generate_light_curves_constant_b(PERIOD, NUM_TRANSITS, b, p, RS_OVER_A,
                                                    cadence_sec=CADENCE_SC)

    # Accumulate LLR^2 across all samples of all transits
    denom = (SIGMA_SC ** 2)
    llr2 = 0.0
    for f1, f0 in zip(fluxes_M1, fluxes_M0):
        if f1.shape != f0.shape:
            raise ValueError("Shape mismatch between model arrays; check generator settings.")
        resid = f1 - f0
        llr2 += float(np.sum((resid * resid) / denom))
    return llr2


import numpy as np
from numpy import sin, cos, pi, sqrt, arcsin, arccos
from numba import njit
from model import evaluate_transit_flux, calculate_stellar_density

cache = False


def impact_parameter_ec(a, i, e, w, tr_sign):
    return a * cos(i) * ((1.-e**2) / (1.+tr_sign*e*sin(w)))


def af_transit(e, w):
    """Calculates the -- factor during the transit"""
    return (1.0-e**2)/(1.0 + e*sin(w))


def i_from_baew(b, a, e, w):
    """Orbital inclination from the impact parameter, scaled semi-major axis, eccentricity and argument of periastron

    Parameters
    ----------

      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]
      e  : eccentricity           [-]
      w  : argument of periastron [rad]

    Returns
    -------

      i  : inclination            [rad]
    """
    return arccos(b / (a*af_transit(e, w)))


def d_from_pkaiews(p, k, a, b, e, w, tr_sign, kind=14):
    """Transit duration (T14 or T23) from p, k, a, i, e, w, and the transit sign.

    Calculates the transit duration (T14) from the orbital period, planet-star radius ratio, scaled semi-major axis,
    orbital inclination, eccentricity, argument of periastron, and the sign of the transit (transit:1, eclipse: -1).

     Parameters
     ----------

       p  : orbital period         [d]
       k  : radius ratio           [R_Star]
       a  : scaled semi-major axis [R_star]
       i  : orbital inclination    [rad]
       e  : eccentricity           [-]
       w  : argument of periastron [rad]
       tr_sign : transit sign, 1 for a transit, -1 for an eclipse
       kind: either 14 for full transit duration or 23 for total transit duration

     Returns
     -------

       d  : transit duration T14  [d]
     """
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    ds = 1. if kind == 14 else -1.
    i = i_from_baew(b, a, e, w)
    return p/pi  * arcsin(sqrt((1.+ds*k)**2-b**2)/(a*sin(i))) * ae


def generate_synthetic_light_curves(period, num_transit, db_over_dt, b, p, RS_OVER_A, noise= 2e-4):

    rho = calculate_stellar_density(RS_OVER_A, period)
    p_1 = period
    k2_1 = p**2
    secw_1 = 0
    sesw_1 = 0
    q1 = 0.5
    q2 = 0.5

    times = []
    fluxes = []

    for i in range(num_transit):
        tc_1 = period * i   # transit center
        b_1 = b + db_over_dt * period * i / 365 # impact parameter
        transit_duration = d_from_pkaiews(p = period, k = p , a = 1/RS_OVER_A, b = b_1, e = 0, w = 0, tr_sign = 1, kind=14)
        pars_input =  rho, tc_1, p_1, b_1, k2_1, secw_1, sesw_1, q1, q2

        num_of_data_points = int(transit_duration * 24 * 60 / 1) # 1 min cadence
        if num_of_data_points < 0:
            num_of_data_points = -num_of_data_points

        time = np.linspace(tc_1 - transit_duration , tc_1 + transit_duration , 2*num_of_data_points)
        fluex = evaluate_transit_flux(pars_input, time)
        # Add noise
        noises = np.random.normal(0, noise, len(time))
        fluex += noises

        times.append(time)
        fluxes.append(fluex)
    
    return times, fluxes
