#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def chi_squared(x, y, y_sigma, model, model_params, ddof = 0):
    statistic = np.sum(np.square((y - model(x, *model_params))/y_sigma))
    dof = len(model_params)
    return statistic, statistic/(len(y) - dof - ddof), len(y) - dof - ddof


def Linear_regression(x_mean, y, y_uncen, koi_number, plot=True, Transit_duration_=True):
    # Define the model
    model = lambda x, A, B: A * x + B
    p0 = [0, 0]

    # Perform curve fitting
    p, pcov = curve_fit(model, x_mean, y, p0=p0, sigma=y_uncen, absolute_sigma=True)
    p_sigma = np.sqrt(np.diagonal(pcov))

    # Generate fit data
    y_fit = model(x_mean, *p)

    # Calculate residuals
    residuals = y - model(x_mean, *p)

    # Perform significance test and chi-squared calculation
    # Assuming these functions are defined elsewhere
    test_coefficient_significance(p, p_sigma, residuals, len(p))
    chi, rchi, v = chi_squared(x_mean, y, y_uncen, model, p)

    if plot:
        plt.figure()
        from fitting import output_root   # single source of truth for the output root (TDV_OUTPUT_ROOT)
        name = "koi-" + str(koi_number)
        folder_name = os.path.join(output_root(), name)

        # Set caption, ylabel and filename based on Transit_duration_
        if Transit_duration_:
            caption = "Linear Regression of the Transit Duration of Koi " + str(koi_number)
            ylabel = "Transit Duration"
            filename = f"linear_regression_transit_duration_koi_{koi_number}.pdf"
        else:
            caption = "Linear Regression of the Impact Parameter of Koi " + str(koi_number)
            ylabel = "Impact Parameter"
            filename = f"linear_regression_impact_parameter_koi_{koi_number}.pdf"

        # Create a smooth set of x values for plotting
        x_smooth = np.linspace(min(x_mean), max(x_mean), 1000)
        y_smooth = model(x_smooth, *p)
        
        # Coefficients over standard error
        coef_over_stderr = p[0] / p_sigma[0]

        # Calculate R^2 value
        ssr = np.sum((y - y_fit) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ssr / sst)
        
        t_value = coef_over_stderr

        
        # Plotting
        plt.plot(x_smooth, y_smooth, label=r"Fitline (t-value = {:.2f}, $R^2 = {:.2f}$)".format(t_value, r_squared))
        plt.errorbar(x_mean, y, y_uncen, fmt="v", label="TDV Uncen")
        plt.legend()
        plt.figtext(.5, -0.1, s="Caption: " + caption, ha='center')
        plt.ylabel(ylabel)
        plt.xlabel("Transit Time (Day)")
        
        # Create directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save figure
        plt.savefig(os.path.join(folder_name, filename))
        plt.show()

    # Also return cov(slope, intercept) = pcov[0, 1] so callers can propagate the line's
    # uncertainty to an arbitrary epoch: Var(A*t + B) = t^2 Var(A) + Var(B) + 2t Cov(A, B).
    return p[0], p_sigma[0], p[1], p_sigma[1], pcov[0, 1]


def filter_nans(duration_array):
    """Filters out tuples in the array where the q50 value is nan.
       Also returns the indices of the valid values.
    """
    valid_indices = [i for i, tup in enumerate(duration_array) if not np.isnan(tup[0])]
    filtered_array = [duration_array[i] for i in valid_indices]
    
    return filtered_array, valid_indices


def get_mid_transit(t_out1, valid_indices):
    """Get mid transit times using only valid indices."""
    x_array = []
    for i in valid_indices:
        x_array.append(np.median(t_out1[i]))
    return x_array


def filter_data(x, y, y_uncen, threshold_factor=0.75):
    # Calculate the mean of the uncertainty
    mean_uncertainty = np.mean(y_uncen)
    
    # Find the indices where uncertainty is below the threshold
    valid_indices = y_uncen < threshold_factor * mean_uncertainty
    
    # Filter the x, y, and y_uncen arrays according to the valid_indices
    x_filtered = x[valid_indices]
    y_filtered = y[valid_indices]
    y_uncen_filtered = y_uncen[valid_indices]

    return x_filtered, y_filtered, y_uncen_filtered


def drop_outliers(x, y, y_uncen, num_std=3):

    x, y, y_uncen = np.array(x), np.array(y), np.array(y_uncen)
    median_y = np.median(y)
    std_y = np.std(y)
    
    inliers_mask = (y > median_y - num_std*std_y) & (y < median_y + num_std*std_y)
    print(inliers_mask)
    x_clean = x[inliers_mask]
    y_clean = y[inliers_mask]
    y_uncen_clean = y_uncen[inliers_mask]
    
    return x_clean, y_clean, y_uncen_clean


def test_coefficient_significance(p, p_sigma, residuals, num_parameters):
    # curve_fit(absolute_sigma=True) already returns the proper coefficient SE in p_sigma,
    # so the t-statistic is simply coefficient / its standard error.
    t_values = p / p_sigma

    # Degrees of freedom
    df = len(residuals) - num_parameters

    # Calculate the critical t-value (two-tailed test)
    critical_t_value = stats.t.ppf(1 - 0.05 / 2, df)

    # Check if each coefficient is statistically significant
    for idx, t_value in enumerate(t_values):
        if abs(t_value) > critical_t_value:
            print(f"Coefficient {idx} is statistically significant.")
        else:
            print(f"Coefficient {idx} is not statistically significant.")


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.pyplot import subplots, setp
from numpy import percentile


# --- Publication (ApJ) styling -------------------------------------------------
# Serif + Computer-Modern mathtext (no usetex, so it works without a LaTeX
# install on the compute node). Applied locally via `with apj_rc():` so we don't
# mutate global rcParams for the rest of the pipeline.
_APJ_RC = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'savefig.dpi': 600,
}


def apj_rc():
    """Context manager applying the ApJ figure style locally."""
    return mpl.rc_context(_APJ_RC)


def _save_fig(fig, koi, stem):
    """Write a figure as both 600-dpi PNG and vector PDF under the KOI output dir."""
    from fitting import output_root   # single source of truth for the output root (TDV_OUTPUT_ROOT)
    folder = os.path.join(output_root(), f'koi-{koi}')
    os.makedirs(folder, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(folder, f'{stem}.{ext}'), bbox_inches='tight')
    plt.close(fig)


def plot_chains(ta):
    #plot the chains
    with sb.axes_style('white'):
        fig, axs = subplots(2,4, figsize=(13,5), sharex=True)
        ls, lc = ['-','--','--'], ['k', '0.5', '0.5']
        percs = [percentile(ta.sampler.chain[:,:,i], [50,16,84], 0) for i in range(8)]
        [axs.flat[i].plot(ta.sampler.chain[:,:,i].T, 'k', alpha=0.01) for i in range(8)]
        [[axs.flat[i].plot(percs[i][j], c=lc[j], ls=ls[j]) for j in range(3)] for i in range(8)]
        setp(axs, yticks=[])
        fig.tight_layout()


def plot_folded_transits(folded_transits, fluxes, koi):
    """Folded light curve, points colored by transit epoch (single-column ApJ)."""
    folded_transits = list(folded_transits)
    n = len(folded_transits)
    cmap = plt.get_cmap('viridis')

    with apj_rc():
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for i, folded_time in enumerate(folded_transits):
            ax.scatter(folded_time, fluxes[i], s=4, edgecolors='none', alpha=0.7,
                       color=cmap(i / max(n - 1, 1)))

        ax.set_xlabel('Time from mid-transit [days]')
        ax.set_ylabel('Normalized flux')

        sm = mpl.cm.ScalarMappable(cmap=cmap,
                                   norm=mpl.colors.Normalize(vmin=0, vmax=max(n - 1, 1)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Transit number')

        _save_fig(fig, koi, f'folded_transits_koi_{koi}')


def _regression_panel(ax, x, y, yerr, fit, ylabel):
    """Draw one impact-parameter / duration regression panel."""
    x, y, yerr = np.asarray(x), np.asarray(y), np.asarray(yerr)
    slope, slope_err, intercept, _ = fit

    t_value = slope / slope_err if slope_err else np.nan
    y_fit = slope * x + intercept
    ssr = np.sum((y - y_fit) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ssr / sst if sst else np.nan

    xs = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xs, slope * xs + intercept, color='C3',
            label=r'fit ($t={:.2f}$, $R^2={:.2f}$)'.format(t_value, r_squared))
    ax.errorbar(x, y, yerr, fmt='o', ms=3, color='C0', ecolor='0.6',
                elinewidth=0.8, capsize=1.5, label='measurements')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')


def plot_db_dt_regression(koi, b_x, b_y, b_yerr, b_fit, dur_panel=None):
    """Stacked two-panel regression: impact parameter (top) + transit duration (bottom).

    `b_fit` and the fit in `dur_panel` are (slope, slope_err, intercept, intercept_err).
    `dur_panel` is (x, y, yerr, fit) or None (e.g. when too few transits for the duration
    regression) -- in which case only the impact-parameter panel is drawn.
    """
    with apj_rc():
        if dur_panel is not None:
            fig, (ax_b, ax_d) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 4.8))
            _regression_panel(ax_b, b_x, b_y, b_yerr, b_fit, r'Impact parameter $b$')
            dx, dy, dyerr, dfit = dur_panel
            _regression_panel(ax_d, dx, dy, dyerr, dfit, 'Transit duration [min]')
            ax_d.set_xlabel('Transit time [BKJD, days]')
            fig.align_ylabels([ax_b, ax_d])
            fig.subplots_adjust(hspace=0.08)
        else:
            fig, ax_b = plt.subplots(figsize=(3.5, 2.6))
            _regression_panel(ax_b, b_x, b_y, b_yerr, b_fit, r'Impact parameter $b$')
            ax_b.set_xlabel('Transit time [BKJD, days]')

        _save_fig(fig, koi, f'linear_regression_koi_{koi}')
