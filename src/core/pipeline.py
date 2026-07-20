#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""

import os

# Configure native-thread limits BEFORE numpy/numba/pytransit import. The per-transit
# ProcessPool workers must stay single-threaded: this account has RLIMIT_NPROC=300, and numba's
# default OpenMP/TBB threading layer would otherwise spawn a CPU-count-sized thread pool per
# worker and blow that limit. workqueue is numba's single-threaded, fork-safe layer.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import emcee
import numpy as np
from pytransit import create_mock_light_curve
from data import get_koi
from data import (get_light_curve, lightcurve_extract, get_transit_arrays,
                  select_cadence_per_transit, segment_coverage_ok, segment_baseline_ok,
                  product_exptimes, exposure_config)
from data import get_transit_ephemeris, get_sibling_ephemerides
from priors import koi_prior_spec, sipva_prior_spec
from limb_darkening import koi_ld_prior
from fitting import (run_transit_analysis, param_posterior_extract,
                     compute_keep_for_fit, analyze_multiple_transits, _fit_one_transit)
from analysis import Linear_regression, filter_nans, get_mid_transit
from analysis import plot_chains, plot_folded_transits, plot_db_dt_regression
from model import evaluate_transit_flux, build_transit_models, Q1_KEY, Q2_KEY
from model import calculate_uncertainty
from fitting import (save_per_transit_csv, save_parameters_csv, save_tdv_metrics_json,
                     save_param_arrays_2_to_csv, param_posterior_est, save_posterior_samples,
                     print_rejection_rate, calculate_and_print_uncertainty)
from fitting import (construct_pv_list,
                              fold_transits_and_calculate_delta_t, update_ferr_out,
                              process_and_filter_data, extract_median,
                              calculate_all_uncertainties, calculate_median_stds,
                              sample_posterior, analyze_mcmc_results, perturb_initial_guess,
                              filter_outliers_by_residual, find_mle_and_save)


# Minimum cadence points for a transit segment to be fittable. PyTransit's vectorized model
# evaluation shapes flux as (npop, npt); a single-point segment (npt==1) makes the per-pv model
# broadcast to (npop, npop) and the in-place update raises "non-broadcastable output operand"
# (observed on KOI 209.02, whose extraction yielded a 1-point transit at a gap edge). Segments this
# short carry no transit information anyway. Raise this for a stricter per-transit quality floor.
MIN_TRANSIT_POINTS = 2


def align_to_selection(values, surv, sel):
    """Selected-segment view of a per-candidate vector: the quality cuts keep candidate
    indices ``surv``, then select_cadence_per_transit picks ``sel`` among those survivors,
    so the selected segment j maps back to candidate ``surv[sel[j]]``. Single source of
    truth for aligning centers / exposures / audit vectors with the selected segments
    (exercised directly by tests/test_model_corrections.py)."""
    return [values[surv[j]] for j in sel]


def _n_workers(n_tasks):
    """Process count for the parallel individual fit. Honors the TDV_N_WORKERS env var; otherwise
    uses all available cores, always capped at the number of transits (never more workers than work)."""
    env = os.environ.get("TDV_N_WORKERS")
    if env:
        try:
            requested = int(env)
        except ValueError:
            requested = 0
        if requested > 0:
            return max(1, min(requested, n_tasks))
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    return max(1, min(avail, n_tasks))


def _mask_sibling_cadences(koi_str, eph, duration, times, fluxs, errs, siblings):
    """Remove cadences within +/-0.75*T14_sib of any sibling's predicted center, in place on
    times/fluxs/errs (kept in lockstep), BEFORE detrending (Component 1 / RC1). Both the GP and its
    savgol fallback then see sibling-free arrays, so neither baseline can be dragged by a sibling dip.

    Returns an audit dict (cadences removed, distinct target epochs affected, >30%-loss review list).
    Each masked (sibling, target-epoch) overlap is logged with the sibling-center provenance class so
    an interpolated/out-of-range O-C is not mistaken for a retained Holczer measurement.
    """
    from collections import defaultdict
    n_masked = 0
    before_counts, removed_counts = defaultdict(int), defaultdict(int)
    overlaps = {}  # (sibling KOI, target epoch) -> (source, sibling eph)

    for k in range(len(times)):
        tk = np.asarray(times[k], dtype=float)
        if tk.size == 0:
            continue
        toff = eph.center_offset(tk)
        in_win = np.abs(toff) < 1.0 * duration       # within the target's own +/-1*T14 window
        epk = eph.epoch_of(tk)

        sib_hit = np.zeros(tk.size, dtype=bool)
        for sib in siblings:
            hit = np.abs(sib["eph"].center_offset(tk)) < 0.75 * sib["t14_days"]
            sib_hit |= hit
            both = hit & in_win
            for e in np.unique(epk[both]):
                overlaps[(sib["koi"], int(e))] = (sib["source"], sib["eph"])

        for e in np.unique(epk[in_win]):
            m = in_win & (epk == e)
            before_counts[int(e)] += int(m.sum())
            removed_counts[int(e)] += int((m & sib_hit).sum())

        keep = ~sib_hit
        n_masked += int(sib_hit.sum())
        times[k] = tk[keep]
        fluxs[k] = np.asarray(fluxs[k], dtype=float)[keep]
        if errs is not None and errs[k] is not None:
            errs[k] = np.asarray(errs[k], dtype=float)[keep]

    affected = sorted(e for e, c in removed_counts.items() if c > 0)
    for (sib_koi, e), (src, sib_eph) in sorted(overlaps.items()):
        if removed_counts.get(e, 0) == 0:
            continue
        prov = sib_eph.oc_provenance(sib_eph.epoch_of(float(eph.predict(e))))
        print(f"[get_time_and_flux] KOI-{koi_str}: target epoch {e} overlapped by KOI-{sib_koi} "
              f"(sibling-center provenance: {prov})")

    review = [e for e in affected
              if before_counts[e] and removed_counts[e] / before_counts[e] > 0.30]
    if review:
        print(f"[get_time_and_flux] KOI-{koi_str}: epochs losing >30% of in-window cadences to "
              f"sibling masking (review): {review}")
    if n_masked:
        print(f"[get_time_and_flux] KOI-{koi_str}: sibling masking removed {n_masked} cadence(s) "
              f"across {len(affected)} target epoch(s).")
    return {"n_cadences_sibling_masked": int(n_masked),
            "epochs_sibling_affected": int(len(affected)),
            "sibling_review_epochs": review}


def get_time_and_flux(koi_number, detrend_method="gp"):

    # Find the target KOI.
    koi_number_str = str(koi_number)
    koi = get_koi(koi_number)
    name = '{}'.format(koi_number_str)
    is_LC, lcs, times, fluxs, errs = get_light_curve(name)

    # Per-product exposure times (days): lightkurve TIMEDEL metadata first, class constants
    # (by median sampling interval) as fallback. Computed on the raw download, then threaded
    # everywhere a transit model is evaluated (SS4 finite-exposure integration).
    exptimes_prod = product_exptimes(lcs, times)

    # Build the TTV-aware ephemeris from the raw download (Holczer measured times, else a
    # PyTransit center fit on these arrays, else linear) BEFORE lightcurve_extract mutates
    # `times`. Threaded through every step that previously assumed the linear t0 + n*P.
    eph = get_transit_ephemeris(koi_number, times, fluxs, is_LC, exptimes=exptimes_prod)
    duration = koi.koi_duration / 24.0   # target T14 in days

    # Component 1 (RC1) -- sibling-transit masking. Enumerate sibling KOIs on the same star and
    # remove cadences within +/-0.75*T14_sib of every sibling predicted center BEFORE detrending,
    # so a multi-planet system's sibling dip can neither drag the baseline nor be captured by the
    # single-planet per-transit fit. No siblings -> a no-op; the path is byte-identical to before.
    siblings = get_sibling_ephemerides(koi_number, times, fluxs, is_LC, exptimes=exptimes_prod)
    for sib in siblings:
        flag = "" if sib["source"] == "holczer2016" else "  [LOWER-CONFIDENCE ephemeris]"
        print(f"[get_time_and_flux] KOI-{koi_number_str}: sibling KOI-{sib['koi']} "
              f"(eph source={sib['source']}, T14={sib['t14_days']*24:.2f}h){flag}")
    sib_audit = ({"n_cadences_sibling_masked": 0, "epochs_sibling_affected": 0,
                  "sibling_review_epochs": []}
                 if not siblings
                 else _mask_sibling_cadences(koi_number_str, eph, duration, times, fluxs, errs,
                                             siblings))

    # detrend_method selects the per-quarter baseline estimator: "gp" (transit-masked Matern-3/2
    # GP, the default) or "savgol". errs feeds the GP's jitter prior; savgol ignores it.
    times, fluxs, transit_mask = lightcurve_extract(koi, is_LC, lcs, times, fluxs, eph,
                                                    errs=errs, method=detrend_method)

    #define the value of ootvs from get_SNR
    window = 21
    savgol_factor = (1. - ((3*(3*window**2. - 7))/(4*window*(window**2. - 4)))**(0.5))
    ootvs = np.asarray([1e6*np.std(fluxs[k][~transit_mask[k]]) for k in range(len(lcs))]) / savgol_factor
    # exp_cand: one scalar exposure time (days) per candidate segment, in lockstep with t_cand.
    t_cand, f_cand, ferr_cand, exp_cand = get_transit_arrays(times, fluxs, ootvs, is_LC, lcs,
                                                             koi, eph, exptimes=exptimes_prod)

    # Per-candidate-segment data-quality cuts (Components 2 + 3 + the pre-existing MIN_TRANSIT_POINTS
    # floor), evaluated on EVERY candidate BEFORE select_cadence_per_transit -- the ordering that
    # function's docstring already requires but get_time_and_flux previously violated. An epoch whose
    # short-cadence candidate fails then falls back to its long-cadence candidate instead of dropping.
    cov_frac = float(os.environ.get("TDV_COVERAGE_FRAC", "0.5"))
    min_in = int(os.environ.get("TDV_MIN_IN_TRANSIT", "3"))
    base_nsig = float(os.environ.get("TDV_BASELINE_NSIGMA", "5"))

    n_cand = len(t_cand)
    cand_center = [float(eph.predict(eph.epoch_of(np.median(s)))) for s in t_cand]
    cand_epoch = [int(eph.epoch_of(np.median(s))) for s in t_cand]
    cand_cad = [float(np.median(np.diff(np.sort(s)))) if len(s) > 1 else float('inf')
                for s in t_cand]
    cand_pass, cand_reason = [], []
    for i in range(n_cand):
        reason = None
        if len(t_cand[i]) < MIN_TRANSIT_POINTS:
            reason = "too_few_points"
        if reason is None:
            ok, r = segment_coverage_ok(t_cand[i], cand_center[i], duration, cov_frac, min_in)
            if not ok:
                reason = r
        if reason is None:
            ok, r, detail = segment_baseline_ok(t_cand[i], f_cand[i], ferr_cand[i],
                                                cand_center[i], duration, base_nsig)
            if not ok:
                reason = r
                print(f"[get_time_and_flux] KOI-{koi_number_str}: candidate epoch {cand_epoch[i]} "
                      f"(cad={cand_cad[i]*1440:.1f} min) flagged bad_baseline: {detail}")
        cand_pass.append(reason is None)
        cand_reason.append(reason)

    # Per-PHYSICAL-EPOCH counters (an epoch can carry both an SC and an LC candidate). A candidate
    # failure where another candidate of the same epoch survives is an SC->LC fallback, not a drop;
    # the no_coverage/bad_baseline/too_few drop counters tick only for epochs where ALL candidates
    # failed, attributed to the preferred (shortest-cadence) candidate's reason.
    from collections import defaultdict
    by_epoch = defaultdict(list)
    for i in range(n_cand):
        by_epoch[cand_epoch[i]].append(i)
    n_no_coverage = n_bad_baseline = n_too_few = n_sc_to_lc = 0
    for e, idxs in by_epoch.items():
        idxs_sorted = sorted(idxs, key=lambda i: cand_cad[i])   # shortest cadence first
        preferred = idxs_sorted[0]
        passing = [i for i in idxs_sorted if cand_pass[i]]
        if passing:
            if passing[0] != preferred and not cand_pass[preferred]:
                n_sc_to_lc += 1
                print(f"[get_time_and_flux] KOI-{koi_number_str}: epoch {e} short-cadence candidate "
                      f"failed ({cand_reason[preferred]}); fell back to longer-cadence candidate.")
        else:
            r = cand_reason[preferred]
            n_no_coverage += (r == "no_coverage")
            n_bad_baseline += (r == "bad_baseline")
            n_too_few += (r == "too_few_points")
            dt = np.asarray(t_cand[preferred], dtype=float) - cand_center[preferred]
            print(f"[get_time_and_flux] KOI-{koi_number_str}: epoch {e} dropped ({r}); "
                  f"dt span [{dt.min()/duration:+.2f}, {dt.max()/duration:+.2f}]*T14, "
                  f"{len(t_cand[preferred])} points.")

    # Survivors of the cuts, then collapse SC/LC duplicates per epoch (shortest cadence wins).
    surv = [i for i in range(n_cand) if cand_pass[i]]
    t_out1, f_out1, ferr_out1, sel = select_cadence_per_transit(
        koi_number,
        [t_cand[i] for i in surv], [f_cand[i] for i in surv], [ferr_cand[i] for i in surv], eph)
    final_idx = [surv[j] for j in sel]   # selected segment -> original candidate index

    # Per-transit predicted-center seeds for the tc_1 prior and per-segment exposure scalars,
    # in lockstep with the surviving segments (same surv->sel mapping, via the tested helper).
    centers = align_to_selection(cand_center, surv, sel)
    exp_out1 = align_to_selection(exp_cand, surv, sel)

    # sibling_overlap_h audit (Component 1 transparency): nearest sibling-center offset in hours per
    # surviving transit; None when the star has no siblings.
    if siblings:
        sibling_overlap_h = [min(abs(float(sib["eph"].center_offset(cand_center[i])))
                                 for sib in siblings) * 24.0 for i in final_idx]
    else:
        sibling_overlap_h = [None] * len(final_idx)

    prefit_audit = {
        "sibling_overlap_h": sibling_overlap_h,
        "n_cadences_sibling_masked": sib_audit["n_cadences_sibling_masked"],
        "epochs_sibling_affected": sib_audit["epochs_sibling_affected"],
        "n_transit_no_coverage": int(n_no_coverage),
        "n_transit_bad_baseline": int(n_bad_baseline),
        "n_transit_too_few_points": int(n_too_few),
        "n_sc_to_lc_fallback": int(n_sc_to_lc),
        "coverage_frac": cov_frac,
    }
    return t_out1, f_out1, ferr_out1, centers, exp_out1, prefit_audit


def TDV_fit(times, fluxes, koi_number, ferr_out, centers=None,
            prior_spec=None, ephemeris=None, detrend_method=None, prefit_audit=None,
            save_posterior=False, exptimes=None):
    """Two-stage TDV fit (per-transit individual fits, then a global db/dt fit).

    For real KOIs, leave ``prior_spec`` and ``ephemeris`` as None: the per-transit priors are
    fetched from the catalog (``koi_prior_spec``) and the transit-number basis comes from
    ``get_koi``. For synthetic/injection systems, pass ``prior_spec`` (e.g. from
    ``synthetic_prior_spec``) so the offset injection priors are used instead of catalog
    priors, and ``ephemeris=(t0, period)`` so transit numbering does not require a catalog
    lookup. ``koi_number`` is then used only as a label for outputs/plots.

    ``detrend_method`` is provenance only: when set (``"gp"``/``"savgol"`` from the real-KOI path)
    it is recorded in the per-KOI tdv_metrics JSON; synthetic direct callers leave it ``None`` and
    the field is omitted. It does not affect the fit (detrending already happened upstream).

    ``prefit_audit`` (real-KOI path only) carries the pre-fit data-cut bookkeeping from
    ``get_time_and_flux`` (sibling-overlap audit vector + drop counters). Default ``None`` keeps
    synthetic/injection callers signature-compatible and leaves the synthetic metrics/CSV schema
    unchanged: the audit columns and the prefit-derived metrics keys are emitted only when it is set.

    ``exptimes`` (real-KOI path): one scalar exposure time (days) per transit segment, in
    lockstep with ``times``; converted once to per-segment (nsamples, exptime) configs and
    applied to every model evaluation (individual fits, DE/MCMC likelihood models, the
    t_c,j locator, and the residual-outlier clip). Default ``None`` keeps every evaluation
    instantaneous -- the synthetic path is byte-identical.
    """


    """
    First Fit
    """
    
    # Initialize dictionaries to store parameters
    derived_parameters = ['t14_1', 't23_1']
    posterior_parameters = ['rho', 'tc_1', 'p_1', 'b_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY, 'wn_loge_0']
    
    param_arrays_0 = {key: [] for key in derived_parameters + posterior_parameters}
    

    num_transit = len(times)
    param_keys = derived_parameters + posterior_parameters
    seed_base = int(os.environ.get("TDV_SEED_BASE", "12345"))

    # Per-transit tc_1 prior centers (TTV-aware). None -> seed each at the segment median in the
    # worker, i.e. today's behavior. Must be one-per-transit, in lockstep with `times`.
    if centers is None:
        centers = [None] * num_transit
    assert len(centers) == num_transit, \
        f"centers/times length mismatch: {len(centers)} != {num_transit}"

    # Per-segment exposure configs (nsamples, exptime_days); None -> instantaneous everywhere.
    if exptimes is None:
        exp_cfgs = None
    else:
        assert len(exptimes) == num_transit, \
            f"exptimes/times length mismatch: {len(exptimes)} != {num_transit}"
        exp_cfgs = [exposure_config(e) for e in exptimes]

    # Priors are identical for every transit -- fetch/build the spec once and pass the plain
    # spec to the workers, which rebuild the TransitAnalysis locally. (A prebuilt
    # TransitAnalysis cannot be sent across the process boundary: its ParameterSet does not
    # survive pickling.) Catalog-derived by default; a caller-supplied prior_spec (e.g. the
    # synthetic offset priors) overrides it for injection studies.
    # Real KOIs (prior_spec is None) get catalog-derived priors, including the PyLDTk-based
    # Normal limb-darkening prior; synthetic/injection runs keep the uniform q1/q2 prior.
    catalog_priors = prior_spec is None
    if prior_spec is None:
        prior_spec = koi_prior_spec(koi_number)

    # Run the per-transit individual fits in parallel -- each transit is an independent MCMC.
    # executor.map preserves input order, so aggregation matches the previous sequential order.
    tasks = [(k, times[k], fluxes[k], koi_number, prior_spec, seed_base, param_keys, centers[k],
              None if exp_cfgs is None else exp_cfgs[k])
             for k in range(num_transit)]
    n_workers = _n_workers(num_transit)
    ctx = mp.get_context('fork')  # fork inherits the parent's compiled numba (fast startup)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        results = list(executor.map(_fit_one_transit, tasks))

    for extracted_params in results:
        for key, value in extracted_params.items():
            param_arrays_0[key].append(value)


    """
    Save First Fit
    """

    # Per-transit fits, tagged with the physical orbital epoch (same formula as
    # select_cadence_per_transit) from each transit's tc_1 median. The (t0, period) basis is
    # caller-supplied for synthetic systems (no catalog row); else read from the KOI catalog.
    if ephemeris is not None:
        t0_ref, period_ref = ephemeris
        rho_star = None                      # synthetic: no catalog density -> rho mask is a no-op
    else:
        koi = get_koi(koi_number)
        t0_ref, period_ref = koi.koi_time0bk, koi.koi_period
        rho_star = float(koi.koi_srho)
    transit_numbers = [int(round((tc[0] - t0_ref) / period_ref))
                       for tc in param_arrays_0['tc_1']]

    # Reject non-physical (low stellar-density) per-transit fits BEFORE they enter the regressions
    # and the global fit. One full-length keep_for_fit mask (rho-consistency for real KOIs, folding
    # in the old remove_none_entries None-drop) is applied to EVERY downstream array, so the
    # param_arrays, light-curve segments, and transit-number labels stay index-aligned -- and the
    # first KEPT transit (not an excluded one) anchors delta_t / b0_seed downstream. The full,
    # unmasked param_arrays_0 is still written to the per-transit CSV, with the rho_consistent flag.
    keep_for_fit, rho_consistent, reject_factor = compute_keep_for_fit(param_arrays_0, rho_star)

    # Component 5 (RC2 backstop, real-KOI only, flag-only): per-transit b_err_ratio =
    # sigma_posterior(b) / sigma_prior(b). A prior-returned fit (transit-less segment that slipped
    # the coverage cut) has ratio ~1; healthy fits with a tight catalog b prior can also run high, so
    # this is surfaced for review (n_prior_dominated = count with ratio > 0.8), never auto-excluded.
    # Gated on catalog_priors so synthetic runs (uniform/non-catalog b prior) keep their CSV schema.
    b_err_ratio = None
    n_prior_dominated = None
    if catalog_priors:
        sig_prior_b = next((float(b) for nm, _d, _a, b in prior_spec if nm == 'b_1'), None)
        if sig_prior_b and sig_prior_b > 0:
            b_err_ratio = [None if row is None else max(float(row[1]), float(row[2])) / sig_prior_b
                           for row in param_arrays_0['b_1']]
            n_prior_dominated = int(sum(1 for r in b_err_ratio if r is not None and r > 0.8))

    sib_overlap = prefit_audit["sibling_overlap_h"] if prefit_audit is not None else None
    save_per_transit_csv(param_arrays_0, str(koi_number), transit_numbers, rho_consistent,
                         sibling_overlap_h=sib_overlap, b_err_ratio=b_err_ratio)

    n_excluded = int(sum(1 for k in keep_for_fit if not k))
    flagged_epochs = [transit_numbers[i] for i, k in enumerate(keep_for_fit) if not k]
    if rho_star is not None:
        n_rho_flagged = int(sum(1 for rc in rho_consistent if not rc))
        print(f"[TDV_fit] KOI-{koi_number}: flagged {n_rho_flagged}/{num_transit} transit(s) as "
              f"rho-inconsistent (rho outside [srho/{reject_factor:g}, srho*{reject_factor:g}], "
              f"srho={rho_star:.4g}); excluded epochs {flagged_epochs}.")
        # Borderline KEPT transits (srho/F <= rho < srho/2): kept, but surfaced for manual review.
        borderline = [transit_numbers[i] for i, rc in enumerate(rho_consistent)
                      if rc and param_arrays_0['rho'][i] is not None
                      and float(param_arrays_0['rho'][i][0]) < rho_star / 2.0]
        if borderline:
            print(f"[TDV_fit] KOI-{koi_number}: borderline-low-rho transits kept for review "
                  f"(rho < srho/2): epochs {borderline}.")

    # Apply the single mask to all per-transit arrays (supersedes remove_none_entries for the fit
    # path). ferr_out1 is derived here from ferr_out, mirroring the old remove_none_entries return.
    keep_idx = [i for i, k in enumerate(keep_for_fit) if k]
    param_arrays_0 = {key: [val[i] for i in keep_idx] for key, val in param_arrays_0.items()}
    times = [times[i] for i in keep_idx]
    fluxes = [fluxes[i] for i in keep_idx]
    ferr_out1 = [ferr_out[i] for i in keep_idx]
    if exp_cfgs is not None:
        exp_cfgs = [exp_cfgs[i] for i in keep_idx]

    def convert_to_numpy_arrays(param_arrays):
        return {k: np.array(v) for k, v in param_arrays.items()}

    # Usage
    param_arrays_0_numpy = convert_to_numpy_arrays(param_arrays_0)
    
    times, fluxes, ferr_out1 = filter_outliers_by_residual(param_arrays_0_numpy, times, fluxes, ferr_out1,
                                                           exp_list=exp_cfgs)

     
    """
    Analysis the First Fit
    """
    

    # Extract duration arrays from the first fit
    duration_array_14 = np.array(param_arrays_0['t14_1'])
    duration_array_23 = np.array(param_arrays_0['t23_1'])
    b_1_array = np.array(param_arrays_0['b_1'])
    
    from analysis import filter_nans, get_mid_transit
    # Filter duration_array_14 and get valid indices
    filtered_array, valid_indices = filter_nans(duration_array_14)
    tc_1_array = np.array(param_arrays_0['tc_1'])
    x_array = tc_1_array[valid_indices][:,0]
    
    #transfer the duration from days to minutes
    duration_array_14_min = np.array(filtered_array)*24*60
    y_uncen_value = np.maximum(duration_array_14_min[:, 1], duration_array_14_min[:, 2])
    b_1_ucen = calculate_uncertainty(b_1_array)
    
    # Filter b_1_array and b_1_ucen using valid_indices
    filtered_b_1_array = b_1_array[valid_indices]
    filtered_b_1_ucen = b_1_ucen[valid_indices]
    
    #x_array = x_array - bkjd
    
    dur_panel = None
    if len(duration_array_14) > 4:
        coeff, coeff_error, intersec, intersec_error, _ = Linear_regression(x_mean=np.array(x_array),
                                                                                     y=duration_array_14_min[:,0],
                                                                                     y_uncen=y_uncen_value,
                                                                                     koi_number=koi_number,
                                                                                     plot=False,
                                                                                     Transit_duration_ = True)
        dur_panel = (np.array(x_array), duration_array_14_min[:,0], y_uncen_value,
                     (coeff, coeff_error, intersec, intersec_error))

    db_over_dt_estim, db_over_dt_error_estim, b_estim, b_estim_error, b_cov_slope_intercept = Linear_regression(tc_1_array[:,0],
                                                                                         b_1_array[:,0],
                                                                                         b_1_ucen,
                                                                                         koi_number = koi_number,
                                                                                         plot = False,
                                                                                         Transit_duration_ = False)

    # Per-system figures are skipped when TDV_MAKE_PLOTS=0 (set by the synthetic batch runner);
    # real-KOI runs leave it unset and still get plots.
    if os.environ.get("TDV_MAKE_PLOTS", "1") != "0":
        plot_db_dt_regression(koi_number,
                              tc_1_array[:,0], b_1_array[:,0], b_1_ucen,
                              (db_over_dt_estim, db_over_dt_error_estim, b_estim, b_estim_error),
                              dur_panel=dur_panel)
    

    # Convert to numpy arrays for filter_outliers_by_residual function
    param_arrays_0_numpy = convert_to_numpy_arrays(param_arrays_0)
    times, fluxes, ferr_out1 = filter_outliers_by_residual(param_arrays_0_numpy, times, fluxes, ferr_out1,
                                                           exp_list=exp_cfgs)
            
    t_score = db_over_dt_estim/db_over_dt_error_estim
    # coeff/coeff_error exist only when the duration regression ran (dur_panel is not None).
    t_score_t14 = (coeff / coeff_error) if dur_panel is not None else None
    
    """
    Fitting All Transit Together
    """
                                                                            
                                                                          
    transit_mid_times_list, transit_mid_times_index_list, filtered_times, filtered_fluxes, filtered_ferr_out1, outlier_indices, indices_with_multiple_values = process_and_filter_data(param_arrays_0_numpy, times, fluxes, ferr_out1, threshold=0.3, exp_list=exp_cfgs)
    # Drop the same outlier segments from the exposure configs (lockstep with filtered_times).
    if exp_cfgs is not None:
        exp_cfgs = [c for idx, c in enumerate(exp_cfgs) if idx not in outlier_indices]
    folded_transits, delta_t_values = fold_transits_and_calculate_delta_t(filtered_times, transit_mid_times_list)

    # Convert delta_t_values from days to years
    delta_t_values = delta_t_values/365
    
    if os.environ.get("TDV_MAKE_PLOTS", "1") != "0":
        plot_folded_transits(folded_transits, filtered_fluxes, koi = koi_number)

    # Build one QuadraticModel per folded transit and reuse it across the whole global fit.
    # folded_transits is final here (no further in-place masking), so these stay valid through
    # the MLE, ferr_out update, and MCMC. Reusing them skips per-call model construction.
    transit_models = build_transit_models(folded_transits, exp_list=exp_cfgs)

    """
    Initialized the Initial Guess of Second Fit
    """
    
    # Initialize dictionaries to store parameters
    derived_parameters = ['t14_1', 't23_1']
    posterior_parameters = ['rho', 'tc_1', 'p_1', 'b_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY, 'wn_loge_0']
    additional_parameters = ['db_dt_1']
    
    
    # Define keys and additional parameters
    keys = ['rho', 'tc_1', 'p_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY]

    if catalog_priors:
        # SS1 -- real KOIs: the SIPVA prior is CATALOG-ONLY (sipva_prior_spec). The
        # individual-fit posteriors and the b/t14 regression above are, from here on, used
        # ONLY as stage-one diagnostics (t_score_b, db_dt_linreg, plots) and the fixed
        # folding inputs (transit_mid_times_list). Explicit separation of concerns:
        #   prior spec        -> sipva_prior_spec (catalog + PyLDTk; b_0 ~ U(0,1))
        #   DE search boxes   -> build_de_bounds(pars, stds, kinds) inside fitting.py
        #   MCMC walker init  -> DE best + deterministic prior-derived scales (fitting.py)
        #   optimizer fallback-> the catalog prior centers (pars) if DE ever fails
        # For 'UP' entries (b_0; q1/q2 when no PyLDTk prior) pars/stds carry (lower, upper).
        gspec = sipva_prior_spec(koi_number)
        kinds = [d for _n, d, _a, _b in gspec]
        pars = [a for _n, _d, a, _b in gspec]
        stds = np.array([b for _n, _d, _a, b in gspec], dtype=float)
        ld_normal = (gspec[6][1] == 'NP')   # PyLDTk Normal available (else UP(0,1) fallback)
    else:
        # Synthetic / injection path: the legacy posterior-derived construction, unchanged.
        kinds = None
        # The global model defines b_0 as b at the FIRST folded transit (Delta t = 0), i.e. at
        # transit_mid_times_list[0], NOT at BKJD = 0 where Linear_regression reports b_estim. Seed
        # the b_0 prior mean by evaluating the fitted line at that reference epoch (a constant
        # x-shift, so the slope / t_score_b / db_dt_linreg are unchanged; only the separately
        # reported global db_dt_global_zscore shifts -- intended, since the b_0 seed/prior moves).
        b0_ref_time = transit_mid_times_list[0]
        b0_seed = b_estim + db_over_dt_estim * b0_ref_time
        # Propagate the line's uncertainty to that epoch (b_estim_error is the marginal intercept
        # error at BKJD = 0, not the width at the first transit):
        #   Var(b@t) = Var(B) + t^2 Var(A) + 2 t Cov(A, B).
        b0_var = (b_estim_error ** 2 + b0_ref_time ** 2 * db_over_dt_error_estim ** 2
                  + 2 * b0_ref_time * b_cov_slope_intercept)
        b0_seed_error = np.sqrt(max(b0_var, 0.0))
        additional_params = [b0_seed, db_over_dt_estim]
        additional_uncertainties = [b0_seed_error, db_over_dt_error_estim]

        # Extract medians and uncertainties
        medians = [extract_median(param_arrays_0_numpy, key) for key in keys] + additional_params
        uncertainties = calculate_all_uncertainties(param_arrays_0_numpy, keys) + additional_uncertainties

        # Calculate median of uncertainties and assign to descriptive variable names
        pars = medians
        stds = calculate_median_stds(uncertainties)

        #convert the unit of db/dt from days to years
        pars[-1] = pars[-1]*365
        stds[-1] = stds[-1]*365

        # we change the tc_1 intial value to the zero as we re-algin the time
        pars[1] = 0

        #We also the change std of tc_1 to 1e-6 so the alogrthims will not fit the transit epoch
        stds[1] = 1e-8

        #non - inform prior for db/dt
        pars[-1] = 0
        stds[-1] = 0.2

        #pars[-2] = 0.2
        #stds[-2] = 10

        # Synthetic runs keep the flat q1/q2 prior.
        ld_normal = False

    """
    Run the Second Fit
    """
    ferr_out1 = [2e-4] * len(filtered_fluxes) 
    
    result = find_mle_and_save(
        initial_guess=pars,
        delta_t_values=delta_t_values,
        f_out=filtered_fluxes,
        t_out=folded_transits,
        ferr_out=ferr_out1,
        means=pars,
        stds=stds,
        save_path=None,  # MLE point estimates not persisted
        models=transit_models,
        kinds=kinds
        )
    

    if result is None:
        print("Optimization failed, using initial parameters.")
        best_params = pars
    else:
        best_params = result
        print("Best Parameters:", best_params)
        # Additional code to save results and perform further operations as you described
    
    ferr_out_updated = update_ferr_out(best_params, delta_t_values, filtered_fluxes, folded_transits, ferr_out1, models=transit_models)

    print("ferr_out_updated", np.shape(ferr_out_updated))
    # MCMC length: production default 10000; TDV_MCMC_NSTEPS is a SMOKE-TEST override (short,
    # clearly non-production chains). The burn-in scales down with it so short chains still
    # yield a posterior (default nsteps leaves the production burn_in=2500 unchanged).
    nsteps_run = int(os.environ.get("TDV_MCMC_NSTEPS", "10000"))
    burn_in = min(2500, nsteps_run // 2)
    # Continue with the sampling process
    sampler = sample_posterior(
        nwalkers=32,
        ndim=10,  # Ensure this matches the updated dimensions
        initial_guess=pars,  # Ensure this is correctly prepared for the sampler
        delta_t_values=delta_t_values,
        f_out=filtered_fluxes,  # Assuming you want to use the entire array here
        t_out=folded_transits,
        ferr_out=ferr_out_updated,
        means=pars,  # This might need to be adjusted to ensure correct dimensions
        nsteps=nsteps_run,
        stds=stds,
        save_path=None,  # raw emcee chains not persisted
        models=transit_models,
        ld_normal=ld_normal,
        kinds=kinds
    )


    samples, best_params,lerrs, uerrs = analyze_mcmc_results(sampler, burn_in=burn_in)
    final_err = np.maximum(lerrs, uerrs)
    print("Best-fitting parameters (medians):", best_params)
    print("Final errors:", final_err)

    # Optional: persist the post-burn-in posterior so the corner plot can be remade without refitting.
    # Default-off; only the SIPVA corner-plot driver sets this. `samples` is already the flattened
    # post-burn-in chain in model units, in the global-fit sampler order below.
    if save_posterior:
        nwalkers_g, nsteps_g, _ = sampler.chain.shape
        meta = {'burn_in': int(burn_in), 'nwalkers': int(nwalkers_g), 'nsteps': int(nsteps_g),
                'seed_base': int(os.environ.get("TDV_SEED_BASE", "12345")),
                'detrend_method': detrend_method, 'fit': 'SIPVA global db/dt'}
        if kinds is not None:
            # Record the catalog prior and the complete hard support so no summary/corner
            # routine has to assume a legacy range.
            from fitting import DB_DT_SUPPORT
            meta['prior_spec'] = [list(t) for t in gspec]
            meta['hard_support'] = {
                'tc_1': [-1e-5, 1e-5], 'q1': [0.0, 1.0], 'q2': [0.0, 1.0],
                'b_0': [0.0, 1.0], 'b_j_per_epoch': [0.0, 1.0],
                'db_dt': [-DB_DT_SUPPORT, DB_DT_SUPPORT],
                'rho': [0.0, 100.0], 'k2_min': 5e-6,
                'geometry_rejection': 'e < 1, |cos i| <= 1, finite orbit scale',
            }
        save_posterior_samples(
            str(koi_number), samples,
            param_names=['rho', 'tc_1', 'p_1', 'k2_1', 'secw_1', 'sesw_1', Q1_KEY, Q2_KEY, 'b_0', 'db_dt'],
            model_units=['g/cm3', 'day', 'day', '', '', '', '', '', '', '1/yr'],
            meta=meta)

    db_over_dt_z_score = best_params[-1]/final_err[-1]
    print("db_over_dt_z_score", db_over_dt_z_score)
    

    """
    Save the Second Fit Result
    """
    keys_extended = keys + ['b_estim', 'db_over_dt_estim']

    # Global-fit parameters -> clean long-format CSV (standard symbols + units).
    save_parameters_csv(str(koi_number), keys_extended, best_params, lerrs, uerrs)

    # Scalar TDV metrics -> JSON. All db/dt in 1/yr: the global db/dt (best_params[-1]) is already
    # per-year; the linear-regression slope is per-day so it is multiplied by 365 here.
    # koi field stays numeric for real KOIs; synthetic labels (e.g. "syn_snr10_000") are
    # kept as strings rather than crashing float().
    try:
        koi_field = float(koi_number)
    except (TypeError, ValueError):
        koi_field = str(koi_number)
    metrics = {
        'koi': koi_field,
        'num_transit': int(num_transit),
        'n_transit_excluded': n_excluded,
        'num_transit_used_global': int(num_transit - n_excluded),
        'flagged_epochs': flagged_epochs,
        'rho_reject_factor': (float(reject_factor) if reject_factor is not None else None),
        'db_dt_global': float(best_params[-1]),
        'db_dt_global_err': float(final_err[-1]),
        'db_dt_global_zscore': float(db_over_dt_z_score),
        'db_dt_linreg': float(db_over_dt_estim * 365),
        'db_dt_linreg_err': float(db_over_dt_error_estim * 365),
        't_score_b': float(t_score),
        't_score_t14': (float(t_score_t14) if t_score_t14 is not None else None),
    }
    # Provenance: record how this KOI was detrended. None (synthetic direct callers) -> field omitted.
    if detrend_method is not None:
        metrics['detrend_method'] = detrend_method
    # Pre-fit data-cut bookkeeping (real-KOI path only). Emitted only when prefit_audit is set, so
    # the synthetic metrics JSON schema is unchanged (keys omitted, not null).
    if prefit_audit is not None:
        for key in ('n_cadences_sibling_masked', 'epochs_sibling_affected', 'n_transit_no_coverage',
                    'n_transit_bad_baseline', 'n_transit_too_few_points', 'n_sc_to_lc_fallback',
                    'coverage_frac'):
            metrics[key] = prefit_audit[key]
    if n_prior_dominated is not None:
        metrics['n_prior_dominated'] = n_prior_dominated
    save_tdv_metrics_json(str(koi_number), metrics)


def execute_TDV_func(koi_number, detrend_method=None, save_posterior=False):
    # Resolve the detrending method: explicit arg -> TDV_DETREND env -> "gp" (the default).
    if detrend_method is None:
        detrend_method = os.environ.get("TDV_DETREND", "gp")
    try:
        t_out1, f_out1, ferr_out1, centers, exp_out1, prefit_audit = get_time_and_flux(koi_number, detrend_method)

        # SMOKE-TEST truncation (non-production): slice ALL aligned per-segment vectors
        # atomically, so centers / exposures / audit stay in lockstep with the segments.
        smoke_n = os.environ.get("TDV_SMOKE_MAX_TRANSITS")
        if smoke_n:
            n = max(1, int(smoke_n))
            t_out1, f_out1, ferr_out1 = t_out1[:n], f_out1[:n], ferr_out1[:n]
            centers, exp_out1 = centers[:n], exp_out1[:n]
            prefit_audit["sibling_overlap_h"] = prefit_audit["sibling_overlap_h"][:n]
            print(f"[execute_TDV_func] SMOKE MODE (TDV_SMOKE_MAX_TRANSITS={n}): truncated to "
                  f"{len(t_out1)} segment(s) -- NON-PRODUCTION run.")

        # The per-transit TransitAnalysis objects (and their tc_1 + KOI priors) are now built
        # inside the parallel workers in TDV_fit, so there is no need to construct them here.
        print(TDV_fit(t_out1, f_out1, koi_number, ferr_out1, centers,
                      detrend_method=detrend_method, prefit_audit=prefit_audit,
                      save_posterior=save_posterior, exptimes=exp_out1))
        return True, None  # If everything goes well, return True
    except Exception as e:
        # This will print the type of error, the error message, and the stack trace
        error_msg = f"An error occurred: {e}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return False, error_msg
