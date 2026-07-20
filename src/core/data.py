#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu"""


import numpy as np
from astropy.units import Quantity
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


# Per-product preflight floors for the GP detrender. Below these a per-quarter GP baseline is
# ill-constrained, so lightcurve_extract falls back to savgol for that product (see lightcurve_extract).
MIN_SEG_POINTS = 5    # savgol needs an odd window > polyorder(3), i.e. >= 5 samples
MIN_OOT_POINTS = 10   # out-of-transit points needed to pin the GP baseline


class _KOIRecord:
    """kplr-like wrapper over one NASA Exoplanet Archive KOI table row."""
    def __init__(self, row):
        object.__setattr__(self, "_row", row)

    def __getattr__(self, name):
        if name not in self._row.colnames:
            raise AttributeError(name)
        val = self._row[name]
        if isinstance(val, Quantity):           # strip units (d, h, ...) -> number
            val = val.value
        if val is np.ma.masked or np.ma.is_masked(val):
            return None
        try:
            fval = float(val)
            return None if np.isnan(fval) else fval
        except (TypeError, ValueError):
            return val


def _format_kepoi_name(koi_number):
    """841.02 -> 'K00841.02' (archive kepoi_name format)."""
    integer, frac = f"{float(koi_number):.2f}".split(".")
    return f"K{int(integer):05d}.{frac}"


def get_koi(koi_number):
    """Fetch a KOI record from the NASA Exoplanet Archive.

    Primary source is the homogeneous DR25 KOI table (``q1_q17_dr25_koi``) used for the catalog
    priors. A few confirmed KOIs have no DR25 row -- notably strong-TTV systems like KOI-142.01
    (Kepler-88) and KOI-377.02 (Kepler-9c), which DR25's fixed-ephemeris pipeline dropped; for
    those we fall back to the ``cumulative`` KOI table so they can still be fit. Cumulative
    parameters/errors differ slightly from DR25, so priors for these targets are not strictly
    homogeneous with the rest of a sample.
    """
    name = _format_kepoi_name(koi_number)
    for table in ("q1_q17_dr25_koi", "cumulative"):
        tbl = NasaExoplanetArchive.query_criteria(
            table=table, select="*", where=f"kepoi_name='{name}'")
        if len(tbl) > 0:
            return _KOIRecord(tbl[0])
    raise ValueError(f"No KOI row found for {name} (tried q1_q17_dr25_koi, cumulative)")


def get_all_kois():
    """Fetch all KOI records from the NASA Exoplanet Archive DR25 KOI table.

    Replacement for the old `kplr.API().kois()` bulk fetch.
    """
    tbl = NasaExoplanetArchive.query_criteria(table="q1_q17_dr25_koi", select="*")
    return [_KOIRecord(tbl[i]) for i in range(len(tbl))]


import os
import numpy as np



# Holczer mid-times are BJD-2454900; the pipeline / lightkurve use BKJD = BJD-2454833.
# Adding this constant converts a Holczer time to BKJD. (Verified 2026-06-06: agrees with
# koi_time0bk to ~1.3 min on KOI-139.01.)
_HOLCZER_OFFSET = 67.0

# Repo-root-anchored cache (NOT cwd-relative: entry points run from scripts/, so a bare
# "data/..." would land in scripts/data). src/core/transit_times.py -> ../../data.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "data", "ttv_catalog")
_T2_PATH = os.path.join(_CACHE_DIR, "holczer2016_table2.ecsv")
_T3_PATH = os.path.join(_CACHE_DIR, "holczer2016_table3.ecsv")

# In-process memo of the two cached tables, so repeated KOI lookups read disk once.
_HOLCZER_TABLES = {}


class TransitEphemeris:
    """Maps transit epoch (integer N) -> predicted mid-transit time in BKJD, TTV-aware.

    A single self-consistent linear basis ``(t0, period)`` underlies ``predict``, ``epoch_of``
    and any epoch-based grouping, plus an optional measured O-C(epoch) curve (days) that is
    linearly interpolated across gaps and is zero outside the measured range.
    """

    def __init__(self, t0, period, oc_epochs=None, oc_minutes=None, source="linear"):
        self.t0 = float(t0)
        self.period = float(period)
        self._source = source
        if oc_epochs is not None and len(oc_epochs) > 0:
            order = np.argsort(np.asarray(oc_epochs, dtype=float))
            self._oc_n = np.asarray(oc_epochs, dtype=float)[order]
            self._oc_d = np.asarray(oc_minutes, dtype=float)[order] / 1440.0  # min -> days
        else:
            self._oc_n = None
            self._oc_d = None

    @property
    def source(self):
        return self._source

    def _oc(self, epoch):
        """O-C (days) at integer epoch(s); 0 outside the measured range / if no measurements."""
        epoch = np.asarray(epoch, dtype=float)
        if self._oc_n is None:
            return np.zeros_like(epoch)
        oc = np.interp(epoch, self._oc_n, self._oc_d)  # clamps to edge values...
        outside = (epoch < self._oc_n[0]) | (epoch > self._oc_n[-1])
        return np.where(outside, 0.0, oc)               # ...so zero them explicitly

    def predict(self, epoch):
        """Predicted mid-transit time(s) in BKJD for integer epoch(s) N: t0 + N*P + OC(N)."""
        scalar = np.ndim(epoch) == 0
        n = np.atleast_1d(np.asarray(epoch, dtype=float))
        out = self.t0 + n * self.period + self._oc(n)
        return out[0] if scalar else out

    def _candidate_centers(self, times):
        """Predicted centers for every epoch spanning ``times`` (ascending: P >> max|OC|)."""
        n_lo = int(np.floor((np.min(times) - self.t0) / self.period)) - 2
        n_hi = int(np.ceil((np.max(times) - self.t0) / self.period)) + 2
        epochs = np.arange(n_lo, n_hi + 1)
        return epochs, self.predict(epochs)

    def _nearest(self, times):
        """For each time, the (center, epoch) of the nearest predicted transit."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        if t.size == 0:
            return np.array([]), np.array([], dtype=int)
        epochs, centers = self._candidate_centers(t)
        pos = np.clip(np.searchsorted(centers, t), 1, len(centers) - 1)
        left_closer = (t - centers[pos - 1]) <= (centers[pos] - t)
        sel = np.where(left_closer, pos - 1, pos)
        return centers[sel], epochs[sel]

    def center_offset(self, times):
        """times - nearest predicted center. Drop-in replacement for the old ``t_oot``."""
        scalar = np.ndim(times) == 0
        t = np.atleast_1d(np.asarray(times, dtype=float))
        centers, _ = self._nearest(t)
        out = t - centers
        return out[0] if scalar else out

    def epoch_of(self, times):
        """Integer epoch N of the nearest predicted transit for each time."""
        scalar = np.ndim(times) == 0
        t = np.atleast_1d(np.asarray(times, dtype=float))
        _, epochs = self._nearest(t)
        return int(epochs[0]) if scalar else epochs

    def oc_provenance(self, epoch):
        """Provenance class of the O-C used at integer ``epoch``: one of ``holczer_measured`` /
        ``holczer_interpolated`` / ``outside_holczer_range`` / ``pytransit_fit`` / ``linear``.

        The Holczer loader drops ``Out``/``Over`` rows before building the O-C curve, so at exactly
        the overlap epochs being sibling-masked the ``holczer2016`` source often means an
        *interpolated* O-C rather than a retained measurement. This keeps that confidence
        distinction visible instead of overstating it as a measured center."""
        if self._source == "pytransit_fit":
            return "pytransit_fit"
        if self._oc_n is None or self._source == "linear":
            return "linear"
        # holczer2016 (or any source carrying a measured O-C curve)
        e = int(round(float(epoch)))
        if e < self._oc_n[0] or e > self._oc_n[-1]:
            return "outside_holczer_range"
        if np.any(np.isclose(self._oc_n, e)):
            return "holczer_measured"
        return "holczer_interpolated"


# --------------------------------------------------------------------------------------------
# Holczer 2016 catalog: download-once cache + per-KOI local lookup
# --------------------------------------------------------------------------------------------

def _ensure_holczer_cache():
    """Download the full Holczer table2 + table3 once and cache them as two ECSV files."""
    if os.path.exists(_T2_PATH) and os.path.exists(_T3_PATH):
        return
    from astroquery.vizier import Vizier
    os.makedirs(_CACHE_DIR, exist_ok=True)
    v = Vizier(columns=["**"])
    v.ROW_LIMIT = -1  # full tables, no KOI constraint
    t2 = v.get_catalogs("J/ApJS/225/9/table2")[0]
    t3 = v.get_catalogs("J/ApJS/225/9/table3")[0]
    t2.write(_T2_PATH, format="ascii.ecsv", overwrite=True)
    t3.write(_T3_PATH, format="ascii.ecsv", overwrite=True)
    print(f"[transit_times] cached Holczer 2016 catalog ({len(t2)} KOIs, {len(t3)} transits) "
          f"to {_CACHE_DIR}")


def _holczer_tables():
    if not _HOLCZER_TABLES:
        from astropy.table import Table
        _ensure_holczer_cache()
        _HOLCZER_TABLES["table2"] = Table.read(_T2_PATH, format="ascii.ecsv")
        _HOLCZER_TABLES["table3"] = Table.read(_T3_PATH, format="ascii.ecsv")
    return _HOLCZER_TABLES["table2"], _HOLCZER_TABLES["table3"]


def _koi_match_mask(col, key):
    """Boolean mask of rows whose KOI equals ``key`` (canonical two-decimal string).

    Normalizes both sides through ``float`` formatting so '841.1', '841.10', and 841.1 all match
    the canonical '841.10'.
    """
    out = np.zeros(len(col), dtype=bool)
    for i, v in enumerate(col):
        try:
            out[i] = (f"{float(v):.2f}" == key)
        except (TypeError, ValueError):
            out[i] = (str(v).strip() == key)
    return out


def _float_col(table, name, fill):
    """Column ``name`` as a float ndarray, masked entries replaced by ``fill``."""
    col = table[name]
    if hasattr(col, "filled"):
        col = col.filled(fill)
    return np.asarray(col, dtype=float)


def _load_holczer_for_koi(koi_number):
    """Return ``(t0_bkjd, period, oc_epochs, oc_minutes)`` for a KOI, or ``None`` if absent.

    Applies Holczer's quality flags: drops outlier (``Out``), overlap (``Over``) and
    ``f_O-C``-flagged transits, plus any with non-finite epoch/O-C. Dropped epochs simply fall
    through to the O-C interpolation like any other gap.
    """
    key = f"{float(koi_number):.2f}"
    t2, t3 = _holczer_tables()

    m2 = _koi_match_mask(t2["KOI"], key)
    if not m2.any():
        return None
    row2 = t2[m2][0]
    t0_bkjd = float(row2["T0"]) + _HOLCZER_OFFSET
    period = float(row2["Per"])

    sub = t3[_koi_match_mask(t3["KOI"], key)]
    if len(sub) == 0:
        return None

    n = _float_col(sub, "N", np.nan)
    oc = _float_col(sub, "O-C", np.nan)  # minutes
    keep = np.isfinite(n) & np.isfinite(oc)
    if "Out" in sub.colnames:
        keep &= (_float_col(sub, "Out", 0.0) == 0)
    if "Over" in sub.colnames:
        keep &= (_float_col(sub, "Over", 0.0) == 0)
    if "f_O-C" in sub.colnames:
        flag = np.array([str(x).strip() for x in sub["f_O-C"]])
        flagged = ~np.isin(flag, ["", "--", "nan", "None", "0"])  # '*' marks unreliable O-C
        keep &= ~flagged

    dropped = int((~keep).sum())
    if dropped:
        print(f"[transit_times] KOI-{key}: dropped {dropped}/{len(sub)} Holczer transits "
              f"(Out/Over/f_O-C/non-finite).")
    if keep.sum() == 0:
        return None
    return t0_bkjd, period, n[keep].astype(int), oc[keep]


# --------------------------------------------------------------------------------------------
# Fallback: PyTransit per-transit center fit (KOI not in Holczer)
# --------------------------------------------------------------------------------------------

def _fallback_fit_centers(koi, times, fluxs, exptimes=None):
    """Measure per-transit centers by a bounded 1-D least-squares fit against PyTransit.

    Coarse->fine bootstrap: a *wide* coarse window (not +/-duration, which is exactly what fails
    for strong-TTV targets) localizes each epoch, then ``minimize_scalar`` finds the mid-time
    that best matches a fixed-shape QuadraticModel. Returns ``(oc_epochs, oc_minutes)`` or
    ``None`` if too few transits could be fit.

    The fixed shape uses the eccentric mid-transit geometry from the catalog (e, omega) --
    circular when the catalog carries none or the geometry is invalid -- and, when the
    per-product ``exptimes`` list (days, aligned with ``times``) is supplied, each epoch is
    evaluated with finite-exposure integration on the shortest *fit-worthy* cadence subset
    (quality first: an SC subset below the 5-point floor falls back to the LC subset; an epoch
    with neither is skipped, exactly as before).
    """
    try:
        from pytransit import QuadraticModel
        from pytransit.orbits import i_from_ba, i_from_baew
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover - dependency guard
        print(f"[transit_times] fallback fit unavailable ({exc}); using linear ephemeris.")
        return None

    t0 = float(koi.koi_time0bk)
    period = float(koi.koi_period)
    dur = float(koi.koi_duration or 3.0) / 24.0           # days
    k = np.sqrt(max(float(koi.koi_depth or 100.0), 1.0) * 1e-6)
    aor = float(koi.koi_dor or (period / (np.pi * max(dur, 1e-3))))
    b = float(koi.koi_impact or 0.0)
    # Eccentric mid-transit inclination from the catalog (e, omega); circular when the catalog
    # has none, e is out of range, or the eccentric geometry is invalid (|cos i| > 1) -- this
    # is a coarse center locator, so an unusable catalog geometry degrades gracefully.
    ecc = float(koi.koi_eccen or 0.0)
    w = np.deg2rad(float(koi.koi_longp)) if koi.koi_longp is not None else 0.0
    if 0.0 < ecc < 1.0 and abs((b / aor) * (1.0 + ecc * np.sin(w)) / (1.0 - ecc ** 2)) <= 1.0:
        inc = i_from_baew(b, aor, ecc, w)
    else:
        ecc, w = 0.0, 0.0
        inc = i_from_ba(b, aor)
    ldc = [0.4, 0.3]                                       # generic quadratic limb darkening

    allt = np.concatenate([np.asarray(t, dtype=float) for t in times])
    allf = np.concatenate([np.asarray(f, dtype=float) for f in fluxs])
    # Per-point exposure rides the same concatenation + sort so product identity survives.
    allexp = None
    if exptimes is not None:
        allexp = np.concatenate([np.full(len(np.asarray(t)), float(e))
                                 for t, e in zip(times, exptimes)])
    order = np.argsort(allt)
    allt, allf = allt[order], allf[order]
    if allexp is not None:
        allexp = allexp[order]

    coarse = min(0.25 * period, 3.0 * dur)
    n_lo = int(np.floor((allt.min() - t0) / period))
    n_hi = int(np.ceil((allt.max() - t0) / period))
    qm = QuadraticModel()

    oc_n, oc_m = [], []
    for nn in range(n_lo, n_hi + 1):
        c_lin = t0 + nn * period
        sel = np.abs(allt - c_lin) < coarse
        if sel.sum() < 5:
            continue
        if allexp is None:
            ts = allt[sel]
            fs = allf[sel] / np.median(allf[sel])
            qm.set_data(ts)
        else:
            # Quality-first cadence choice within the window, then finite-exposure
            # integration with the chosen subset's scalar config.
            chosen = _choose_cadence_subset(allt[sel], allf[sel], allexp[sel])
            if chosen is None:
                continue
            ts, fs, exp_med = chosen
            fs = fs / np.median(fs)
            nsm, expt = exposure_config(exp_med)
            if nsm > 1:
                qm.set_data(ts, nsamples=nsm, exptimes=expt)
            else:
                qm.set_data(ts)

        def chi2(tc):
            return np.sum((fs - qm.evaluate(k, ldc, tc, period, aor, inc, ecc, w)) ** 2)

        res = minimize_scalar(chi2, bounds=(c_lin - coarse, c_lin + coarse), method="bounded")
        tc = float(res.x)
        # reject fits pinned at the window edge (no real transit localized)
        if min(abs(tc - (c_lin - coarse)), abs(tc - (c_lin + coarse))) < 1e-6:
            continue
        oc_n.append(nn)
        oc_m.append((tc - c_lin) * 1440.0)  # minutes

    if len(oc_n) < 2:
        print(f"[transit_times] fallback fit found only {len(oc_n)} transit(s); "
              f"using linear ephemeris.")
        return None
    print(f"[transit_times] fallback fit measured {len(oc_n)} transit centers.")
    return np.array(oc_n, dtype=int), np.array(oc_m, dtype=float)


# --------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------

def get_transit_ephemeris(koi_number, times=None, fluxs=None, is_LC=None, exptimes=None):
    """Build a :class:`TransitEphemeris` for ``koi_number``.

    Holczer 2016 is tried first (needs only the KOI). If the KOI is absent and the downloaded
    light curve (``times``/``fluxs``) is supplied, a PyTransit center fit is attempted. Otherwise
    the linear ephemeris is returned. ``is_LC`` is accepted for interface symmetry but unused.
    ``exptimes`` (per-product exposure times in days, aligned with ``times``) enables
    finite-exposure integration inside the fallback center fit.
    """
    koi = get_koi(koi_number)

    # Escape hatch for A/B validation and debugging: force the old linear ephemeris.
    if os.environ.get("TDV_FORCE_LINEAR"):
        return TransitEphemeris(float(koi.koi_time0bk), float(koi.koi_period), source="linear")

    holczer = _load_holczer_for_koi(koi_number)
    if holczer is not None:
        t0_bkjd, period, oc_n, oc_m = holczer
        return TransitEphemeris(t0_bkjd, period, oc_n, oc_m, source="holczer2016")

    if times is not None and fluxs is not None and len(times) > 0:
        fit = _fallback_fit_centers(koi, times, fluxs, exptimes=exptimes)
        if fit is not None:
            oc_n, oc_m = fit
            return TransitEphemeris(float(koi.koi_time0bk), float(koi.koi_period),
                                    oc_n, oc_m, source="pytransit_fit")

    print(f"[transit_times] KOI-{koi_number}: no Holczer entry and no fallback fit; "
          f"using linear ephemeris.")
    return TransitEphemeris(float(koi.koi_time0bk), float(koi.koi_period), source="linear")


def _kepoi_name_to_number(name):
    """'K00841.02' -> '841.02' (the KOI-number string accepted by get_koi / Holczer lookups)."""
    integer, frac = str(name).strip().lstrip("K").split(".")
    return f"{int(integer)}.{frac}"


def _row_scalar(val):
    """One astropy table cell -> float, or None (strip Quantity units; masked/NaN -> None)."""
    if isinstance(val, Quantity):
        val = val.value
    if val is np.ma.masked or np.ma.is_masked(val):
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def get_sibling_ephemerides(koi_number, times=None, fluxs=None, is_LC=None, exptimes=None):
    """Sibling KOIs on the same star, each with a TTV-aware ephemeris and transit duration.

    One archive query against the ``cumulative`` KOI table (the superset that also carries
    cumulative-only KOIs) for ``kepoi_name like 'K00841.%'``, excluding the target itself and any
    ``FALSE POSITIVE`` disposition. Each surviving sibling gets an ephemeris from the usual fallback
    chain (Holczer -> PyTransit center fit -> linear; ``times``/``fluxs`` enable the fit fallback for
    a non-Holczer sibling). Returns a list of dicts ``{koi, eph, t14_days, source}``; empty when the
    star has no other non-FP KOI -- in which case the sibling-masking path is a no-op.
    """
    target_name = _format_kepoi_name(koi_number)
    star_prefix = target_name.split(".")[0]            # 'K00841'
    tbl = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select="kepoi_name,koi_disposition,koi_duration",
        where=f"kepoi_name like '{star_prefix}.%'")

    siblings = []
    for row in tbl:
        name = str(row["kepoi_name"]).strip()
        if name == target_name:
            continue
        disp_raw = row["koi_disposition"]
        disp = ("" if (disp_raw is None or np.ma.is_masked(disp_raw))
                else str(disp_raw).strip().upper())
        if disp == "FALSE POSITIVE":
            continue
        sib_koi = _kepoi_name_to_number(name)
        eph = get_transit_ephemeris(sib_koi, times, fluxs, is_LC, exptimes=exptimes)
        # T14 (days) sets the mask half-width: cumulative koi_duration (hours) where present, else
        # the sibling's own KOI record (get_transit_ephemeris has already fetched its row).
        dur_h = _row_scalar(row["koi_duration"])
        if dur_h is None or dur_h <= 0:
            dur_h = float(get_koi(sib_koi).koi_duration)
        siblings.append({"koi": sib_koi, "eph": eph,
                         "t14_days": float(dur_h) / 24.0, "source": eph.source})
    return siblings


import os
import math
import lightkurve as lk
import numpy as np
from scipy.signal import savgol_filter


# ---- Cadence / exposure conventions (SS4 finite-exposure integration) -----------------------
# Kepler cadence periods: short cadence 58.85 s, long cadence 1765.5 s (~29.42 min). Metadata
# (TIMEDEL, days) is preferred; these constants are the documented fallback, selected by the
# product's median sampling interval (< 5 min -> short cadence).
SC_EXPTIME_D = 58.85 / 86400.0
LC_EXPTIME_D = 1765.5 / 86400.0
SC_MAX_CADENCE_D = 5.0 / 1440.0       # cadence-class threshold: exposure/sampling < 5 min = SC
LC_NSAMPLES = 15                      # task-fixed long-cadence integration point count


def exposure_config(exptime_days):
    """Per-segment scalar exposure config for the transit models: (nsamples, exptime_days).
    Long cadence -> LC_NSAMPLES-point finite-exposure integration; short cadence -> midpoint
    evaluation (nsamples=1: output is independent of exptime, i.e. exactly the legacy
    instantaneous behavior). None -> None (no exposure information; legacy path)."""
    if exptime_days is None:
        return None
    e = float(exptime_days)
    return (LC_NSAMPLES, e) if e >= SC_MAX_CADENCE_D else (1, e)


def product_exptimes(lcs, times):
    """Per light-curve product exposure time in days, index-aligned with ``lcs``.

    Metadata first: ``lc.meta['TIMEDEL']`` (days) when present and finite. Fallback: classify
    by the product's median sampling interval (< 5 min -> SC_EXPTIME_D, else LC_EXPTIME_D)."""
    out = []
    for lc, t in zip(lcs, times):
        exp = None
        try:
            td = lc.meta.get('TIMEDEL')
            if td is not None and np.isfinite(float(td)) and float(td) > 0:
                exp = float(td)
        except Exception:
            exp = None
        if exp is None:
            t = np.asarray(t, dtype=float)
            cad = float(np.median(np.diff(np.sort(t)))) if t.size > 1 else LC_EXPTIME_D
            exp = SC_EXPTIME_D if cad < SC_MAX_CADENCE_D else LC_EXPTIME_D
        out.append(exp)
    return out


def _choose_cadence_subset(ts, fs, exps, min_points=5):
    """Quality-first cadence choice for one fallback-fitter epoch window: the shortest cadence
    class whose subset is still fit-worthy (>= min_points), SC before LC; None when neither
    qualifies (the epoch is skipped, exactly as the pre-exposure code skipped windows with
    fewer than 5 points). Returns (ts_sel, fs_sel, exptime_median) or None."""
    exps = np.asarray(exps, dtype=float)
    sc_m = exps < SC_MAX_CADENCE_D
    for m in (sc_m, ~sc_m):
        if int(m.sum()) >= int(min_points):
            return ts[m], fs[m], float(np.median(exps[m]))
    return None


def get_light_curve(name):
    # Initialize empty lists for return values
    all_times = []
    all_fluxes = []
    all_errs = []
    lcs = []
    is_LC = []

    try:
        # Search for light curves using the KOI number
        search_result = lk.search_lightcurve(f"KOI-{name}", mission="Kepler")

        # If there are no search results, return empty lists
        if len(search_result) == 0:
            print(f"No light curves found for KOI-{name}")
            return is_LC, lcs, all_times, all_fluxes, all_errs

        # Download the light curve files. Pin the ingest defaults explicitly rather than
        # relying on lightkurve's implicits: quality_bitmask='default' drops the worst-flagged
        # cadences (cosmic rays, momentum dumps, safe modes), and flux_column='pdcsap_flux'
        # selects the PDC-corrected flux. Both match lightkurve 2.x defaults today, so this is
        # behavior-preserving, but robust to a future default change.
        lcs_temp = search_result.download_all(quality_bitmask='default',
                                              flux_column='pdcsap_flux')

    except Exception as e:
        print(f"An error occurred: {e}")
        return is_LC, lcs, all_times, all_fluxes, all_errs

    # Iterate over the downloaded light curves
    for zz, lc in enumerate(lcs_temp):
        lcs.append(lc)
        is_LC.append('LC' in str(lc))
        time = lc.time.value
        flux = lc.flux.value
        # valid_indices is intentionally NaN-masked on (time, flux) ONLY -- the savgol path must
        # stay byte-identical, so flux_err must not change which cadences survive. flux_err rides
        # the same mask; its own NaN/<=0 entries are sanitized inside the GP detrender.
        flux_err = lc.flux_err.value
        valid_indices = ~np.isnan(time) & ~np.isnan(flux)
        all_times.append(time[valid_indices])
        all_fluxes.append(flux[valid_indices])
        all_errs.append(flux_err[valid_indices])

    return is_LC, lcs, all_times, all_fluxes, all_errs


def calculate_bic(y, y_pred, k):
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    bic = n * np.log(rss/n) + k * np.log(n)
    return bic


def _safe_savgol_window(window, n):
    """Clamp a savgol window to be valid for an n-sample product: odd, <= n, and > polyorder (3).
    A no-op for well-sampled products (window stays as computed); only engages on short/gappy
    products that would otherwise make savgol_filter raise."""
    w = min(int(window), int(n))
    if w % 2 == 0:
        w -= 1                      # savgol_filter requires an odd window
    return max(w, 5)                # > polyorder(3) and odd


def _savgol_detrend_product(koi, eph, duration, times, fluxs, transit_mask, k):
    """Polynomial-fill + iterative 5-sigma clip + savgol-divide detrending for one product, in place
    on times[k]/fluxs[k]. Byte-identical to the pre-GP pipeline on well-sampled products; the only
    additions are guards (window clamp, per-degree OOT-point check) that engage ONLY on degenerate
    (short/sparse) products that previously raised."""
    # A product shorter than a valid savgol window (odd, > polyorder 3, i.e. < 5 samples) cannot be
    # savgol-detrended at all (the old code crashed here). Median-normalize so it leaves this
    # function normalized rather than raising. Never triggers for real Kepler quarters (thousands of
    # points), so the well-sampled path stays byte-identical.
    if len(fluxs[k]) < MIN_SEG_POINTS:
        med = np.median(fluxs[k])
        fluxs[k] = np.asarray(fluxs[k], dtype=float) / (med if (np.isfinite(med) and med != 0) else 1.0)
        return
    flux_orig = fluxs[k]
    contains_transit = np.sum(transit_mask[k]) > 0.
    cut = 5.0  ## How many standard deviation (sigma) to remove for outliers
    ## Size the savgol window from the median cadence (robust to a leading gap, which the
    ## first-spacing-only np.diff(times[k])[0] would mis-size).
    cad_min = np.median(np.diff(np.sort(times[k]))) * 24. * 60.
    window = int(round(10 * (30. / cad_min)) * 2 + 1)
    window = _safe_savgol_window(window, len(fluxs[k]))

    ## If the data contains a transit, mask out the transit, do the polynomial fit, smooth it, then generate the smoothed lightcurve.
    ## Then, find datapoints that are more than 5.0 sigma from the running mean and remove them.
    if contains_transit:
        ## This is detrending part
        tran_times = times[k][transit_mask[k]]
        mask_indxs = np.where(transit_mask[k])[0]
        tran_times = times[k][mask_indxs]
        times[k] = np.delete(times[k], mask_indxs)
        fluxs[k] = np.delete(fluxs[k], mask_indxs)
        t_oot = eph.center_offset(times[k])

        # Inside your main function
        best_bic = np.inf
        best_degree = 0
        best_fit = None

        local = np.abs(t_oot) < 4.*duration
        n_local = int(np.sum(local))
        for degree in range(1, 7):  # Looping through degrees 1 to 6
            if n_local <= degree + 1:        # too few local OOT points to fit this degree
                continue
            polyfit = np.polyfit(times[k][local], fluxs[k][local], degree)
            p = np.poly1d(polyfit)
            y_pred = p(times[k][local])
            bic = calculate_bic(fluxs[k][local], y_pred, degree + 1)

            if bic < best_bic:
                best_bic = bic
                best_degree = degree
                best_fit = polyfit

        if best_fit is None:
            # No degree had enough local OOT points: fill the masked transit gap with the median
            # rather than crashing on np.poly1d(None).
            fill = np.median(fluxs[k]) if len(fluxs[k]) else np.median(flux_orig)
            tran_fluxs = np.full(len(tran_times), fill)
        else:
            # Create a polynomial function using the best_fit coefficients
            best_poly = np.poly1d(best_fit)
            # Calculate the transit fluxes using the best-fit polynomial
            tran_fluxs = best_poly(tran_times)

        times[k] = np.concatenate((times[k], tran_times))
        fluxs[k] = np.concatenate((fluxs[k], tran_fluxs))
        idx = np.argsort(times[k])
        times[k] = times[k][idx]
        fluxs[k] = fluxs[k][idx]

    ## This is outlier removing part. Note that it iteratively removes 5-sigma until no points are removed anymore.
    to_delete_num = 1.
    while to_delete_num > 0.:
        filter_flux = savgol_filter(fluxs[k], window, 3)
        to_delete = np.where(np.abs(fluxs[k]/filter_flux - 1.)/np.std(fluxs[k]/filter_flux) > cut )[0]
        fluxs[k][to_delete] = filter_flux[to_delete]
        flux_orig[to_delete] = filter_flux[to_delete]
        to_delete_num = len(to_delete)

    ## Detrend by dividing the observed signal with a smoothed version of the trend
    fluxs[k] = flux_orig / savgol_filter(fluxs[k], window, 3)


def _gp_detrend_product(eph, duration, times, fluxs, errs, k, gp_mask_margin):
    """Matern-3/2 GP-baseline detrending for one product, in place on fluxs[k] (times[k] untouched).

    The GP uses its OWN wide in-transit mask (+/- gp_mask_margin * duration) so it cannot latch onto
    transit wings; this is separate from the +/-0.75*duration ``transit_mask`` the function returns.
    A mask-protected 5-sigma clip cleans ONLY out-of-transit cadences (in-transit flux is never
    modified, preserving the transit for TDV), then flux is divided by the GP baseline. Raises on a
    degenerate/failed fit so the caller can fall back to savgol."""
    from gp_detrend import gp_baseline   # lazy import: data.py stays importable without celerite2

    n = len(fluxs[k])
    gp_mask = np.abs(eph.center_offset(times[k])) < gp_mask_margin * duration
    oot = ~gp_mask
    cut = 5.0
    cad_min = np.median(np.diff(np.sort(times[k]))) * 24. * 60.
    window = _safe_savgol_window(int(round(10 * (30. / cad_min)) * 2 + 1), n)

    e = (np.asarray(errs[k], dtype=float)
         if (errs is not None and errs[k] is not None) else np.full(n, np.nan))

    # Mask-protected iterative 5-sigma clip: detect/replace outliers among OUT-OF-TRANSIT cadences
    # only; in-transit flux is never touched.
    flux_clean = np.asarray(fluxs[k], dtype=float).copy()
    to_delete_num = 1
    while to_delete_num > 0:
        filt = savgol_filter(flux_clean, window, 3)
        ratio = flux_clean / filt - 1.
        sd = np.std(ratio)
        if not (np.isfinite(sd) and sd > 0):
            break
        to_delete = np.where((np.abs(ratio) / sd > cut) & oot)[0]
        flux_clean[to_delete] = filt[to_delete]
        to_delete_num = len(to_delete)

    baseline = gp_baseline(times[k], flux_clean, e, gp_mask)   # raises on degenerate/insane fit
    fluxs[k] = flux_clean / baseline


def lightcurve_extract(koi, is_LC, lcs, times, fluxs, eph=None, errs=None,
                       method="gp", gp_mask_margin=1.5):
    ## eph maps each time to its nearest (TTV-aware) transit center; eph=None reproduces the
    ## old linear ephemeris t0 + n*P so existing callers are unchanged.
    if eph is None:
        eph = TransitEphemeris(koi.koi_time0bk, koi.koi_period)
    duration = koi.koi_duration / 24. ## transit duration in days
    ## t_oot is the time from the nearest transit center (TTV-aware via eph)
    t_oot = [eph.center_offset(times[i]) for i in range(len(times))]## transit_mask is a boolean (true/false) array for whether at a certain time, a transit is occuring
    ## NOTE: the returned transit_mask stays at +/-0.75*duration for BOTH methods (downstream
    ## SNR/extraction depend on it). The GP uses its own wider mask internally (gp_mask_margin).
    transit_mask = [(np.abs(t_oot[i]) < .75*duration ) for i in range(len(times))]

    method = (method or "savgol").lower()

    ## Loop over all lightcurves
    for k in range(len(lcs)):
        use_method = method
        if use_method == "gp":
            n = len(fluxs[k])
            n_oot = int(np.sum(np.abs(eph.center_offset(times[k])) >= gp_mask_margin * duration))
            if n < MIN_SEG_POINTS or n_oot < MIN_OOT_POINTS:
                print(f"[lightcurve_extract] product {k}: too few points for GP "
                      f"(n={n}, n_oot={n_oot}); falling back to savgol.")
                use_method = "savgol"
            else:
                try:
                    _gp_detrend_product(eph, duration, times, fluxs, errs, k, gp_mask_margin)
                except Exception as err:  # noqa: BLE001 -- any GP failure -> safe savgol fallback
                    print(f"[lightcurve_extract] product {k}: GP detrend failed ({err}); "
                          f"falling back to savgol.")
                    use_method = "savgol"
        if use_method == "savgol":
            _savgol_detrend_product(koi, eph, duration, times, fluxs, transit_mask, k)
    return times, fluxs, transit_mask


def transit_duration(koi_number):
    # Find the target KOI.
    koi = get_koi(koi_number)

    # Get the period of the exoplanet's orbit (in days).
    period = koi.koi_period

    # Get the distance between the planet and the star at mid-transit divided by the stellar radius.
    koi_dor = koi.koi_dor

    # Assuming b = 0 (central transit).
    b = koi.koi_impact

    # Calculate the transit duration using the given formula.
    duration = (period / math.pi) * math.asin(math.sqrt(1 - b**2) / koi_dor)

    return duration


def select_cadence_per_transit(koi_number, t_out1, f_out1, ferr_out1, eph=None):
    """Collapse duplicate observations of the same transit down to a single cadence.

    A Kepler quarter can carry both a long- (30 min) and a short-cadence (1 min) product, so the
    same physical transit gets extracted twice -- once per cadence -- which double-counts transits
    downstream. Group the surviving per-transit segments by orbital epoch and keep, for each epoch,
    the short-cadence segment when one is present, else the long-cadence one. Cadence is inferred
    from each segment's own median sampling interval (short cadence ~1 min, long cadence ~30 min),
    so the smaller interval wins. Run this AFTER the transit-quality cuts: that way an epoch whose
    short-cadence segment is unusable falls back to its long-cadence segment instead of dropping
    the transit entirely.

    Also returns the surviving segment indices (into the input lists) as a 4th value, so a caller
    can keep per-candidate audit vectors aligned with the selected segments.
    """
    ## eph assigns each segment to its nearest (TTV-aware) transit epoch; eph=None reproduces
    ## the old round((median - t0)/P) grouping so existing callers are unchanged.
    if eph is None:
        koi = get_koi(koi_number)
        eph = TransitEphemeris(koi.koi_time0bk, koi.koi_period)

    best = {}  # epoch index -> (segment index, cadence in days)
    for i in range(len(t_out1)):
        epoch = eph.epoch_of(np.median(t_out1[i]))
        cad = float(np.median(np.diff(np.sort(t_out1[i]))))
        if epoch not in best or cad < best[epoch][1]:
            best[epoch] = (i, cad)

    keep = sorted(idx for idx, _ in best.values())
    return ([t_out1[i] for i in keep],
            [f_out1[i] for i in keep],
            [ferr_out1[i] for i in keep],
            keep)


def segment_coverage_ok(seg_t, center, t14, frac=0.5, min_in_transit=3):
    """Near-center coverage requirement (Component 2 / RC2).

    With ``dt = seg_t - center`` and half-window ``h = frac * t14``, a segment is kept iff it has at
    least one point just before the center (-h <= dt < 0), at least one just after (0 < dt <= +h),
    and at least ``min_in_transit`` points within +/-h. A transit sitting in a data gap, or with only
    one side observed, fails -- its likelihood carries no transit and the fit would return the
    catalog prior. Returns ``(ok, reason)`` where reason is ``None`` on pass else ``"no_coverage"``.
    """
    dt = np.asarray(seg_t, dtype=float) - float(center)
    h = float(frac) * float(t14)
    before = bool(np.any((dt >= -h) & (dt < 0)))
    after = bool(np.any((dt > 0) & (dt <= h)))
    n_in = int(np.sum(np.abs(dt) <= h))
    ok = before and after and (n_in >= int(min_in_transit))
    return ok, (None if ok else "no_coverage")


def segment_baseline_ok(seg_t, seg_f, seg_ferr, center, t14, nsigma=5.0):
    """Segment baseline-quality guard (Component 3 / RC3), evaluated on detrended flux.

    Over the out-of-transit cadences (``|dt| > 0.6 * t14``), the baseline must be flat at the noise
    level. The segment is flagged (dropped) if either
      - the OOT median deviates from 1 by more than ``nsigma * sigma / sqrt(n_oot)`` (a gap-edge
        ramp), or
      - a run of >= 3 consecutive OOT cadences (in time order) sits below ``1 - nsigma * sigma``
        (a local post-gap dip),
    where ``sigma = median(seg_ferr)`` (finite/positive-guarded). The median test requires
    ``n_oot >= 5`` and the run test ``n_oot >= 3``; below those the corresponding test is skipped.
    Returns ``(ok, reason, detail)``; ``detail`` is a short human-readable string (always set, for
    the log) and ``reason`` is ``None`` on pass else ``"bad_baseline"``.
    """
    dt = np.asarray(seg_t, dtype=float) - float(center)
    f = np.asarray(seg_f, dtype=float)
    oot = np.abs(dt) > 0.6 * float(t14)
    n_oot = int(np.sum(oot))

    ferr = np.asarray(seg_ferr, dtype=float)
    ferr = ferr[np.isfinite(ferr) & (ferr > 0)]
    if ferr.size == 0:
        return True, None, "skipped (no usable ferr)"
    sigma = float(np.median(ferr))
    if not (np.isfinite(sigma) and sigma > 0):
        return True, None, "skipped (sigma unusable)"

    if n_oot < 3:
        return True, None, f"skipped (n_oot={n_oot} < 3)"

    # median test (needs >= 5 OOT points)
    if n_oot >= 5:
        med = float(np.median(f[oot]))
        thresh = float(nsigma) * sigma / np.sqrt(n_oot)
        if abs(med - 1.0) > thresh:
            return False, "bad_baseline", (f"OOT median {med:.5f} (|dev|={abs(med-1.0):.5f} "
                                           f"> {thresh:.5f})")

    # consecutive-run test (needs >= 3 OOT points): walk points in time order, counting a run of
    # consecutive OOT cadences that all sit below 1 - nsigma*sigma.
    order = np.argsort(dt)
    f_ord, oot_ord = f[order], oot[order]
    low = 1.0 - float(nsigma) * sigma
    run = 0
    for val, is_oot in zip(f_ord, oot_ord):
        if is_oot and val < low:
            run += 1
            if run >= 3:
                return False, "bad_baseline", f"run of >=3 OOT cadences below {low:.5f}"
        else:
            run = 0

    return True, None, "ok"


def get_transit_arrays(times, fluxs,  ootvs, is_LC, lcs, koi, eph=None, exptimes=None):
        # Break down lightcurve into individual transits.
        # Phase-fold raw BJD time into out of transit time and
        # construct timing cadence array accordingly.
        # eph=None reproduces the old linear ephemeris so existing callers are unchanged.
        # exptimes: per-product exposure times in days (product_exptimes); None falls back to
        # the class constants selected by each product's is_LC flag. The 4th return value is
        # now one scalar exposure time (days) per segment -- segments are single-cadence by
        # construction (see the class split below), so a scalar is exact, survives every
        # downstream point mask untouched, and is what the transit models consume.
        if eph is None:
            eph = TransitEphemeris(koi.koi_time0bk, koi.koi_period)
        t_out, f_out, ferr_out, exp_out = [], [], [], []
        duration = koi.koi_duration/24.
        t_out_of_transit = [eph.center_offset(times[i]) for i in range(len(times))]
        transit_mask = [(np.abs(t_out_of_transit[i]) < 1.*duration ) for i in range(len(times))]
        for kk in range(len(lcs)):
            if np.sum(transit_mask[kk]) > 0:
                #tt, ff = t_out_of_transit[kk][transit_mask[kk]], fluxs[kk][transit_mask[kk]]
                tt, ff = times[kk][transit_mask[kk]], fluxs[kk][transit_mask[kk]]
                t_out.append(tt)
                f_out.append(ff)
                flux_err = np.std(fluxs[kk][~transit_mask[kk]])
                ferr_out.append(np.ones(len(tt))*flux_err)
                exp_kk = (float(exptimes[kk]) if exptimes is not None
                          else (LC_EXPTIME_D if is_LC[kk] else SC_EXPTIME_D))
                exp_out.append(np.ones(len(tt)) * exp_kk)
        tt1 = np.concatenate(t_out)
        ff1 = np.concatenate(f_out)
        ferr1 = np.concatenate(ferr_out)
        exp1 = np.concatenate(exp_out)
        t_out, f_out, ferr_out, exp_out = [], [], [], []
        ## Split into individual transits at any change of (TTV-aware) epoch. tt1 is concatenated
        ## across light-curve products without a global sort, so the epoch index can step in
        ## either direction at a boundary -- split on != 0, not on a positive period-sized gap.
        ## ALSO split at a cadence-class change within an epoch: adjacent products can both end/
        ## start on the same epoch (an LC and an SC observation of the same transit), which the
        ## epoch-only split would merge into one mixed-cadence "segment". Splitting keeps every
        ## segment single-cadence (a scalar exposure per segment is then exact) and lets
        ## select_cadence_per_transit dedup the two observations as intended.
        epochs = eph.epoch_of(tt1)
        cls1 = exp1 < SC_MAX_CADENCE_D
        brk = (np.diff(epochs) != 0) | (np.diff(cls1.astype(int)) != 0)
        n_class_splits = int(np.sum((np.diff(epochs) == 0) & (np.diff(cls1.astype(int)) != 0)))
        if n_class_splits:
            print(f"[get_transit_arrays] split {n_class_splits} same-epoch segment boundary(ies) "
                  f"at a cadence-class change (mixed LC/SC observation of one transit).")
        transit_list = np.append(-1, np.where(brk)[0])
        transit_list = np.append(transit_list,len(tt1)-1) + 1
        for kkk in range(len(transit_list) - 1):
            t_out.append(tt1[transit_list[kkk]:transit_list[kkk+1]])
            f_out.append(ff1[transit_list[kkk]:transit_list[kkk+1]])
            ferr_out.append(ferr1[transit_list[kkk]:transit_list[kkk+1]])
            seg_exp = exp1[transit_list[kkk]:transit_list[kkk+1]]
            if seg_exp.size and float(seg_exp.max() - seg_exp.min()) > 1e-9:
                print("[get_transit_arrays] WARNING: mixed exposure inside a segment after the "
                      "class split -- using the median; this should not happen.")
            exp_out.append(float(np.median(seg_exp)))
        return t_out, f_out, ferr_out, exp_out
