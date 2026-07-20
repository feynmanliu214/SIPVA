#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate synthetic TDV systems at target detection SNRs.

For each target SNR (default 5/10/20/30/50) this draws Kepler-like systems and
solves the impact-parameter trend ``db/dt`` so
that the limb-darkened TDV significance ``sqrt(SNR_square_LD)`` (snr.py) equals
the target. Systems whose required ``db/dt`` falls outside the physical band
[0.01, 0.03] are rejected, as are systems that would stop transiting across the
trend.

See docs/2026-06-06_generate_synthetic_snr_systems_plan.md for the full plan
(Codex-reviewed, approved round 5).

IMPORTANT: run under the repo venv, which has pytransit:
    .venv/bin/python scripts/generate_synthetic_systems.py --diagnostic
"""

import argparse
import csv
import sys
import time
import pathlib
from functools import partial

import numpy as np
from scipy.optimize import brentq

# --- import bootstrap: core modules use flat imports (snr.py:12), so src/core
#     must be on sys.path before importing snr (matches generate_SNR.py:7,
#     multiprocess.py:18). ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "core"))

try:
    from model import calculate_stellar_density
    from noise import sample_red_noise, apply_gaps, averaging_factor
    from pytransit.utils.mocklc import create_mock_light_curve
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"Missing dependency ({exc}). Run under the repo venv: "
        f".venv/bin/python {pathlib.Path(__file__).name}"
    )

OUT_DIR = REPO_ROOT / "data" / "SNR_data"

# Fixed conventions (match SNR_square_LD defaults, snr.py:177)
SIGMA = 1e-4 * 2          # per-sample photometric noise (200 ppm relative flux)
CADENCE_SEC = 60          # exposure / sampling cadence
PASSBAND = "Kepler"       # user decision 2026-06-06: standardize on Kepler LD
DB_BAND = (0.01, 0.03)    # physical db/dt band
BASELINE_DAYS = 1460.0    # Kepler ~4 yr
DUTY_CYCLE = 0.92

# CDPP conventions (realistic noise). create_mock_light_curve returns a fixed
# 5 h window; CDPP_6.5h is the standard Kepler precision timescale. CDPP is a
# property of the (stationary) noise process, so it is well-defined on a 6.5 h
# grid even though saved windows are 5 h.
MOCK_TOBS_HOURS = 5.0     # create_mock_light_curve window (tobs default)
CDPP_HOURS = 6.5          # CDPP reference timescale
N5 = round(MOCK_TOBS_HOURS * 3600 / CADENCE_SEC)   # points per transit window (300)
N6 = round(CDPP_HOURS * 3600 / CADENCE_SEC)        # points in a 6.5 h window (390)
DEFAULT_TARGET_CDPP_PPM = 34.5   # Kepler Kp~12 6.5h CDPP (literature anchor)

# Prior ranges (user 2026-06-09: hard boxes on b, p, a/Rs, num_transits).
# num_transits = floor(BASELINE_DAYS*DUTY_CYCLE/period), so bounding the count to [30, 100] is
# exactly a period box [BASELINE_DAYS*DUTY_CYCLE/100, BASELINE_DAYS*DUTY_CYCLE/30] = (~13.43, ~44.77) d
# -- the lower count floor (30) caps period at ~44.77 d; the upper cap (100) keeps runtime bounded.
MAX_TRANSITS = 100
MIN_TRANSITS = 30
PERIOD_RANGE = (BASELINE_DAYS * DUTY_CYCLE / MAX_TRANSITS,
                BASELINE_DAYS * DUTY_CYCLE / MIN_TRANSITS)   # log-uniform [d] (~13.43, ~44.77)
A_OVER_RS_RANGE = (30.0, 100.0)  # log-uniform; a/Rs sampled directly, rho_star derived (user 2026-06-09)
B_RANGE = (0.07, 0.5)           # uniform
P_RANGE = (0.01, 0.06)          # log-uniform (Rp/Rs); narrowed 2026-07-10 to cap per-transit SNR tail

DEFAULT_TARGETS = [5, 10, 20, 30, 50]


# --------------------------------------------------------------------------- #
# Prior sampling
# --------------------------------------------------------------------------- #
def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_system(rng):
    """Draw one system from the priors.

    a/Rs is sampled directly in A_OVER_RS_RANGE (user 2026-06-09) and rho_star is
    DERIVED from (a/Rs, period) via calculate_stellar_density -- self-consistent
    with the rho SystemSNR feeds the Kepler-LD model. RS_OVER_A = 1/(a/Rs) is the
    codebase convention. NUM_TRANSITS is fixed by the period box to [30, 100].
    """
    period = _loguniform(rng, *PERIOD_RANGE)
    a_over_rs = _loguniform(rng, *A_OVER_RS_RANGE)
    rs_over_a = 1.0 / a_over_rs
    b = float(rng.uniform(*B_RANGE))
    p = _loguniform(rng, *P_RANGE)
    num_transits = int(np.floor(BASELINE_DAYS * DUTY_CYCLE / period))
    num_transits = max(num_transits, MIN_TRANSITS)
    rho_star = float(calculate_stellar_density(rs_over_a, period))
    return {
        "b": b,
        "p": p,
        "PERIOD": period,
        "RS_OVER_A": rs_over_a,
        "NUM_TRANSITS": num_transits,
        "rho_star": rho_star,
    }


# --------------------------------------------------------------------------- #
# Limb-darkened SNR (replicates snr.SNR_square_LD, caching the M0 model)
# --------------------------------------------------------------------------- #
def _model_fluxes(period, num_transit, b0, p, rho, db_over_dt):
    """Noise-free Kepler-LD model fluxes, one array per transit.

    Mirrors snr._generate_model_light_curves exactly (passband='Kepler',
    noise=0.0, b_i = b0 + db/dt * period * i / 365).
    """
    times, fluxes = [], []
    for i in range(num_transit):
        bi = b0 + db_over_dt * period * i / 365.0
        t, f, _ = create_mock_light_curve(
            texp=CADENCE_SEC,
            passband=PASSBAND,
            noise=0.0,
            transit_pars={"period": period, "t0": period * i,
                          "ror": p, "rho": rho, "b": bi},
        )
        times.append(t)
        fluxes.append(f)
    return times, fluxes


class SystemSNR:
    """Evaluate sqrt(SNR_square_LD) vs db/dt for one system, caching M0."""

    def __init__(self, system):
        self.period = system["PERIOD"]
        self.num_transit = system["NUM_TRANSITS"]
        self.b0 = system["b"]
        self.p = system["p"]
        self.rs_over_a = system["RS_OVER_A"]
        self.rho = calculate_stellar_density(self.rs_over_a, self.period)
        # M0: constant-b model, independent of db/dt -> compute once.
        self._times0, self._f0 = _model_fluxes(
            self.period, self.num_transit, self.b0, self.p, self.rho, 0.0)

    def llr2(self, db_over_dt):
        _, f1 = _model_fluxes(self.period, self.num_transit, self.b0, self.p,
                              self.rho, db_over_dt)
        denom = SIGMA ** 2
        total = 0.0
        for a, b in zip(f1, self._f0):
            r = a - b
            total += float(np.sum(r * r / denom))
        return total

    def snr(self, db_over_dt):
        return float(np.sqrt(self.llr2(db_over_dt)))

    def endpoints(self):
        return self.snr(DB_BAND[0]), self.snr(DB_BAND[1])

    def model_times_fluxes(self, db_over_dt):
        return _model_fluxes(self.period, self.num_transit, self.b0, self.p,
                             self.rho, db_over_dt)


def b_final(system, db_over_dt):
    return system["b"] + db_over_dt * system["PERIOD"] * (system["NUM_TRANSITS"] - 1) / 365.0


def geometry_ok(system, db_over_dt):
    # transit persists across the full trend (matches code's T14 grazing limit).
    return b_final(system, db_over_dt) < 1.0 + system["p"]


def solve_db(system, target):
    """Return (db, achieved_snr) hitting `target`, or None if rejected.

    Reject reasons: target outside [SNR(0.01), SNR(0.03)] (unreachable), or the
    solved trend would stop the planet transiting.
    """
    ssnr = SystemSNR(system)
    snr_lo, snr_hi = ssnr.endpoints()
    if not (snr_lo <= target <= snr_hi):
        return None
    db = brentq(lambda d: ssnr.snr(d) - target, DB_BAND[0], DB_BAND[1], xtol=1e-9, rtol=1e-10)
    if not geometry_ok(system, db):
        return None
    return float(db), ssnr.snr(db)


# --------------------------------------------------------------------------- #
# Worker entry points (top-level for multiprocessing picklability)
# --------------------------------------------------------------------------- #
def probe_one(seed_int):
    """Diagnostic: draw a system, return its (snr_lo, snr_hi) and params."""
    rng = np.random.default_rng(seed_int)
    system = draw_system(rng)
    ssnr = SystemSNR(system)
    snr_lo, snr_hi = ssnr.endpoints()
    return {"system": system, "snr_lo": snr_lo, "snr_hi": snr_hi}


def solve_one(seed_int, target):
    """Generation: draw a system, try to hit `target`. Return dict or None."""
    rng = np.random.default_rng(seed_int)
    system = draw_system(rng)
    result = solve_db(system, target)
    if result is None:
        return None
    db, achieved = result
    return {"system": system, "db": db, "achieved_snr": achieved, "seed": int(seed_int)}


def _map(jobs, func, iterable):
    if jobs <= 1:
        return [func(x) for x in iterable]
    import multiprocessing as mp
    with mp.Pool(jobs) as pool:
        return pool.map(func, iterable)


# --------------------------------------------------------------------------- #
# Diagnostic mode
# --------------------------------------------------------------------------- #
def run_diagnostic(targets, n_probe, jobs, master_seed):
    master = np.random.default_rng(master_seed)
    seeds = master.integers(0, 2**63 - 1, size=n_probe)
    t0 = time.time()
    results = _map(jobs, probe_one, list(seeds))
    elapsed = time.time() - t0

    los = np.array([r["snr_lo"] for r in results])
    his = np.array([r["snr_hi"] for r in results])
    print(f"\nReachability diagnostic: {n_probe} systems, {elapsed:.1f}s "
          f"({elapsed / n_probe * 1000:.0f} ms/system), {jobs} job(s)\n")
    print(f"  SNR @ db=0.01 (band low):  min {los.min():.2f}  median {np.median(los):.2f}  max {los.max():.2f}")
    print(f"  SNR @ db=0.03 (band high): min {his.min():.2f}  median {np.median(his):.2f}  max {his.max():.2f}\n")
    print(f"  {'target':>7}  {'reachable':>9}  {'rate':>6}  est.draws/100")
    for target in targets:
        reach = np.mean((los <= target) & (target <= his))
        est = "inf" if reach == 0 else f"{int(np.ceil(100 / reach))}"
        print(f"  {target:>7}  {int(reach * n_probe):>9}  {reach:>6.1%}  {est:>12}")
    print()
    return {t: float(np.mean((los <= t) & (t <= his))) for t in targets}


# --------------------------------------------------------------------------- #
# Generation mode
# --------------------------------------------------------------------------- #
CSV_COLUMNS = ["b", "p", "PERIOD", "DB_OVER_DT", "RS_OVER_A",
               "NUM_TRANSITS", "rho_star", "achieved_snr", "name"]


def save_lightcurve(lc_dir, name, times, fluxes):
    np.savez(
        lc_dir / f"{name}.npz",
        times=np.array(times, dtype=object),
        fluxes=np.array(fluxes, dtype=object),
        allow_pickle=True,
    )


REALISTIC_MANIFEST_COLUMNS = [
    "target", "row_index", "name",
    "n_transits_ideal", "n_transits_kept", "dropped_fraction", "n_points",
    "target_cdpp_6p5_ppm", "cdpp_6p5_model_ppm", "cdpp_6p5_measured_ppm",
    "cdpp_white_6p5_ppm", "cdpp_red_6p5_ppm",
    "rms_red_only_ppm", "rms_white_plus_red_ppm",
    "sigma_r", "red_tau_hours", "tau_days",
    "gap_prob", "min_transits_kept", "gap_retry_exhausted", "parent_seed",
]


def calibrate_cdpp(target_cdpp_ppm, red_tau_hours):
    """Solve the Matern-3/2 GP amplitude sigma_r so the white+red light curve has
    CDPP_6.5h ~= target_cdpp_ppm (req: calibrate amplitude to a literature CDPP).

    White noise sets the floor (sigma_w averaged over 6.5 h); the red GP supplies
    the remainder. Anchor: Kepler Kp~12 -> CDPP_6.5h ~ 34.5 ppm. Returns a config
    dict reused for every curve (sigma_r depends only on tau + cadence, not on
    the system).
    """
    tau_days = red_tau_hours / 24.0
    cadence_days = CADENCE_SEC / 86400.0
    g6 = averaging_factor(N6, cadence_days, tau_days)
    g5 = averaging_factor(N5, cadence_days, tau_days)
    cdpp_white_6 = 1e6 * SIGMA / np.sqrt(N6)
    if target_cdpp_ppm <= cdpp_white_6:
        raise SystemExit(
            f"--target-cdpp-6p5-ppm ({target_cdpp_ppm}) must exceed the white-noise "
            f"floor {cdpp_white_6:.2f} ppm (per-point SIGMA={SIGMA}).")
    # CDPP_total^2 = cdpp_white^2 + (1e6 sigma_r)^2 g6  ->  solve sigma_r
    sigma_r = float(np.sqrt((target_cdpp_ppm / 1e6) ** 2 - SIGMA ** 2 / N6) / np.sqrt(g6))
    cdpp_red_6 = 1e6 * sigma_r * np.sqrt(g6)
    cdpp_total_6 = float(np.hypot(cdpp_white_6, cdpp_red_6))
    cdpp_total_5 = 1e6 * float(np.sqrt(SIGMA ** 2 / N5 + sigma_r ** 2 * g5))
    return {
        "sigma_r": sigma_r, "red_tau_hours": red_tau_hours, "tau_days": tau_days,
        "target_cdpp": target_cdpp_ppm, "cdpp_white_6p5": cdpp_white_6,
        "cdpp_red_6p5": cdpp_red_6, "cdpp_total_6p5": cdpp_total_6,
        "cdpp_total_5h": cdpp_total_5,
    }


def save_realistic_curve(lc_dir, name, rec, times, model_fluxes, target,
                         row_index, cdpp_cfg, gap_prob, min_transits_kept):
    """Save a realistic-noise curve (model + white + calibrated red, gaps removed).

    Three independent streams are spawned from the system's parent seed in the
    fixed order [white, red, gap], so the curve is reproducible from the same
    invocation. Kept windows keep the per-transit object-array layout of
    ``save_lightcurve``. Reports CDPP_6.5h (model + measured) and the separately
    named red-only / white+red per-point RMS. Returns the manifest row (dict).
    """
    parent_seed = rec["seed"]
    white_ss, red_ss, gap_ss = np.random.SeedSequence(parent_seed).spawn(3)
    white_rng = np.random.default_rng(white_ss)
    red_rng = np.random.default_rng(red_ss)
    gap_rng = np.random.default_rng(gap_ss)

    sigma_r, tau_days = cdpp_cfg["sigma_r"], cdpp_cfg["tau_days"]
    n_ideal = len(model_fluxes)
    keep_mask, gap_retry_exhausted = apply_gaps(
        n_ideal, gap_prob, min_transits_kept, gap_rng)

    kept_times, kept_fluxes, kept_idx = [], [], []
    window_means, red_parts, total_parts, n_points = [], [], [], 0
    for i in range(n_ideal):
        if not keep_mask[i]:
            continue
        t_i = times[i]
        assert t_i.size == N5, f"window length {t_i.size} != N5={N5}"
        red = sample_red_noise(t_i, sigma_r, tau_days, red_rng)
        noise = white_rng.normal(0.0, SIGMA, size=t_i.size) + red
        kept_times.append(t_i)
        kept_fluxes.append(model_fluxes[i] + noise)
        kept_idx.append(i)
        window_means.append(float(noise.mean()))   # for measured 5 h CDPP
        red_parts.append(red)
        total_parts.append(noise)
        n_points += t_i.size

    np.savez(
        lc_dir / f"{name}.npz",
        times=np.array(kept_times, dtype=object),
        fluxes=np.array(kept_fluxes, dtype=object),
        kept_transits=np.array(kept_idx, dtype=int),
        target_cdpp_6p5_ppm=cdpp_cfg["target_cdpp"],
        cdpp_6p5_model_ppm=cdpp_cfg["cdpp_total_6p5"],
        sigma_r=sigma_r, red_tau_hours=cdpp_cfg["red_tau_hours"],
        tau_days=tau_days, gap_prob=gap_prob, sigma=SIGMA,
        allow_pickle=True,
    )

    # Measured CDPP: std of the per-window noise means is an empirical estimate
    # of the 5 h (window-scale) CDPP; express at 6.5 h via the model ratio.
    wm = np.array(window_means)
    cdpp_5h_meas = 1e6 * float(wm.std(ddof=1)) if wm.size >= 2 else float("nan")
    cdpp_6p5_meas = cdpp_5h_meas * (cdpp_cfg["cdpp_total_6p5"] / cdpp_cfg["cdpp_total_5h"])
    # Per-point RMS, red-only vs white+red kept distinct (not conflated).
    rms_red_only = 1e6 * float(np.std(np.concatenate(red_parts))) if red_parts else float("nan")
    rms_total = 1e6 * float(np.std(np.concatenate(total_parts))) if total_parts else float("nan")

    n_kept = len(kept_idx)
    return {
        "target": target, "row_index": row_index, "name": name,
        "n_transits_ideal": n_ideal, "n_transits_kept": n_kept,
        "dropped_fraction": (n_ideal - n_kept) / n_ideal if n_ideal else 0.0,
        "n_points": n_points,
        "target_cdpp_6p5_ppm": cdpp_cfg["target_cdpp"],
        "cdpp_6p5_model_ppm": cdpp_cfg["cdpp_total_6p5"],
        "cdpp_6p5_measured_ppm": cdpp_6p5_meas,
        "cdpp_white_6p5_ppm": cdpp_cfg["cdpp_white_6p5"],
        "cdpp_red_6p5_ppm": cdpp_cfg["cdpp_red_6p5"],
        "rms_red_only_ppm": rms_red_only,
        "rms_white_plus_red_ppm": rms_total,
        "sigma_r": sigma_r, "red_tau_hours": cdpp_cfg["red_tau_hours"],
        "tau_days": tau_days, "gap_prob": gap_prob,
        "min_transits_kept": min_transits_kept,
        "gap_retry_exhausted": gap_retry_exhausted, "parent_seed": parent_seed,
    }


def generate_target(target, n_systems, jobs, master_seed, max_draws,
                    max_seconds, n_probe, force, noise_model="white",
                    cdpp_cfg=None, gap_prob=0.0, min_transits_kept=3):
    master = np.random.default_rng(master_seed)

    # Branch at entry on noise_model BEFORE binding/checking/creating any white
    # output path (the white CSV bind + exists-check + mkdir below must never run
    # in realistic mode). The draw loop is shared; only path setup, per-record
    # save, and final output differ between modes.
    csv_path = None
    if noise_model == "white":
        lc_dir = OUT_DIR / "lightcurves" / f"SNR_{target}"
        csv_path = OUT_DIR / f"SNR_{target}.csv"
        if csv_path.exists() and not force:
            raise SystemExit(f"Refusing to overwrite {csv_path} (use --force).")
        lc_dir.mkdir(parents=True, exist_ok=True)
    else:  # realistic: own dir, own collision check, never touch white paths
        lc_dir = OUT_DIR / "lightcurves_realistic" / f"SNR_{target}"
        if lc_dir.exists() and any(lc_dir.glob("*.npz")) and not force:
            raise SystemExit(f"Refusing to overwrite {lc_dir} (use --force).")
        lc_dir.mkdir(parents=True, exist_ok=True)
        if force:  # replace only this target's realistic curves
            for old in lc_dir.glob("*.npz"):
                old.unlink()

    accepted = []
    draws = 0
    t0 = time.time()
    batch = max(jobs * 4, 16)
    warned_abort = False

    while len(accepted) < n_systems:
        if draws >= max_draws:
            print(f"  [SNR {target}] hit --max-draws={max_draws}; stopping with "
                  f"{len(accepted)}/{n_systems}.")
            break
        if time.time() - t0 >= max_seconds:
            print(f"  [SNR {target}] hit --max-seconds={max_seconds}; stopping with "
                  f"{len(accepted)}/{n_systems}.")
            break

        seeds = master.integers(0, 2**63 - 1, size=batch)
        draws += batch
        results = _map(jobs, partial(solve_one, target=target), list(seeds))
        accepted.extend(r for r in results if r is not None)

        # Early-abort: after warmup, if observed acceptance can't reach the goal
        # within --max-draws, stop this target rather than spin.
        if not warned_abort and draws >= n_probe:
            rate = len(accepted) / draws
            if rate * max_draws < n_systems:
                print(f"  [SNR {target}] acceptance {rate:.2%} too low to reach "
                      f"{n_systems} within --max-draws={max_draws}; aborting with "
                      f"{len(accepted)} accepted.")
                break
            warned_abort = True

    accepted = accepted[:n_systems]

    rows = []
    achieved = []
    manifest_rows = []
    for idx, rec in enumerate(accepted):
        name = f"syn_snr{target}_{idx:03d}"
        system, db = rec["system"], rec["db"]
        # Saved curve: Kepler M1 model (noise=0) + our own seeded noise.
        ssnr = SystemSNR(system)
        times, model_fluxes = ssnr.model_times_fluxes(db)
        if noise_model == "white":
            noise_rng = np.random.default_rng(rec["seed"])
            noisy = [f + noise_rng.normal(0.0, SIGMA, size=f.size) for f in model_fluxes]
            save_lightcurve(lc_dir, name, times, noisy)
        else:
            manifest_rows.append(save_realistic_curve(
                lc_dir, name, rec, times, model_fluxes, target, idx,
                cdpp_cfg, gap_prob, min_transits_kept))
        achieved.append(rec["achieved_snr"])
        rows.append([system["b"], system["p"], system["PERIOD"], db,
                     system["RS_OVER_A"], system["NUM_TRANSITS"],
                     system["rho_star"], rec["achieved_snr"], name])

    # White mode writes the per-target system CSV; realistic mode never does
    # (it reproduces the same systems and writes only curves + manifest).
    if noise_model == "white":
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

    achieved = np.array(achieved) if achieved else np.array([np.nan])
    elapsed = time.time() - t0
    summary = {
        "target": target,
        "accepted": len(rows),
        "draws": draws,
        "acceptance_rate": len(rows) / draws if draws else 0.0,
        "achieved_mean": float(np.nanmean(achieved)),
        "achieved_std": float(np.nanstd(achieved)),
        "elapsed_s": elapsed,
    }
    out_label = csv_path if noise_model == "white" else lc_dir
    print(f"  [SNR {target}] {len(rows)}/{n_systems} accepted, "
          f"{summary['acceptance_rate']:.1%} rate, {draws} draws, {elapsed:.0f}s, "
          f"achieved {summary['achieved_mean']:.2f} +/- {summary['achieved_std']:.2f} "
          f"-> {out_label}")
    return summary, manifest_rows


SUMMARY_COLUMNS = ["target", "accepted", "draws", "acceptance_rate",
                   "achieved_mean", "achieved_std", "elapsed_s"]


def write_summary(summaries):
    path = OUT_DIR / "summary.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    print(f"\nSummary -> {path}")


def _merge_csv_rows(path, cols, new_rows, targets_done):
    """Rewrite ``path`` keeping existing rows for untouched targets, then append
    ``new_rows`` (rows for the just-(re)generated targets). Matches the manifest
    / realistic-summary target-row replacement semantics in the plan."""
    done = {str(t) for t in targets_done}
    kept = []
    if path.exists():
        with open(path, newline="") as fh:
            kept = [r for r in csv.DictReader(fh) if str(r["target"]) not in done]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in kept + list(new_rows):
            writer.writerow(r)


def write_realistic_manifest(rows, targets_done):
    path = OUT_DIR / "lightcurves_realistic" / "realistic_manifest.csv"
    _merge_csv_rows(path, REALISTIC_MANIFEST_COLUMNS, rows, targets_done)
    print(f"Manifest -> {path}")


def write_realistic_summary(summaries, targets_done):
    path = OUT_DIR / "lightcurves_realistic" / "realistic_summary.csv"
    _merge_csv_rows(path, SUMMARY_COLUMNS, summaries, targets_done)
    print(f"Summary -> {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS,
                    help="Target SNR values (default: 5 10 20 30 50)")
    ap.add_argument("--n-systems", type=int, default=50,
                    help="Systems to generate per target (default: 50)")
    ap.add_argument("--seed", type=int, default=0, help="Master RNG seed")
    ap.add_argument("--jobs", type=int, default=4, help="Worker processes")
    ap.add_argument("--max-draws", type=int, default=50000,
                    help="Max candidate draws per target")
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="Wall-clock cap per target (s)")
    ap.add_argument("--n-probe", type=int, default=2000,
                    help="Warmup draws before the early-abort check / diagnostic pool")
    ap.add_argument("--diagnostic", action="store_true",
                    help="Run reachability diagnostic only (no files written)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs for the targets being run")
    ap.add_argument("--noise-model", choices=["white", "realistic"], default="white",
                    help="white: model + N(0,sigma) (default, existing behavior); "
                         "realistic: model + white + CDPP-calibrated red noise")
    ap.add_argument("--target-cdpp-6p5-ppm", type=float, default=DEFAULT_TARGET_CDPP_PPM,
                    help="Target 6.5h CDPP (ppm) the white+red curve is calibrated to; "
                         f"default {DEFAULT_TARGET_CDPP_PPM} (Kepler Kp~12)")
    ap.add_argument("--red-tau-hours", type=float, default=2.0,
                    help="Red-noise (granulation) correlation time in hours (default 2.0)")
    ap.add_argument("--gap-prob", type=float, default=0.0,
                    help="STRESS-TEST per-transit dropout probability; default 0.0 "
                         "(NUM_TRANSITS already encodes the 0.92 duty cycle)")
    ap.add_argument("--min-transits-kept", type=int, default=3,
                    help="Floor on surviving transits per system (default 3)")
    args = ap.parse_args()

    targets = [int(t) if float(t).is_integer() else t for t in args.targets]

    if args.diagnostic:
        run_diagnostic(targets, args.n_probe, args.jobs, args.seed)
        return

    # Preflight validation (matters in realistic mode; cheap to always check).
    if not (0.0 <= args.gap_prob <= 1.0):
        raise SystemExit(f"--gap-prob must be in [0, 1] (got {args.gap_prob}).")
    if args.red_tau_hours <= 0:
        raise SystemExit(f"--red-tau-hours must be > 0 (got {args.red_tau_hours}).")
    if args.target_cdpp_6p5_ppm <= 0:
        raise SystemExit(f"--target-cdpp-6p5-ppm must be > 0 (got {args.target_cdpp_6p5_ppm}).")
    if args.min_transits_kept < 1:
        raise SystemExit(f"--min-transits-kept must be >= 1 (got {args.min_transits_kept}).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.noise_model == "white":
        summaries = []
        for i, target in enumerate(targets):
            # distinct seed stream per target for independence
            summary, _ = generate_target(
                target, args.n_systems, args.jobs, args.seed + 1 + i,
                args.max_draws, args.max_seconds, args.n_probe, args.force)
            summaries.append(summary)
        write_summary(summaries)
    else:
        # Calibrate the GP amplitude once (independent of target SNR / system).
        cdpp_cfg = calibrate_cdpp(args.target_cdpp_6p5_ppm, args.red_tau_hours)
        print(f"CDPP calibration (tau={args.red_tau_hours} h): "
              f"target={cdpp_cfg['target_cdpp']:.1f}  white={cdpp_cfg['cdpp_white_6p5']:.2f}  "
              f"red={cdpp_cfg['cdpp_red_6p5']:.2f}  model_total={cdpp_cfg['cdpp_total_6p5']:.2f} ppm  "
              f"-> sigma_r={cdpp_cfg['sigma_r']:.3e} ({1e6*cdpp_cfg['sigma_r']:.1f} ppm/point)")
        (OUT_DIR / "lightcurves_realistic").mkdir(parents=True, exist_ok=True)
        summaries, manifest_rows = [], []
        for i, target in enumerate(targets):
            summary, mrows = generate_target(
                target, args.n_systems, args.jobs, args.seed + 1 + i,
                args.max_draws, args.max_seconds, args.n_probe, args.force,
                noise_model="realistic", cdpp_cfg=cdpp_cfg, gap_prob=args.gap_prob,
                min_transits_kept=args.min_transits_kept)
            summaries.append(summary)
            manifest_rows.extend(mrows)
        write_realistic_manifest(manifest_rows, targets)
        write_realistic_summary(summaries, targets)


if __name__ == "__main__":
    main()
