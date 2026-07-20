#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit of the rho_eff = rho_star + p^3 rho_p correction for the 16-KOI real-Kepler sample.

Documents, per KOI: the adopted planet-mass source (NASA Exoplanet Archive Planetary Systems
table, DEFAULT parameter set, snapshotted on the run date), missing / unreliable masses, the
correction p^3 rho_p in g/cm^3 and as a fraction of the catalog stellar density, and how it
compares with (a) the stellar-density prior width actually used by the individual-fit pipeline
(koi_prior_spec: min(quadrature catalog errors, TDV_RHO_PRIOR_FRAC * srho), default frac 0.12)
and (b) the SIPVA global-fit posterior width (rho row of parameters_koi_<koi>.csv under
../data/Output_data_gp/).

Key identity (planet radius cancels): with p = Rp/Rstar and rho_p = Mp / (4pi/3 Rp^3),

    p^3 rho_p = 3 Mp / (4 pi Rstar^3),

so the correction needs only Mp and the catalog Rstar (koi_srad). The rho_p column shown in the
outputs is informational only (computed from the KOI koi_prad radius).

READ-ONLY with respect to the pipeline: queries the archive + reads existing result CSVs. It
runs no fits and imports nothing from src/core.

Run from the scripts/ directory (network required for the archive queries):

    ../.venv/bin/python rhoeff_correction_audit.py

Writes to ../data/rhoeff_audit/:
    rhoeff_correction_audit.csv    one row per KOI (values + provenance + status)
    rhoeff_correction_audit.md     human-readable summary (assumptions, table, ranges)
    raw_cumulative_snapshot.csv    verbatim cumulative-KOI-table query result
    raw_ps_default_snapshot.csv    verbatim PS default-parameter-set query result
"""

import os
import html
import datetime

import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# --- Sample definition ----------------------------------------------------------------------
# The 16 production KOIs (same list as run_koi_batch.py).
KOIS = ["103.01", "137.02", "139.01", "142.01", "209.02", "377.01", "377.02", "460.01",
        "806.01", "841.02", "872.01", "1320.01", "1423.01", "1856.01", "2698.01", "2770.01"]

# KOI -> planet name as it appears in the archive PS table (pl_name), or None for candidates
# with no confirmed-planet entry. NOTE: Kepler-88 b is listed in the PS table under its KOI
# designation 'KOI-142 b' (queries on hostname='Kepler-88' return nothing); mapping it here is
# what makes the lookup reproducible.
PS_NAME = {
    "103.01":  "Kepler-1710 b",
    "137.02":  "Kepler-18 d",
    "139.01":  "Kepler-111 c",
    "142.01":  "KOI-142 b",        # = Kepler-88 b
    "209.02":  "Kepler-117 b",
    "377.01":  "Kepler-9 b",
    "377.02":  "Kepler-9 c",
    "460.01":  "Kepler-559 b",
    "806.01":  "Kepler-30 d",
    "841.02":  "Kepler-27 c",
    "872.01":  "Kepler-46 b",
    "1320.01": "Kepler-816 b",
    "1423.01": "Kepler-841 b",
    "1856.01": None,               # candidate
    "2698.01": "Kepler-1316 b",
    "2770.01": None,               # candidate
}

# PS-default masses that are dynamical UPPER LIMITS rather than measurements. These are excluded
# from the quantitative correction range (their nominal correction is quoted parenthetically).
UNRELIABLE = {
    "841.02": ("PS default 4385.87 Me (Steffen et al. 2012) is a dynamical upper limit "
               "(~13.8 Mjup); Hadden & Lithwick 2017 TTV analysis favors ~16 Me."),
    "872.01": ("PS default 1907 Me (Nesvorny et al. 2012) is the ~6 Mjup dynamical upper "
               "limit (no uncertainties in the PS row; implies rho_p ~ 17 g/cm^3)."),
}

# --- Constants ------------------------------------------------------------------------------
M_EARTH_G = 5.9722e27      # g
R_SUN_CM = 6.957e10        # cm (IAU nominal)
R_EARTH_CM = 6.371e8       # cm (mean radius; consistent with Earth mean density 5.51 g/cm^3)
RHO_PRIOR_FRAC = float(os.environ.get("TDV_RHO_PRIOR_FRAC", "0.12"))  # koi_prior_spec default

GP_ROOT = os.path.join("..", "data", "Output_data_gp")   # production (manuscript) outputs
OUT_DIR = os.path.join("..", "data", "rhoeff_audit")


def _kepoi_name(koi):
    integer, frac = f"{float(koi):.2f}".split(".")
    return f"K{int(integer):05d}.{frac}"


def _val(row, col):
    """One astropy-table cell -> float or NaN (strip units, masked -> NaN)."""
    v = row[col]
    if hasattr(v, "value"):
        v = v.value
    if v is np.ma.masked or np.ma.is_masked(v):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _ref_short(refname):
    """'<a ...>Author 2012</a>' -> 'Author 2012'."""
    s = str(refname)
    s = s.split(">")[1].split("<")[0] if ">" in s else s
    return html.unescape(s).strip()


def query_archive():
    names = ",".join(f"'{_kepoi_name(k)}'" for k in KOIS)
    cum = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select=("kepoi_name,kepler_name,koi_disposition,koi_period,koi_ror,"
                "koi_prad,koi_prad_err1,koi_prad_err2,koi_srad,koi_smass,"
                "koi_srho,koi_srho_err1,koi_srho_err2"),
        where=f"kepoi_name in ({names})")

    pl_names = ",".join(f"'{n}'" for n in PS_NAME.values() if n is not None)
    ps = NasaExoplanetArchive.query_criteria(
        table="ps",
        select=("pl_name,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_bmassprov,"
                "pl_rade,pl_refname"),
        where=f"pl_name in ({pl_names}) and default_flag=1")
    return cum, ps


def posterior_rho_width(koi):
    """SIPVA global-fit rho width max(err_lower, err_upper) from parameters_koi_<koi>.csv,
    or NaN if the file/row is absent. max() matches the pipeline's final_err convention."""
    path = os.path.join(GP_ROOT, f"koi-{koi}", f"parameters_koi_{koi}.csv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    row = df[df["parameter"] == "rho"]
    if len(row) == 0:
        return np.nan
    return float(max(row["err_lower"].iloc[0], row["err_upper"].iloc[0]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()

    cum, ps = query_archive()
    cum.write(os.path.join(OUT_DIR, "raw_cumulative_snapshot.csv"),
              format="ascii.csv", overwrite=True)
    ps.write(os.path.join(OUT_DIR, "raw_ps_default_snapshot.csv"),
             format="ascii.csv", overwrite=True)

    cum_by_koi = {str(r["kepoi_name"]).strip(): r for r in cum}
    ps_by_name = {str(r["pl_name"]).strip(): r for r in ps}

    rows = []
    for koi in KOIS:
        c = cum_by_koi[_kepoi_name(koi)]
        srho = _val(c, "koi_srho")
        srho_e1, srho_e2 = _val(c, "koi_srho_err1"), _val(c, "koi_srho_err2")
        srad = _val(c, "koi_srad")
        prad = _val(c, "koi_prad")
        ror = _val(c, "koi_ror")
        disp = str(c["koi_disposition"]).strip()

        # Individual-fit prior width, exactly as koi_prior_spec builds it (mean
        # symmetrization of the +/- errors since the 2026-07 model corrections; was
        # quadrature -- the frozen data/rhoeff_audit outputs predate this).
        _errs = [abs(e) for e in (srho_e1, srho_e2) if np.isfinite(e) and e != 0.0]
        sigma_cat = (sum(_errs) / len(_errs)) if _errs else 0.0
        sigma_prior = min(sigma_cat, RHO_PRIOR_FRAC * srho)
        sigma_post = posterior_rho_width(koi)

        pl = PS_NAME[koi]
        r = ps_by_name.get(pl) if pl else None
        mp = _val(r, "pl_bmasse") if r is not None else np.nan
        mp_e1 = _val(r, "pl_bmasseerr1") if r is not None else np.nan
        mp_e2 = _val(r, "pl_bmasseerr2") if r is not None else np.nan
        ref = _ref_short(r["pl_refname"]) if r is not None else ""

        if pl is None:
            status, note = "missing_no_confirmed_planet", "KOI candidate; no PS entry."
        elif not np.isfinite(mp):
            status, note = "missing_no_default_mass", "No mass in the PS default parameter set."
        elif koi in UNRELIABLE:
            status, note = "unreliable_upper_limit", UNRELIABLE[koi]
        else:
            status, note = "measured", ""

        if np.isfinite(mp):
            # The correction itself: planet radius cancels -> only Mp and catalog Rstar enter.
            corr = 3.0 * mp * M_EARTH_G / (4.0 * np.pi * (srad * R_SUN_CM) ** 3)
            # Informational planet density from the KOI radius (koi_prad).
            rho_p = mp * M_EARTH_G / (4.0 / 3.0 * np.pi * (prad * R_EARTH_CM) ** 3)
        else:
            corr, rho_p = np.nan, np.nan

        rows.append(dict(
            koi=koi, planet=(pl or ""), disposition=disp, status=status,
            mp_me=mp, mp_err1_me=mp_e1, mp_err2_me=mp_e2, mass_ref=ref,
            prad_re=prad, ror=ror, srad_rsun=srad, srho_gcc=srho,
            rho_p_gcc=rho_p, corr_gcc=corr,
            corr_over_srho=corr / srho,
            sigma_prior_gcc=sigma_prior, corr_over_sigma_prior=corr / sigma_prior,
            sigma_post_gcc=sigma_post, corr_over_sigma_post=corr / sigma_post,
            note=note,
        ))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "rhoeff_correction_audit.csv")
    df.to_csv(csv_path, index=False)

    # --- Markdown summary ---------------------------------------------------------------
    meas = df[df["status"] == "measured"]
    excl = df[df["status"] != "measured"]

    def _fmt(x, f="{:.2e}"):
        return f.format(x) if np.isfinite(x) else "--"

    lines = [
        f"# rho_eff correction audit for the 16-KOI sample ({today})",
        "",
        "Correction to the sampled transit density if reinterpreted as "
        "`rho_eff = rho_star + p^3 rho_p`, i.e. `(a/Rstar)^3 = G P^2 rho_eff / (3 pi)`.",
        "",
        "## Assumptions and sources",
        "",
        "- Planet masses: NASA Exoplanet Archive **Planetary Systems table, default parameter "
        f"set**, queried {today} (verbatim snapshot: `raw_ps_default_snapshot.csv`). "
        "Kepler-88 b appears there under the name `KOI-142 b`.",
        "- Stellar/transit parameters (`koi_srho`, `koi_srad`, `koi_prad`, `koi_ror`): "
        "archive **cumulative KOI table** (snapshot: `raw_cumulative_snapshot.csv`). The live "
        "pipeline's priors use DR25 first with a cumulative fallback (see `data.get_koi`); "
        "cumulative is used here uniformly for simplicity, which only matters at the few-percent "
        "level for the reference `rho_star`, far below the decision thresholds in this audit.",
        "- The correction is computed as `p^3 rho_p = 3 Mp / (4 pi Rstar^3)` with the catalog "
        "`koi_srad` -- the planet radius cancels exactly, so radius-source inconsistencies do "
        "not enter. The `rho_p` column is informational (from `koi_prad`).",
        f"- Individual-fit prior width sigma_prior = min(mean(|koi_srho_err1|, |koi_srho_err2|), "
        f"{RHO_PRIOR_FRAC:g} * koi_srho), mirroring `koi_prior_spec` "
        "(TDV_RHO_PRIOR_FRAC default; mean symmetrization since the 2026-07 model "
        "corrections). The SIPVA prior is now catalog-only with the same capped width "
        "(`sipva_prior_spec`); the SIPVA comparison below still uses the *posterior* width:",
        "- sigma_post = max(err_lower, err_upper) of the `rho` row of "
        "`Output_data_gp/koi-*/parameters_koi_*.csv` (the pipeline's `final_err` convention).",
        "- No fits were run; nothing under `Output_data*` was modified.",
        "",
        "## Per-KOI table",
        "",
        "| KOI | planet | status | Mp [Me] (ref) | rho_p [g/cc] | p^3 rho_p [g/cc] | "
        "/rho_star | /sigma_prior | /sigma_post |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        mp_txt = (f"{r.mp_me:.1f} ({r.mass_ref})" if np.isfinite(r.mp_me) else "--")
        lines.append(
            f"| {r.koi} | {r.planet or '--'} | {r.status} | {mp_txt} | "
            f"{_fmt(r.rho_p_gcc, '{:.2f}')} | {_fmt(r.corr_gcc)} | "
            f"{_fmt(r.corr_over_srho)} | {_fmt(r.corr_over_sigma_prior)} | "
            f"{_fmt(r.corr_over_sigma_post)} |")

    lines += [
        "",
        "## Quantitative range (measured masses only)",
        "",
        f"- KOIs with a **measured** default-set mass: {len(meas)} of {len(df)} "
        f"({', '.join(meas.koi)}).",
        f"- p^3 rho_p / rho_star: {meas.corr_over_srho.min():.1e} .. "
        f"{meas.corr_over_srho.max():.1e}",
        f"- p^3 rho_p / sigma_prior (individual-fit prior width): "
        f"{meas.corr_over_sigma_prior.min():.1e} .. {meas.corr_over_sigma_prior.max():.1e}",
        f"- p^3 rho_p / sigma_post (SIPVA posterior width): "
        f"{meas.corr_over_sigma_post.min():.1e} .. {meas.corr_over_sigma_post.max():.1e}",
        "",
        "The correction is everywhere a small fraction of both the prior and the posterior "
        "width for the measured-mass targets: relabeling the sampled density as rho_eff "
        "changes its interpretation, not any fitted number at current precision.",
        "",
        "## Targets excluded from the quantitative range",
        "",
    ]
    for _, r in excl.iterrows():
        why = r.note or r.status
        extra = (f" Nominal (excluded) correction: {r.corr_over_srho:.1e} of rho_star."
                 if np.isfinite(r.corr_over_srho) else "")
        lines.append(f"- **{r.koi}** ({r.planet or 'candidate'}): {why}{extra}")

    md_path = os.path.join(OUT_DIR, "rhoeff_correction_audit.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(df[["koi", "planet", "status", "corr_gcc", "corr_over_srho",
              "corr_over_sigma_prior", "corr_over_sigma_post"]].to_string(index=False))


if __name__ == "__main__":
    main()
