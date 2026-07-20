#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization (golden-value) check for src/core/priors.py.

Pins the numerical output of all three prior-setting strategies so the
reorganization can be verified to preserve behavior exactly. The catalog path's
network dependency (`get_koi`) is bypassed with a fixed fake KOI record, so this
exercises the *transformation* math only -- which is what the refactor touches.

Run with the project venv and src/core on the path:

    PYTHONPATH=src/core .venv/bin/python tests/golden_priors.py

First run writes tests/golden_priors.json (the baseline). Later runs compare
against it: names/distributions/dict-keys must match exactly; mu/sigma (floats
from catalog math) are compared with math.isclose.
"""
import json
import math
import os
import sys

import priors


# --- Fixtures ---------------------------------------------------------------

class FakeKOI:
    """Duck-typed stand-in for data._KOIRecord (attribute access -> float|None)."""
    def __init__(self, **fields):
        self.__dict__.update(fields)
    def __getattr__(self, name):  # missing fields behave like a masked archive cell
        return None


# Realistic KOI-like values; the exact numbers don't matter, only that the
# baseline and post-refactor runs feed identical inputs.
_KOI_FULL = dict(
    koi_num_transits=42,
    koi_period=12.3456789, koi_period_err1=1.2e-4, koi_period_err2=-1.1e-4,
    koi_impact=0.345, koi_impact_err1=0.08, koi_impact_err2=-0.07,
    koi_srho=1.234, koi_srho_err1=0.10, koi_srho_err2=-0.12,
    koi_ror=0.0456, koi_ror_err1=7.0e-4, koi_ror_err2=-6.0e-4,
    koi_longp=88.0, koi_eccen=0.13,
)
# Eccentricity-missing variant exercises the secw/sesw -> 0.0 fallback.
_KOI_NO_ECC = dict(_KOI_FULL, koi_longp=None, koi_eccen=None)


class FakeTA:
    """Records set_prior(name, dist, a, b) calls keyed by parameter name.

    Keyed (not ordered) because PyTransit's set_prior resolves by name, so call
    order carries no behavior -- the R4 reorder of the synthetic path must NOT
    register as a change here. A given parameter is only ever set once.
    """
    def __init__(self):
        self.priors = {}
    def set_prior(self, name, dist, a, b):
        assert name not in self.priors, f"parameter {name} set twice"
        self.priors[name] = [dist, float(a), float(b)]


# --- Capture each pathway ---------------------------------------------------

def _capture():
    results = {}

    # 1. Catalog path -- bypass network get_koi with the fake record. koi_ld_prior is ALSO stubbed
    # to None: it has its own get_koi/PyLDTk path that would otherwise hit the network for the real
    # KOI 999.01 and append q1/q2 keys, defeating this test's network-free "transformation math only"
    # contract. The limb-darkening prior is exercised elsewhere; here we pin the catalog core math.
    orig_get_koi = priors.get_koi
    orig_ld = priors.koi_ld_prior
    try:
        priors.koi_ld_prior = lambda koi_number: None
        priors.get_koi = lambda koi_number: FakeKOI(**_KOI_FULL)
        results["catalog_full"] = _spec_to_jsonable(priors.koi_prior_spec(999.01))
        priors.get_koi = lambda koi_number: FakeKOI(**_KOI_NO_ECC)
        results["catalog_no_ecc"] = _spec_to_jsonable(priors.koi_prior_spec(999.01))
    finally:
        priors.get_koi = orig_get_koi
        priors.koi_ld_prior = orig_ld

    # 2. Synthetic path -- all four ld_uniform x noise_uniform combinations.
    for ld in (True, False):
        for noise in (True, False):
            ta = FakeTA()
            _, prior_dict = priors.set_synthetic_priors(
                ta, num_transits=5, period=3650.0, impact_param=0.3, rho=1.0,
                planet_star_ratio=0.3, secw_1_value=0.0, sesw_1_value=0.0,
                ld_uniform=ld, noise_uniform=noise)
            key = f"synthetic_ld{int(ld)}_noise{int(noise)}"
            results[key] = {"priors": ta.priors, "prior_dict": prior_dict}

    # 3. Posterior path -- median/uncertainty math on a tiny synthetic dict.
    ta = FakeTA()
    sample = {
        "b_1":  [[0.30, 0.05, 0.04], [0.32, 0.06, 0.05], [0.28, 0.04, 0.06]],
        "t14":  [[0.10, 0.01, 0.02], [0.11, 0.02, 0.01]],
        "rho":  [[1.20, 0.10, 0.11], [1.25, 0.12, 0.09]],  # not in allowed -> skipped
    }
    priors.set_posterior_priors(ta, sample, allowed_params=["b_1", "t14"], factor=2)
    results["posterior"] = {"priors": ta.priors}

    return results


def _spec_to_jsonable(spec):
    # Key by parameter name (order-insensitive): set_prior is name-keyed, so the
    # spec's list order is not part of the behavioral contract.
    return {name: [dist, float(a), float(b)] for (name, dist, a, b) in spec}


# --- Compare ----------------------------------------------------------------

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _diff(path, a, b, errs):
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            errs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
            return
        if a.keys() != b.keys():
            errs.append(f"{path}: keys {sorted(a)} != {sorted(b)}")
            return
        for k in a:
            _diff(f"{path}.{k}", a[k], b[k], errs)
    elif isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            errs.append(f"{path}: list shape mismatch")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, errs)
    elif _is_number(a) and _is_number(b):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
            errs.append(f"{path}: {a!r} != {b!r}")
    else:  # strings (names, dists, dict types) -> exact
        if a != b:
            errs.append(f"{path}: {a!r} != {b!r}")


def main():
    golden_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "golden_priors.json")
    current = _capture()

    if not os.path.exists(golden_path):
        with open(golden_path, "w") as fh:
            json.dump(current, fh, indent=2, sort_keys=True)
        print(f"BASELINE WRITTEN -> {golden_path}")
        return 0

    with open(golden_path) as fh:
        golden = json.load(fh)

    errs = []
    _diff("root", golden, current, errs)
    if errs:
        print("GOLDEN MISMATCH:")
        for e in errs:
            print("  -", e)
        return 1
    print("GOLDEN OK -- all three prior pathways match baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
