#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: feynmanliu

TransitAnalysis with consistent eccentric mid-transit geometry (real-Kepler individual fits).

Upstream pytransit 2.7.1 ``TransitAnalysis`` mixes geometries:
  - ``transit_model`` computes the inclination with the CIRCULAR ``i_from_ba`` even though it
    passes the sampled (e, w) to the evaluator, so ``b`` is not the mid-transit impact
    parameter whenever e != 0;
  - ``posterior_samples`` derives the inclination with the eccentric ``i_from_baew`` but then
    hard-codes e = 0, w = 0 into the t14/t23 durations (``d_from_pkaiews``).

Both overrides below are verbatim copies of the pinned 2.7.1 methods with only the geometry
corrected: the flux model uses cos i = (b/a)(1 + e sin w)/(1 - e^2) (the ``i_from_baew``
convention), and the derived durations use the *sampled* e, w. Parameter draws with no valid
transit geometry (e >= 1 or |cos i| > 1) get flux = +inf, which the Gaussian likelihood maps
to a -inf log-probability -- a per-sample rejection, never a clamp.

The overrides copy 2.7.1 internals, so tests/test_model_corrections.py pins the upstream
method sources by hash: a silent PyTransit upgrade fails the test instead of silently
diverging from these copies.
"""

import arviz as az
import numpy as np
import xarray as xa
from numpy import arange, arctan2, atleast_2d, inf, ones, sqrt, squeeze

from pytransit import TransitAnalysis
from pytransit.lpf.lpf import map_ldc
from pytransit.orbits import as_from_rhop, d_from_pkaiews, i_from_baew


class EccentricTransitAnalysis(TransitAnalysis):
    """Drop-in replacement for TransitAnalysis on the real-Kepler per-transit fit path.

    Also inherits the ``nsamples``/``exptimes`` constructor arguments used for the
    long-cadence finite-exposure integration (SS4)."""

    def transit_model(self, pv, copy=True, planets=None):
        # Verbatim upstream 2.7.1 except the inclination line (was i_from_ba) and the
        # invalid-geometry rejection mask.
        pv = atleast_2d(pv)
        flux = ones([pv.shape[0], self.timea.size])
        ldc = map_ldc(pv[:, self._sl_ld])
        planets = planets if planets is not None else arange(self.nplanets)
        for ipl in planets:
            ist = 6 * ipl
            t0 = pv[:, 1 + ist]
            p = pv[:, 2 + ist]
            k = sqrt(pv[:, 4 + ist: 5 + ist])
            aor = as_from_rhop(pv[:, 0], p)
            ecc = pv[:, 5 + ist] ** 2 + pv[:, 6 + ist] ** 2
            w = arctan2(pv[:, 6 + ist], pv[:, 5 + ist])
            # Eccentric mid-transit inclination: cos i = (b/a)(1 + e sin w)/(1 - e^2).
            with np.errstate(invalid='ignore', divide='ignore'):
                cosi_arg = pv[:, 3 + ist] / aor * (1.0 + ecc * np.sin(w)) / (1.0 - ecc ** 2)
            bad = (~np.isfinite(cosi_arg)) | (np.abs(cosi_arg) > 1.0) | (ecc >= 1.0) | \
                  ~(aor > 0.0)
            inc = np.arccos(np.clip(cosi_arg, -1.0, 1.0))
            f = atleast_2d(self.tm.evaluate(k, ldc, t0, p, aor, inc, ecc, w, copy))
            if bad.any():
                f = f.copy()
                f[bad, :] = inf
            flux += f - 1.
        return squeeze(flux)

    def posterior_samples(self):
        # Verbatim upstream 2.7.1 except that t14/t23 use the sampled (e, w) instead of the
        # hard-coded (0., 0.).
        dd = az.from_emcee(self.sampler, var_names=self.ps.names)
        ds = xa.Dataset()
        pst = dd.posterior
        c = pst.rho.coords
        DA = xa.DataArray
        for i in range(1, self.nplanets + 1):
            p = pst[f'p_{i}'].values
            ds[f'k_{i}'] = k = DA(sqrt(pst[f'k2_{i}']), coords=c)
            ds[f'a_{i}'] = a = DA(as_from_rhop(pst.rho.values, p), coords=c)
            ds[f'e_{i}'] = e = DA(pst[f'secw_{i}'] ** 2 + pst[f'sesw_{i}'] ** 2, coords=c)
            ds[f'w_{i}'] = w = DA(arctan2(pst[f'sesw_{i}'], pst[f'secw_{i}']), coords=c)
            ds[f'i_{i}'] = inc = DA(i_from_baew(pst[f'b_{i}'].values, a.values, e.values,
                                                w.values), coords=c)
            ds[f't14_{i}'] = DA(d_from_pkaiews(p, k.values, a.values, inc.values,
                                               e.values, w.values, 1, kind=14), coords=c)
            ds[f't23_{i}'] = DA(d_from_pkaiews(p, k.values, a.values, inc.values,
                                               e.values, w.values, 1, kind=23), coords=c)
        dd.add_groups({'derived_parameters': ds})
        return dd
