r"""
Calibration of the LIBOR market model to caplets and swaptions.

The usual two-stage procedure (Brigo & Mercurio, ch. 7; Rebonato 2004):

1. **Caplets.** The abcd shape is fitted to the caplet volatilities by least
   squares, and the scalings :math:`k_i` then make every caplet exact,

   .. math:: k_i^2 = \frac{\sigma_{\mathrm{cpl}, i}^2 T_i}{\int_0^{T_i} f(T_i - t)^2\,dt}.

   Values of :math:`k_i` near one mean the time-homogeneous shape explains
   the caplet curve on its own.

2. **Swaptions.** With the volatilities fixed, the correlation parameters
   (:math:`\beta` and :math:`\rho_\infty` of :func:`rebonato_correlation`)
   are fitted to swaption volatilities through Rebonato's formula, which
   is fast because no simulation is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.market.curves import YieldCurve
from spxa.rates.lmm import AbcdVol, LiborMarketModel, rebonato_correlation

__all__ = ["LMMCalibration", "calibrate_lmm", "fit_abcd"]


def fit_abcd(
    times: Sequence[float],
    caplet_vols: Sequence[float],
    initial: Sequence[float] = (0.05, 0.1, 0.5, 0.1),
) -> AbcdVol:
    r"""Least-squares abcd fit of caplet vols :math:`\sqrt{\int_0^{T_i} f^2 / T_i}` (``a + d > 0``, ``c, d > 0``)."""
    T = np.asarray(times, float)
    target = np.asarray(caplet_vols, float)

    def model(p: np.ndarray) -> np.ndarray:
        f = AbcdVol(*p)
        return np.array([math.sqrt(max(f.covariance(t, t, 0.0, t), 0.0) / t) for t in T])

    res = least_squares(
        lambda p: model(p) - target,
        np.asarray(initial, float),
        bounds=([-1.0, -2.0, 1e-3, 1e-4], [2.0, 5.0, 10.0, 2.0]),
        x_scale="jac",
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    return AbcdVol(*res.x)


@dataclass(frozen=True)
class LMMCalibration:
    model: LiborMarketModel
    beta: float
    rho_inf: float
    swaption_vols: np.ndarray
    targets: np.ndarray

    @property
    def max_swaption_error(self) -> float:
        return float(np.abs(self.swaption_vols - self.targets).max())


def calibrate_lmm(
    curve: YieldCurve,
    times: Sequence[float],
    caplet_vols: Sequence[float],
    swaptions: Sequence[tuple[int, int]],
    swaption_vols: Sequence[float],
    shift: float = 0.0,
    initial_correlation: tuple[float, float] = (0.1, 0.5),
) -> LMMCalibration:
    """Caplets exactly (abcd + ``k_i``), then ``(beta, rho_inf)`` to swaption vols ``(alpha, beta)``.

    ``caplet_vols[i]`` is the Black vol of the caplet on forward ``i``
    (fixing at ``times[i]``); the first fixing must be after zero.
    """
    T = np.asarray(times, float)
    fix = T[:-1]
    if fix[0] <= 0:
        raise ValueError("the first forward must fix after t = 0")
    cv = np.asarray(caplet_vols, float)
    shape = fit_abcd(fix, cv)
    k = np.array([cv[i] * math.sqrt(t / shape.covariance(t, t, 0.0, t)) for i, t in enumerate(fix)])
    targets = np.asarray(swaption_vols, float)

    def build(beta: float, rho_inf: float) -> LiborMarketModel:
        rho = rebonato_correlation(fix, beta, rho_inf)
        return LiborMarketModel(curve, T, shape, k=k, correlation=rho, shift=shift)

    def resid(p: np.ndarray) -> np.ndarray:
        m = build(p[0], p[1])
        return np.array([m.swaption_vol(a, b) for a, b in swaptions]) - targets

    res = least_squares(
        resid,
        np.asarray(initial_correlation, float),
        bounds=([1e-4, 0.0], [5.0, 0.999]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    model = build(*res.x)
    vols = np.array([model.swaption_vol(a, b) for a, b in swaptions])
    return LMMCalibration(model, float(res.x[0]), float(res.x[1]), vols, targets)
