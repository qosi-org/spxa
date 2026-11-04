"""Calibration of the Cheyette model to swaption smiles.

With ``kappa`` fixed, the piecewise-constant level :math:`\\sigma(t)` (one
level per distinct expiry, as in :func:`~spxa.rates.hw_calibration.calibrate_hull_white`)
and the global skew :math:`\\beta` are fitted by least squares. The
residuals are relative price errors, each priced with
:func:`~spxa.rates.cheyette_swaption.cheyette_swaption`. Strikes on both
sides of the money at each expiry pin down :math:`\\beta`, and the ATM
quotes pin down the levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.market.curves import YieldCurve
from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_swaption import cheyette_swaption

__all__ = ["CheyetteCalibration", "calibrate_cheyette"]


@dataclass(frozen=True)
class CheyetteCalibration:
    model: Cheyette
    prices: np.ndarray
    targets: np.ndarray

    @property
    def max_relative_error(self) -> float:
        return float(np.max(np.abs(self.prices / self.targets - 1)))


def calibrate_cheyette(
    curve: YieldCurve,
    swaptions: Sequence[Any],
    targets: Sequence[float],
    projection: Optional[YieldCurve] = None,
    kappa: float = 0.03,
    initial: tuple[float, float] = (0.01, 0.0),
    beta_bounds: tuple[float, float] = (-2.0, 5.0),
) -> CheyetteCalibration:
    tgt = np.asarray(targets, float)
    if len(swaptions) != tgt.size:
        raise ValueError("need one target per swaption")
    expiries = sorted({float(curve.time(s.expiry)) for s in swaptions})
    if tgt.size < len(expiries) + 1:
        raise ValueError("need more quotes than levels: add strikes away from the money")
    knots = expiries[:-1]

    def build(p: np.ndarray) -> Cheyette:
        return Cheyette(curve, kappa, np.exp(p[:-1]), float(p[-1]), knots)

    def resid(p: np.ndarray) -> np.ndarray:
        m = build(p)
        return (
            np.array([cheyette_swaption(m, s, projection, n_time=64) for s in swaptions]) / tgt
            - 1.0
        )

    p0 = np.r_[np.full(len(expiries), np.log(initial[0])), initial[1]]
    lo = np.r_[np.full(len(expiries), np.log(1e-5)), beta_bounds[0]]
    hi = np.r_[np.full(len(expiries), np.log(0.2)), beta_bounds[1]]
    sol = least_squares(
        resid, p0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14, x_scale="jac"
    )
    model = build(sol.x)
    prices = np.array([cheyette_swaption(model, s, projection, n_time=64) for s in swaptions])
    return CheyetteCalibration(model, prices, tgt)
