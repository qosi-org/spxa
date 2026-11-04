"""Calibration of G2++ to swaptions and caps.

The five parameters are fitted by least squares on relative price errors,
with the parametrisation :math:`a = b + e^{p_0}`, :math:`\\sigma = e^{p_1}`,
:math:`b = e^{p_2}` (so :math:`x` is the fast factor),
:math:`\\eta = e^{p_3}` and :math:`\\rho = \\tanh p_4`. Swap cashflows are
mapped once (:func:`~spxa.rates.model_cashflows.swap_cashflows`) and every
swaption is priced by the one-dimensional formula with a Gauss–Hermite
rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.market.curves import YieldCurve
from spxa.rates.capfloor import CapFloor
from spxa.rates.g2pp import G2PP
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import Swaption

__all__ = ["G2PPCalibration", "calibrate_g2pp"]


@dataclass(frozen=True)
class G2PPCalibration:
    model: G2PP
    prices: np.ndarray
    targets: np.ndarray
    nfev: int

    @property
    def max_relative_error(self) -> float:
        return float(np.max(np.abs(self.prices / self.targets - 1)))


def calibrate_g2pp(
    curve: YieldCurve,
    instruments: Sequence[Any],
    targets: Sequence[float],
    projection: Optional[YieldCurve] = None,
    initial: tuple[float, float, float, float, float] = (0.5, 0.01, 0.05, 0.008, -0.5),
    n_quad: int = 24,
) -> G2PPCalibration:
    """Fit ``(a, sigma, b, eta, rho)`` to instrument prices. ``initial`` must have ``a > b``."""
    tgt = np.asarray(targets, float)
    if len(instruments) != tgt.size or tgt.size < 5:
        raise ValueError("need at least five instruments, one target each")
    a0, s0, b0, e0, r0 = initial
    if not a0 > b0:
        raise ValueError("initial guess needs a > b")
    pricers = []
    for inst in instruments:
        if isinstance(inst, Swaption):
            T = float(curve.time(inst.expiry))
            cf = swap_cashflows(inst.swap, curve, projection)
            live = cf.times > T + 1e-12
            args = (T, cf.times[live], cf.amounts[live], float(cf.amounts[~live].sum()))
            pricers.append(lambda m, args=args: m.option_on_cashflows(*args, n_quad=n_quad))
        elif isinstance(inst, CapFloor):
            pricers.append(lambda m, inst=inst: m.cap(inst, projection))
        else:
            raise TypeError(f"cannot calibrate to {type(inst).__name__}")

    def build(p: np.ndarray) -> G2PP:
        b = float(np.exp(p[2]))
        a = b + float(np.exp(p[0]))
        return G2PP(curve, a, float(np.exp(p[1])), b, float(np.exp(p[3])), float(np.tanh(p[4])))

    def resid(p: np.ndarray) -> np.ndarray:
        m = build(p)
        return np.array([f(m) for f in pricers]) / tgt - 1.0

    p0 = np.array([np.log(a0 - b0), np.log(s0), np.log(b0), np.log(e0), np.arctanh(r0)])
    sol = least_squares(resid, p0, xtol=1e-15, ftol=1e-15, gtol=1e-15, x_scale="jac")
    model = build(sol.x)
    return G2PPCalibration(model, np.array([f(model) for f in pricers]), tgt, int(sol.nfev))
