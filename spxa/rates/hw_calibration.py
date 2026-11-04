"""Calibration of the Hull–White model to swaptions and caps.

Two modes:

* ``'bootstrap'``: the mean reversion is fixed and the volatility is
  piecewise constant, with one level per instrument and knots at the
  instruments' expiries. A European option expiring at :math:`T` depends
  on :math:`\\sigma` only on :math:`[0, T]`, so the levels are solved one
  at a time in expiry order and every instrument is matched exactly. This
  is the usual calibration to co-terminal or diagonal swaptions before
  pricing a Bermudan.
* ``'global'``: least squares over the mean reversion (unless fixed) and
  a constant volatility, or one level per given ``sigma_times`` piece,
  fitted to any set of swaptions and caps.

Targets are prices. :func:`market_prices` converts Black, shifted Black or
normal vols with the instruments' own pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

from spxa.market.curves import YieldCurve
from spxa.rates.capfloor import CapFloor, Caplet
from spxa.rates.hull_white import HullWhite
from spxa.rates.swaptions import Swaption

__all__ = ["HullWhiteCalibration", "calibrate_hull_white", "market_prices", "model_price"]


def model_price(model: HullWhite, inst: Any, projection: Optional[YieldCurve] = None) -> float:
    """Price of a swaption, cap/floor or caplet under ``model``."""
    if isinstance(inst, Swaption):
        return model.swaption(inst, projection)
    if isinstance(inst, CapFloor):
        return model.cap(inst, projection)
    if isinstance(inst, Caplet):
        return model.caplet(inst, projection)
    raise TypeError(f"cannot price {type(inst).__name__}")


def _expiry(curve: YieldCurve, inst: Any) -> float:
    if isinstance(inst, Swaption):
        return float(curve.time(inst.expiry))
    if isinstance(inst, CapFloor):
        return float(curve.time(inst.caplets[-1].coupon.fixing))
    if isinstance(inst, Caplet):
        return float(curve.time(inst.coupon.fixing))
    raise TypeError(f"cannot calibrate to {type(inst).__name__}")


def market_prices(
    instruments: Sequence[Any],
    vols: Sequence[float],
    discount: YieldCurve,
    projection: Optional[YieldCurve] = None,
    model: str = "black",
    shift: float = 0.0,
) -> np.ndarray:
    """Prices of ``instruments`` at the quoted ``vols``."""
    proj = projection if projection is not None else discount
    return np.array(
        [inst.price(discount, proj, v, model, shift) for inst, v in zip(instruments, vols)]
    )


@dataclass(frozen=True)
class HullWhiteCalibration:
    model: HullWhite
    prices: np.ndarray
    targets: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return self.prices - self.targets

    @property
    def max_relative_error(self) -> float:
        return float(np.max(np.abs(self.errors / self.targets)))


def calibrate_hull_white(
    curve: YieldCurve,
    instruments: Sequence[Any],
    targets: Sequence[float],
    projection: Optional[YieldCurve] = None,
    a: Optional[float] = 0.03,
    mode: str = "bootstrap",
    sigma_times: Optional[Sequence[float]] = None,
    sigma_bounds: tuple[float, float] = (1e-6, 0.2),
) -> HullWhiteCalibration:
    """Fit :class:`HullWhite` on ``curve`` to instrument prices ``targets``.

    ``a=None`` (global mode only) calibrates the mean reversion too.
    """
    tgt = np.asarray(targets, float)
    if len(instruments) != tgt.size or tgt.size == 0:
        raise ValueError("need one target per instrument")
    if mode == "bootstrap":
        if a is None:
            raise ValueError("bootstrap mode needs a fixed mean reversion")
        order = np.argsort([_expiry(curve, i) for i in instruments], kind="stable")
        T = np.array([_expiry(curve, instruments[k]) for k in order])
        if np.any(np.diff(T) <= 0):
            raise ValueError("bootstrap needs distinct expiries")
        knots = T[:-1]
        levels = np.full(T.size, 0.01)
        for n, k in enumerate(order):

            def f(s: float, n: int = n, k: int = k) -> float:
                lv = levels.copy()
                lv[n:] = s
                return (
                    model_price(HullWhite(curve, a, lv, knots), instruments[k], projection) - tgt[k]
                )

            lo, hi = sigma_bounds
            if f(lo) * f(hi) > 0:
                raise ValueError(f"instrument {k} cannot be matched within sigma_bounds")
            levels[n:] = brentq(f, lo, hi, xtol=1e-14, rtol=1e-13)
        model = HullWhite(curve, a, levels, knots)
    elif mode == "global":
        st = np.asarray([] if sigma_times is None else sigma_times, float)
        n_sig = st.size + 1

        def build(p: np.ndarray) -> HullWhite:
            aa = float(np.exp(p[n_sig])) if a is None else float(a)
            return HullWhite(curve, aa, np.exp(p[:n_sig]), st)

        def resid(p: np.ndarray) -> np.ndarray:
            m = build(p)
            return np.array([model_price(m, i, projection) for i in instruments]) / tgt - 1.0

        p0 = np.r_[np.full(n_sig, np.log(0.01)), [] if a is not None else [np.log(0.05)]]
        sol = least_squares(resid, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        model = build(sol.x)
    else:
        raise ValueError("mode must be 'bootstrap' or 'global'")
    prices = np.array([model_price(model, i, projection) for i in instruments])
    return HullWhiteCalibration(model, prices, tgt)
