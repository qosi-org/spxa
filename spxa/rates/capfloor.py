r"""Caps and floors on term-rate indices.

A caplet on the coupon :math:`(t_s, t_e]`, which fixes at :math:`T_f` and pays
at :math:`t_p`, is worth

.. math:: N\tau\,P(t_p)\,\mathrm{Black}(F, K, T_f, \sigma)

under the :math:`t_p`-forward measure, where :math:`F` is the projected
index forward. The Black formula can be replaced by Bachelier (normal
vol) or by a shifted lognormal for low or negative rates. A cap is the
sum of its caplets. The first period usually fixes at trade time and is
left out (``skip_first``). Put-call parity makes cap − floor equal to the
floating leg minus :math:`K` times the floating annuity
:math:`\sum N\tau P(t_p)`.

The time to fixing is measured ACT/365F from the discount curve's
reference date. The volatility may be one flat number, as quoted for
caps, or a function ``vol(T_f, K)`` such as a stripped caplet surface.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq

from spxa.market.black import bachelier_price, black_price
from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike
from spxa.market.daycount import DayCount
from spxa.market.schedule import Schedule
from spxa.rates.cashflows import IborCoupon, ibor_leg
from spxa.rates.indices import EURIBOR6M, IborIndex
from spxa.rates.swaps import _dates

__all__ = ["CapFloor", "Caplet", "cap_floor", "VolLike"]

VolLike = Union[float, Callable[[float, float], float]]
_MODELS = ("black", "normal", "shifted_black")
_ACT365 = DayCount("ACT/365F")


def _vol(vol: VolLike, T: float, K: float) -> float:
    return float(vol(T, K)) if callable(vol) else float(vol)


def option_price(
    F: float, K: float, T: float, sigma: float, df: float, kind: str, model: str, shift: float
) -> float:
    """Undiscounted-by-accrual optionlet value ``df * E[(eta (F_T - K))^+]``."""
    opt = "call" if kind == "cap" else "put"
    if model == "normal":
        return float(bachelier_price(F, K, T, sigma, df, opt))
    s = shift if model == "shifted_black" else 0.0
    if F + s <= 0 or K + s <= 0:
        raise ValueError("lognormal model needs F + shift > 0 and K + shift > 0")
    return float(black_price(F + s, K + s, T, sigma, df, opt))


@dataclass(frozen=True)
class Caplet:
    """A caplet or floorlet on one term-rate coupon."""

    coupon: IborCoupon
    strike: float
    kind: str = "cap"

    def __post_init__(self) -> None:
        if self.kind not in ("cap", "floor"):
            raise ValueError("kind must be 'cap' or 'floor'")

    def expiry(self, discount: YieldCurve) -> float:
        """Year fraction (ACT/365F) from the curve's reference date to the fixing."""
        if discount.reference_date is None:
            raise ValueError("discount curve needs a reference_date")
        return _ACT365.year_fraction(discount.reference_date, self.coupon.fixing)

    def forward(self, projection: YieldCurve) -> float:
        return self.coupon.index_rate(projection)

    def weight(self, discount: YieldCurve) -> float:
        """:math:`N\\tau P(t_p)`."""
        c = self.coupon
        return c.notional * c.accrual * float(discount.discount(c.payment_date))

    def price(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve],
        vol: VolLike,
        model: str = "black",
        shift: float = 0.0,
    ) -> float:
        if model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}")
        proj = discount if projection is None else projection
        T = self.expiry(discount)
        if T < 0:
            raise ValueError("caplet has already fixed")
        F = self.forward(proj)
        sigma = _vol(vol, T, self.strike)
        return self.weight(discount) * option_price(
            F, self.strike, T, sigma, 1.0, self.kind, model, shift
        )


@dataclass(frozen=True)
class CapFloor:
    """A strip of caplets (``kind='cap'``) or floorlets (``kind='floor'``)."""

    caplets: tuple[Caplet, ...]
    strike: float
    kind: str = "cap"

    @property
    def maturity(self) -> _dt.date:
        return self.caplets[-1].coupon.payment_date

    def price(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        vol: VolLike = 0.2,
        model: str = "black",
        shift: float = 0.0,
    ) -> float:
        return sum(c.price(discount, projection, vol, model, shift) for c in self.caplets)

    def caplet_prices(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        vol: VolLike = 0.2,
        model: str = "black",
        shift: float = 0.0,
    ) -> np.ndarray:
        return np.array([c.price(discount, projection, vol, model, shift) for c in self.caplets])

    def annuity(self, discount: YieldCurve) -> float:
        """Floating annuity :math:`\\sum N\\tau P(t_p)` over the optionlets."""
        return sum(c.weight(discount) for c in self.caplets)

    def atm_strike(self, discount: YieldCurve, projection: Optional[YieldCurve] = None) -> float:
        """Strike at which the cap and the floor are worth the same (forward swap rate of the strip)."""
        proj = discount if projection is None else projection
        return sum(c.weight(discount) * c.forward(proj) for c in self.caplets) / self.annuity(
            discount
        )

    def with_strike(self, strike: float, kind: Optional[str] = None) -> "CapFloor":
        k = self.kind if kind is None else kind
        return CapFloor(tuple(Caplet(c.coupon, strike, k) for c in self.caplets), strike, k)

    def implied_flat_vol(
        self,
        price: float,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        model: str = "black",
        shift: float = 0.0,
        bounds: Sequence[float] = (1e-6, 5.0),
    ) -> float:
        """The single volatility at which every optionlet reproduces ``price``."""
        lo, hi = bounds

        def f(s: float) -> float:
            return self.price(discount, projection, s, model, shift) - price

        if f(lo) > 0 or f(hi) < 0:
            raise ValueError("price outside the attainable range")
        return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))


def cap_floor(
    trade_date: DateLike,
    tenor: str,
    strike: float,
    index: IborIndex = EURIBOR6M,
    notional: float = 1.0,
    kind: str = "cap",
    skip_first: bool = True,
    forward_start: Optional[str] = None,
    use_index_dates: bool = True,
) -> CapFloor:
    """Cap or floor on ``index`` from its spot date, rolled like :func:`vanilla_swap` floating legs."""
    start, end = _dates(trade_date, tenor, index.calendar, index.fixing_days, forward_start)
    sch = Schedule(start, end, index.tenor, index.calendar, "MF", end_of_month=index.end_of_month)
    leg: list[Any] = ibor_leg(sch, notional, index, use_index_dates=use_index_dates)
    if skip_first and forward_start is None:
        leg = leg[1:]
    if not leg:
        raise ValueError("no optionlets left")
    return CapFloor(tuple(Caplet(c, strike, kind) for c in leg), strike, kind)
