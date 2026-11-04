r"""European swaptions.

A physically settled payer swaption with expiry :math:`T` on a swap with
fixed rate :math:`K` is a call on the forward swap rate :math:`S`, struck
at :math:`K`, under the annuity measure:

.. math:: V = A\,\mathrm{Black}(S, K, T, \sigma), \qquad
          A = \sum_i N\tau_i P(t_i), \qquad S = \frac{\text{floating leg}}{A}.

Receivers are puts, and payer − receiver is the forward swap. Bachelier and
shifted Black replace Black for normal or shifted quotes.

Cash-settled (par-yield) swaptions pay
:math:`\bar A(S_T)\,(S_T - K)^+` at the start date, with the *cash
annuity*

.. math:: \bar A(S) = \sum_{i=1}^{n} \frac{N/m}{(1 + S/m)^i}
          = N\,\frac{1 - (1 + S/m)^{-n}}{S}

on :math:`n` fixed periods at frequency :math:`m`. The market formula
values them as :math:`P(T_s)\,\bar A(S)\,\mathrm{Black}(S, K, T, \sigma)`.
This is a convention, not an arbitrage-free price: the exact value
needs a convexity adjustment (see the CMS replication tools).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from spxa.market.calendars import get_calendar
from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike
from spxa.market.daycount import DayCount
from spxa.rates.capfloor import option_price
from spxa.rates.cashflows import FixedCoupon
from spxa.rates.indices import EURIBOR6M, IborIndex
from spxa.rates.swaps import Swap, vanilla_swap

__all__ = ["Swaption", "cash_annuity", "swaption"]

_ACT365 = DayCount("ACT/365F")
_MODELS = ("black", "normal", "shifted_black")


def cash_annuity(S: float, n: int, m: int, notional: float = 1.0) -> float:
    """Par-yield annuity :math:`N(1 - (1 + S/m)^{-n})/S` (``n/m`` at ``S = 0``)."""
    if n < 1 or m < 1:
        raise ValueError("need n >= 1 and m >= 1")
    if S == 0.0:
        return notional * n / m
    if S <= -m:
        raise ValueError("par yield below -m")
    return notional * float(-np.expm1(-n * np.log1p(S / m))) / S


@dataclass(frozen=True)
class Swaption:
    """European option to enter ``swap`` at ``expiry`` (payer if ``swap.payer``)."""

    swap: Swap
    expiry: _dt.date
    settlement: str = "physical"

    def __post_init__(self) -> None:
        if self.settlement not in ("physical", "cash"):
            raise ValueError("settlement must be 'physical' or 'cash'")
        if self.expiry > self.swap.start:
            raise ValueError("expiry after the swap start")

    @property
    def strike(self) -> float:
        return self.swap.fixed_rate

    @property
    def is_payer(self) -> bool:
        return self.swap.payer

    @property
    def notional(self) -> float:
        c = self.swap.fixed[0]
        assert isinstance(c, FixedCoupon)
        return float(c.notional)

    def time_to_expiry(self, discount: YieldCurve) -> float:
        if discount.reference_date is None:
            raise ValueError("discount curve needs a reference_date")
        return _ACT365.year_fraction(discount.reference_date, self.expiry)

    def forward_rate(self, discount: YieldCurve, projection: Optional[YieldCurve] = None) -> float:
        return self.swap.par_rate(discount, projection)

    def frequency(self) -> int:
        """Fixed coupons per year, from the average fixed accrual."""
        acc = [c.accrual for c in self.swap.fixed if isinstance(c, FixedCoupon)]
        return max(1, round(len(acc) / sum(acc)))

    def annuity(self, discount: YieldCurve, projection: Optional[YieldCurve] = None) -> float:
        """Physical annuity, or :math:`P(T_s)\\bar A(S)` for cash settlement."""
        if self.settlement == "physical":
            return abs(self.swap.annuity(discount))
        S = self.forward_rate(discount, projection)
        n = len(self.swap.fixed)
        return float(discount.discount(self.swap.start)) * cash_annuity(
            S, n, self.frequency(), self.notional
        )

    def price(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        vol: float = 0.2,
        model: str = "black",
        shift: float = 0.0,
    ) -> float:
        if model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}")
        T = self.time_to_expiry(discount)
        if T < 0:
            raise ValueError("swaption has expired")
        S = self.forward_rate(discount, projection)
        kind = "cap" if self.is_payer else "floor"
        unit = option_price(S, self.strike, T, float(vol), 1.0, kind, model, shift)
        return self.annuity(discount, projection) * unit

    def implied_vol(
        self,
        price: float,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        model: str = "black",
        shift: float = 0.0,
        bounds: Sequence[float] = (1e-7, 5.0),
    ) -> float:
        lo, hi = bounds

        def f(s: float) -> float:
            return self.price(discount, projection, s, model, shift) - price

        if f(lo) > 0 or f(hi) < 0:
            raise ValueError("price outside the attainable range")
        return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-14))


def swaption(
    trade_date: DateLike,
    expiry: str,
    tenor: str,
    strike: float,
    index: IborIndex = EURIBOR6M,
    payer: bool = True,
    notional: float = 1.0,
    settlement: str = "physical",
    fixed_tenor: str = "1Y",
    fixed_day_count: str = "30/360",
) -> Swaption:
    """``expiry`` x ``tenor`` swaption; the option expires ``index.fixing_days`` before the swap start."""
    swap = vanilla_swap(
        trade_date,
        tenor,
        strike,
        index,
        notional,
        payer,
        fixed_tenor=fixed_tenor,
        fixed_day_count=fixed_day_count,
        forward_start=expiry,
    )
    cal = get_calendar(index.calendar)
    exp = cal.add_business_days(swap.start, -index.fixing_days)
    return Swaption(swap, exp, settlement)
