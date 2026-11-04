r"""Inflation-linked bonds (US TIPS conventions) and their deflation floor.

Coupons and principal are paid on the real notional times the index ratio
:math:`\mathrm{RefCPI}(d)/\mathrm{RefCPI}(\text{dated date})`. At maturity the
principal is floored at par: the holder receives
:math:`100\max(\mathrm{IR}_T, 1)`. Prices are quoted in real terms, as the
unadjusted price per 100 at a semiannual real yield. Treasury's formula,
with simple interest over the fraction :math:`r/s` of the current period, is

.. math::

    P = \frac{C/2 + (C/2)\,a_n + 100\,v^n}{1 + (r/s)(i/2)} - \frac{s - r}{s}\frac C2,
    \qquad v = \frac{1}{1 + i/2},\quad a_n = \frac{1 - v^n}{i/2},

and the settlement amount is :math:`(P + A)` times the index ratio, where
:math:`A` is the accrued real coupon. On curves, each cash flow is worth its
real amount times :math:`P_n(0,t)\,\hat I(t)/I_{\text{base}}`. The floor is a
zero-coupon inflation put, priced in :class:`JarrowYildirim`.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Optional

from scipy.optimize import brentq

from spxa.market.dates import DateLike, add_months, to_date
from spxa.rates.inflation import InflationCurve, InflationIndex
from spxa.rates.jarrow_yildirim import JarrowYildirim

__all__ = ["InflationLinkedBond"]


@dataclass(frozen=True)
class InflationLinkedBond:
    """A TIPS-style bond: real annual ``coupon`` (decimal) paid ``frequency`` times a year."""

    coupon: float
    maturity: _dt.date
    dated_date: _dt.date
    index: InflationIndex
    frequency: int = 2
    face: float = 100.0
    deflation_floor: bool = True
    base_cpi: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity", to_date(self.maturity))
        object.__setattr__(self, "dated_date", to_date(self.dated_date))
        if self.maturity <= self.dated_date or 12 % self.frequency:
            raise ValueError("need maturity after the dated date and a frequency dividing 12")

    @property
    def base(self) -> float:
        """Reference CPI on the dated date (the base CPI of the issue)."""
        return (
            self.base_cpi
            if self.base_cpi is not None
            else self.index.reference_cpi(self.dated_date)
        )

    def coupon_dates(self) -> list[_dt.date]:
        """Coupon dates after the dated date, rolled back from maturity."""
        step, out, k = 12 // self.frequency, [], 0
        while True:
            d = add_months(self.maturity, -step * k)
            if d <= self.dated_date:
                break
            out.append(d)
            k += 1
        return out[::-1]

    def _period(self, settlement: DateLike) -> tuple[_dt.date, _dt.date, int]:
        s = to_date(settlement)
        dates = self.coupon_dates()
        nxt = next((d for d in dates if d > s), None)
        if nxt is None:
            raise ValueError("settlement is on or after maturity")
        prev = add_months(nxt, -12 // self.frequency)
        return prev, nxt, len(dates) - dates.index(nxt) - 1

    def accrued(self, settlement: DateLike) -> float:
        """Unadjusted accrued real interest per ``face``."""
        prev, nxt, _ = self._period(settlement)
        s, r = (nxt - prev).days, (nxt - to_date(settlement)).days
        return (s - r) / s * self.coupon * self.face / self.frequency

    def real_price(self, settlement: DateLike, real_yield: float) -> float:
        """Unadjusted clean price per ``face`` at a real yield (Treasury's formula)."""
        prev, nxt, n = self._period(settlement)
        s, r = (nxt - prev).days, (nxt - to_date(settlement)).days
        c, y = self.coupon * self.face / self.frequency, real_yield / self.frequency
        v = 1.0 / (1.0 + y)
        an = n if y == 0 else (1.0 - v**n) / y
        return (c + c * an + self.face * v**n) / (1.0 + r / s * y) - (s - r) / s * c

    def real_yield(self, settlement: DateLike, real_price: float) -> float:
        """Inverse of :meth:`real_price`."""
        return float(
            brentq(lambda y: self.real_price(settlement, y) - real_price, -0.2, 1.0, xtol=1e-15)
        )

    def settlement_amount(
        self, settlement: DateLike, real_yield: float, digits: Optional[int] = 6
    ) -> float:
        """Adjusted price plus adjusted accrued interest, per ``face`` of original principal."""
        ratio = (
            self.index.index_ratio(settlement, self.dated_date)
            if self.base_cpi is None
            else (self.index.reference_cpi(settlement) / self.base_cpi)
        )

        def rnd(x: float) -> float:
            return x if digits is None else round(x, digits)

        p, a = rnd(self.real_price(settlement, real_yield)), rnd(self.accrued(settlement))
        return rnd(rnd(p * ratio) + rnd(a * ratio))

    # -- curve and model prices -------------------------------------------------
    def present_value(
        self,
        nominal: Any,
        inflation: InflationCurve,
        today: DateLike,
        model: Optional[JarrowYildirim] = None,
    ) -> dict[str, float]:
        """Dirty present value per ``face`` today, in money, split into coupons, principal and floor.

        Without ``model`` the floor is valued at intrinsic on the forward index ratio.
        With a :class:`JarrowYildirim` model it is a zero-coupon inflation put struck
        at the base CPI.
        """
        t0 = to_date(today)
        base = self.base
        c = self.coupon * self.face / self.frequency
        coupons = sum(
            c * float(nominal.discount(d)) * inflation.reference_cpi(d) / base
            for d in self.coupon_dates()
            if d > t0
        )
        ratio_T = inflation.reference_cpi(self.maturity) / base
        df_T = float(nominal.discount(self.maturity))
        principal = self.face * df_T * ratio_T
        floor = 0.0
        if self.deflation_floor:
            if model is None:
                floor = self.face * df_T * max(1.0 - ratio_T, 0.0)
            else:
                T = float(nominal.time(self.maturity))
                i0 = inflation.reference_cpi(t0)
                strike = (base / i0) ** (1.0 / T) - 1.0  # I(T)/I(0) <= base/I(0)
                floor = self.face * (i0 / base) * model.zero_coupon_caplet(T, strike, "put")
        return {
            "coupons": coupons,
            "principal": principal,
            "floor": floor,
            "total": coupons + principal + floor,
        }

    def breakeven_inflation(self, nominal_yield: float, real_yield: float) -> float:
        """Fisher breakeven from semiannual nominal and real yields."""
        f = self.frequency
        return ((1 + nominal_yield / f) / (1 + real_yield / f)) ** f - 1.0

    def __repr__(self) -> str:
        return f"InflationLinkedBond({self.coupon:.4%} {self.maturity:%Y-%m-%d})"
