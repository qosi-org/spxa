r"""Fixed-rate bullet bonds.

Coupons accrue on the unadjusted schedule and are paid on the following
business day. With settlement in the period :math:`(t_{j-1}, t_j]`, the
street (ICMA) yield :math:`y` with compounding frequency :math:`f` prices
the bond as

.. math:: P_{\text{dirty}} = \sum_{i \ge j} \frac{C_i}{(1 + y/f)^{w + i - j}}, \qquad
          w = f\,\tau_{\text{ICMA}}(t_s, t_j),

so that :math:`w` is the fraction of the current period still to run. The
clean price is :math:`P_{\text{dirty}}` minus accrued interest
:math:`F c\,\tau(t_{j-1}, t_s)`. With :math:`e_i = w + i - j` and
:math:`v = 1/(1 + y/f)`:

* Macaulay duration :math:`D = \sum (e_i/f)\,C_i v^{e_i}/P`;
* modified duration :math:`D/(1 + y/f) = -P^{-1}\,\partial P/\partial y`;
* convexity :math:`\sum e_i(e_i + 1)C_i v^{e_i + 2}/(f^2 P) = P^{-1}\,\partial^2 P/\partial y^2`.

:meth:`FixedRateBond.dirty_price_from_curve` discounts the cashflows on a
curve (optionally with a z-spread on continuously compounded zero rates).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq

from spxa.market.calendars import get_calendar
from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, Tenor, to_date
from spxa.market.daycount import DayCount
from spxa.market.schedule import Schedule, adjust

__all__ = ["BondCashflow", "FixedRateBond"]


@dataclass(frozen=True)
class BondCashflow:
    accrual_start: _dt.date
    accrual_end: _dt.date
    payment_date: _dt.date
    amount: float
    is_redemption: bool = False


class FixedRateBond:
    """Bullet bond paying ``coupon`` (annual rate) at ``frequency``.

    Parameters
    ----------
    effective, maturity : date-like
        First accrual date and maturity (unadjusted).
    coupon : float
        Annual coupon rate.
    frequency : str
        Coupon period, ``"6M"`` (default) or ``"1Y"`` etc.
    day_count : str
        Accrual basis (default ``"ACT/ACT ICMA"``).
    calendar : str
        Payment calendar (payments roll to the following business day).
    settlement_days : int
        Business days from trade to settlement.
    face : float
        Redemption amount.
    first_coupon : date-like, optional
        End of an irregular first period.
    """

    def __init__(
        self,
        effective: DateLike,
        maturity: DateLike,
        coupon: float,
        frequency: str = "6M",
        day_count: str = "ACT/ACT ICMA",
        calendar: str = "US-SIFMA",
        settlement_days: int = 1,
        face: float = 100.0,
        first_coupon: Optional[DateLike] = None,
    ) -> None:
        self.coupon = float(coupon)
        self.face = float(face)
        self.day_count = DayCount(day_count)
        self.calendar = get_calendar(calendar)
        self.settlement_days = int(settlement_days)
        tenor = Tenor(frequency)
        self.frequency = 12 // tenor.months
        if tenor.months <= 0 or 12 % tenor.months:
            raise ValueError("frequency must divide a year")
        self.schedule = Schedule(
            effective,
            maturity,
            tenor,
            self.calendar,
            "U",
            end_of_month=True,
            first_date=first_coupon,
        )
        self.maturity = to_date(maturity)
        self._periods = self.schedule.periods()

    # -- cashflows ----------------------------------------------------------
    def _period_fraction(self, start: _dt.date, end: _dt.date, p: object) -> float:
        return self.day_count.year_fraction(
            start,
            end,
            ref_start=p.ref_start,
            ref_end=p.ref_end,
            frequency=self.frequency,
        )

    def cashflows(self) -> list[BondCashflow]:
        out = []
        for p in self._periods:
            pay = adjust(p.end, self.calendar, "F")
            amt = self.face * self.coupon * self._period_fraction(p.start, p.end, p)
            out.append(BondCashflow(p.start, p.end, pay, amt))
        last = self._periods[-1]
        out.append(
            BondCashflow(
                last.start, last.end, adjust(last.end, self.calendar, "F"), self.face, True
            )
        )
        return out

    def settlement_date(self, trade_date: DateLike) -> _dt.date:
        cal = self.calendar
        return cal.add_business_days(cal.next_business_day(trade_date), self.settlement_days)

    def _current(self, settle: _dt.date) -> int:
        for j, p in enumerate(self._periods):
            if p.start <= settle < p.end:
                return j
        raise ValueError(f"settlement {settle} outside the bond's life")

    def accrued_interest(self, settle: DateLike) -> float:
        s = to_date(settle)
        j = self._current(s)
        p = self._periods[j]
        return self.face * self.coupon * self._period_fraction(p.start, s, p)

    # -- yield ----------------------------------------------------------------
    def _exponents(self, s: _dt.date) -> tuple[int, float]:
        j = self._current(s)
        p = self._periods[j]
        # remaining part of the current period, in coupon periods
        return j, self._period_fraction(s, p.end, p) * self.frequency

    def _flows(self, s: _dt.date) -> list[tuple[float, float]]:
        """(exponent in periods, amount) for every cashflow after settlement."""
        j, w = self._exponents(s)
        cfs = self.cashflows()
        coupons = cfs[:-1]
        out = [(w + i - j, c.amount) for i, c in enumerate(coupons) if i >= j]
        out.append((w + len(coupons) - 1 - j, cfs[-1].amount))
        return out

    def dirty_price(self, y: float, settle: DateLike) -> float:
        f = self.frequency
        v = 1.0 / (1.0 + y / f)
        return sum(a * v**e for e, a in self._flows(to_date(settle)))

    def clean_price(self, y: float, settle: DateLike) -> float:
        return self.dirty_price(y, settle) - self.accrued_interest(settle)

    def yield_from_clean(
        self, clean: float, settle: DateLike, lo: float = -0.5, hi: float = 2.0
    ) -> float:
        target = clean + self.accrued_interest(settle)
        return float(
            brentq(
                lambda y: self.dirty_price(y, settle) - target,
                lo,
                hi,
                xtol=1e-15,
                rtol=1e-15,
                maxiter=200,
            )
        )

    def macaulay_duration(self, y: float, settle: DateLike) -> float:
        f = self.frequency
        v = 1.0 / (1.0 + y / f)
        flows = self._flows(to_date(settle))
        P = sum(a * v**e for e, a in flows)
        return sum(e / f * a * v**e for e, a in flows) / P

    def modified_duration(self, y: float, settle: DateLike) -> float:
        return self.macaulay_duration(y, settle) / (1.0 + y / self.frequency)

    def convexity(self, y: float, settle: DateLike) -> float:
        f = self.frequency
        v = 1.0 / (1.0 + y / f)
        flows = self._flows(to_date(settle))
        P = sum(a * v**e for e, a in flows)
        return sum(e * (e + 1) * a * v ** (e + 2) for e, a in flows) / (f * f * P)

    # -- curve pricing --------------------------------------------------------
    def dirty_price_from_curve(
        self, curve: YieldCurve, settle: DateLike, z_spread: float = 0.0
    ) -> float:
        """Forward dirty price at ``settle``: :math:`\\sum C_i P(t_i)e^{-z(t_i - t_s)}/P(t_s)`."""
        s = to_date(settle)
        ts = float(curve.time(s))
        ps = float(curve.discount(s))
        total = 0.0
        for c in self.cashflows():
            if c.payment_date > s and c.accrual_end > s:
                t = float(curve.time(c.payment_date))
                total += (
                    c.amount
                    * float(curve.discount(c.payment_date))
                    * math.exp(-z_spread * (t - ts))
                )
        return total / ps

    def z_spread(self, clean: float, curve: YieldCurve, settle: DateLike) -> float:
        target = clean + self.accrued_interest(settle)
        return float(
            brentq(
                lambda z: self.dirty_price_from_curve(curve, settle, z) - target,
                -0.5,
                2.0,
                xtol=1e-15,
                maxiter=200,
            )
        )
