r"""Zero-coupon and year-on-year inflation swaps on dated schedules.

A zero-coupon swap exchanges :math:`N\,[I(M)/I(S) - 1]` for
:math:`N\,[(1+K)^n - 1]` at maturity. Its value needs only the inflation curve.
A year-on-year swap pays :math:`I(T_i)/I(T_{i-1}) - 1` each period against
:math:`K\tau_i`. The ratio of two future fixings is not a forward, so its
expectation under the payment measure carries the convexity
:math:`e^{C(T_{i-1},T_i)}` of a model. :class:`JarrowYildirim` provides it. Without a
model the ratio of forward fixings is used, which is exact only when
real rates are deterministic.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any, Optional

from spxa.market.daycount import DayCount
from spxa.market.dates import add_months, to_date
from spxa.rates.inflation import InflationCurve
from spxa.rates.jarrow_yildirim import JarrowYildirim

__all__ = ["YearOnYearInflationSwap", "ZeroCouponInflationSwap"]


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """Receive inflation, pay fixed (``notional`` positive), ``years`` whole years from ``start``."""

    start: _dt.date
    years: int
    fixed_rate: float
    notional: float = 1.0

    @property
    def maturity(self) -> _dt.date:
        return add_months(to_date(self.start), 12 * self.years)

    def inflation_leg(self, nominal: Any, curve: InflationCurve) -> float:
        ratio = curve.reference_cpi(self.maturity) / curve.reference_cpi(self.start)
        return self.notional * float(nominal.discount(self.maturity)) * (ratio - 1.0)

    def fixed_leg(self, nominal: Any) -> float:
        return (
            self.notional
            * float(nominal.discount(self.maturity))
            * ((1.0 + self.fixed_rate) ** self.years - 1.0)
        )

    def present_value(self, nominal: Any, curve: InflationCurve) -> float:
        return self.inflation_leg(nominal, curve) - self.fixed_leg(nominal)

    def fair_rate(self, curve: InflationCurve) -> float:
        return curve.zero_coupon_rate(self.start, self.maturity, self.years)


@dataclass(frozen=True)
class YearOnYearInflationSwap:
    """Receive :math:`I(T_i)/I(T_{i-1}) - 1`, pay ``fixed_rate`` times the accrual (30/360, so whole years on anniversary dates), every ``months`` months."""

    start: _dt.date
    years: int
    fixed_rate: float
    notional: float = 1.0
    months: int = 12
    day_count: str = "30/360"

    def schedule(self) -> list[_dt.date]:
        s = to_date(self.start)
        return [add_months(s, self.months * k) for k in range(12 * self.years // self.months + 1)]

    def _times(self, nominal: Any, dates: list[_dt.date]) -> list[float]:
        return [float(nominal.time(d)) for d in dates]

    def inflation_leg(
        self, nominal: Any, curve: InflationCurve, model: Optional[JarrowYildirim] = None
    ) -> float:
        dates = self.schedule()
        times = self._times(nominal, dates)
        total = 0.0
        for d0, d1, t0, t1 in zip(dates, dates[1:], times, times[1:]):
            ratio = curve.reference_cpi(d1) / curve.reference_cpi(d0)
            if model is not None and t0 > 0:
                ratio *= math.exp(model.yoy_convexity(t0, t1))
            total += float(nominal.discount(d1)) * (ratio - 1.0)
        return self.notional * total

    def annuity(self, nominal: Any) -> float:
        dc, dates = DayCount(self.day_count), self.schedule()
        return self.notional * sum(
            dc.year_fraction(a, b) * float(nominal.discount(b)) for a, b in zip(dates, dates[1:])
        )

    def present_value(
        self, nominal: Any, curve: InflationCurve, model: Optional[JarrowYildirim] = None
    ) -> float:
        return self.inflation_leg(nominal, curve, model) - self.fixed_rate * self.annuity(nominal)

    def fair_rate(
        self, nominal: Any, curve: InflationCurve, model: Optional[JarrowYildirim] = None
    ) -> float:
        return self.inflation_leg(nominal, curve, model) / self.annuity(nominal)
