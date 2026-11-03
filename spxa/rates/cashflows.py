r"""Coupons and legs.

A leg is a list of cashflows, each with a payment date and an amount
that may depend on a projection curve and on published fixings:

:class:`FixedCoupon`
    :math:`N \cdot c \cdot \tau(t_s, t_e)`.
:class:`IborCoupon`
    :math:`N\,(g\,L + s)\,\tau(t_s, t_e)` with the term rate :math:`L`
    fixed ``fixing_days`` before the period start. Projected from a curve,
    :math:`L = (P(v)/P(v + \text{tenor}) - 1)/\tau_{\text{idx}}` over the
    index's own value and maturity dates.
:class:`OvernightCoupon`
    :math:`N\,R\,\tau` with :math:`R` the compounded (or averaged)
    overnight rate of :mod:`spxa.rates.overnight`.
:class:`SimpleCashflow`
    A fixed amount (notional exchanges, fees).

``fixed_leg``, ``ibor_leg`` and ``overnight_leg`` build legs from a
:class:`~spxa.market.Schedule`, with optional amortising notionals and a
payment lag.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from spxa.market.calendars import Calendar, get_calendar
from spxa.market.curves import YieldCurve
from spxa.market.daycount import DayCount
from spxa.market.schedule import Schedule
from spxa.rates.indices import FixingHistory, IborIndex, OvernightIndex
from spxa.rates.overnight import (
    compounded_overnight_rate,
    curve_projection,
    projected_overnight_rate,
)

__all__ = [
    "Cashflow",
    "FixedCoupon",
    "IborCoupon",
    "Leg",
    "OvernightCoupon",
    "SimpleCashflow",
    "fixed_leg",
    "ibor_leg",
    "overnight_leg",
]


@dataclass(frozen=True)
class SimpleCashflow:
    """A known amount paid on ``payment_date``."""

    payment_date: _dt.date
    value: float

    def amount(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        return self.value


@dataclass(frozen=True)
class FixedCoupon:
    """Fixed-rate coupon."""

    payment_date: _dt.date
    start: _dt.date
    end: _dt.date
    notional: float
    rate: float
    day_count: str = "30/360"
    ref_start: Optional[_dt.date] = None
    ref_end: Optional[_dt.date] = None
    frequency: Optional[int] = None

    @property
    def accrual(self) -> float:
        dc = DayCount(self.day_count)
        if dc.name == "ACT/ACT ICMA":
            return dc.year_fraction(
                self.start,
                self.end,
                ref_start=self.ref_start or self.start,
                ref_end=self.ref_end or self.end,
                frequency=self.frequency,
            )
        return dc.year_fraction(self.start, self.end)

    def amount(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        return self.notional * self.rate * self.accrual


@dataclass(frozen=True)
class IborCoupon:
    """Floating coupon on a term-rate index.

    ``fixing_date`` defaults to the index fixing date of ``start``.
    With ``use_index_dates=False`` the projected forward runs over the
    coupon dates instead of the index value and maturity dates.
    """

    payment_date: _dt.date
    start: _dt.date
    end: _dt.date
    notional: float
    index: IborIndex
    spread: float = 0.0
    gearing: float = 1.0
    day_count: Optional[str] = None
    fixing_date: Optional[_dt.date] = None
    use_index_dates: bool = True

    @property
    def accrual(self) -> float:
        return DayCount(self.day_count or self.index.day_count).year_fraction(self.start, self.end)

    @property
    def fixing(self) -> _dt.date:
        return (
            self.fixing_date if self.fixing_date is not None else self.index.fixing_date(self.start)
        )

    def index_rate(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        """Published fixing if available, otherwise the projected forward."""
        if fixings is not None:
            v = fixings.get(self.fixing)
            if v is not None:
                return v
        if projection is None:
            raise KeyError(f"no {self.index.name} fixing for {self.fixing} and no projection curve")
        ref = projection.reference_date
        if ref is not None and self.fixing < ref:
            raise KeyError(f"missing past {self.index.name} fixing for {self.fixing}")
        if self.use_index_dates:
            v0 = self.index.value_date(self.fixing)
            v1 = self.index.maturity_date(v0)
        else:
            v0, v1 = self.start, self.end
        tau = self.index.dc.year_fraction(v0, v1)
        return (float(projection.discount(v0)) / float(projection.discount(v1)) - 1.0) / tau

    def rate(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        return self.gearing * self.index_rate(projection, fixings) + self.spread

    def amount(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        return self.notional * self.rate(projection, fixings) * self.accrual


@dataclass(frozen=True)
class OvernightCoupon:
    """Coupon paying a compounded (or averaged) overnight rate."""

    payment_date: _dt.date
    start: _dt.date
    end: _dt.date
    notional: float
    index: OvernightIndex
    spread: float = 0.0
    lookback: int = 0
    lockout: int = 0
    observation_shift: bool = False
    spread_compounding: str = "exclusive"
    method: str = "compounded"
    day_count: Optional[str] = None

    @property
    def accrual(self) -> float:
        return DayCount(self.day_count or self.index.day_count).year_fraction(self.start, self.end)

    def rate(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        if self.method == "compounded" and projection is not None:
            return projected_overnight_rate(
                self.index,
                self.start,
                self.end,
                projection,
                fixings,
                lookback=self.lookback,
                lockout=self.lockout,
                observation_shift=self.observation_shift,
                spread=self.spread,
                spread_compounding=self.spread_compounding,
            )
        return compounded_overnight_rate(
            self.index,
            self.start,
            self.end,
            fixings,
            curve_projection(self.index, projection) if projection is not None else None,
            lookback=self.lookback,
            lockout=self.lockout,
            observation_shift=self.observation_shift,
            spread=self.spread,
            spread_compounding=self.spread_compounding,
            method=self.method,
        ).rate

    def amount(
        self, projection: Optional[YieldCurve] = None, fixings: Optional[FixingHistory] = None
    ) -> float:
        return self.notional * self.rate(projection, fixings) * self.accrual


Cashflow = Union[SimpleCashflow, FixedCoupon, IborCoupon, OvernightCoupon]
Leg = list  # list of Cashflow


def _notionals(notional: Union[float, Sequence[float]], n: int) -> list[float]:
    if isinstance(notional, (int, float)):
        return [float(notional)] * n
    out = [float(x) for x in notional]
    if len(out) != n:
        raise ValueError(f"{len(out)} notionals for {n} periods")
    return out


def _pay_date(end: _dt.date, lag: int, calendar: "str | Calendar") -> _dt.date:
    if lag == 0:
        return end
    return get_calendar(calendar).add_business_days(end, lag)


def fixed_leg(
    schedule: Schedule,
    notional: Union[float, Sequence[float]],
    rate: float,
    day_count: str = "30/360",
    payment_lag: int = 0,
    payment_calendar: Optional[str] = None,
) -> list[Cashflow]:
    """Fixed coupons over the schedule's accrual periods."""
    periods = schedule.periods()
    ns = _notionals(notional, len(periods))
    freq = schedule.frequency
    cal: "str | Calendar" = payment_calendar or schedule.calendar
    out: list[Cashflow] = []
    for p, n in zip(periods, ns):
        out.append(
            FixedCoupon(
                _pay_date(p.end, payment_lag, cal),
                p.start,
                p.end,
                n,
                rate,
                day_count,
                p.ref_start,
                p.ref_end,
                freq,
            )
        )
    return out


def ibor_leg(
    schedule: Schedule,
    notional: Union[float, Sequence[float]],
    index: IborIndex,
    spread: float = 0.0,
    gearing: float = 1.0,
    day_count: Optional[str] = None,
    payment_lag: int = 0,
    use_index_dates: bool = True,
) -> list[Cashflow]:
    """Term-rate coupons over the schedule's accrual periods."""
    periods = schedule.periods()
    ns = _notionals(notional, len(periods))
    return [
        IborCoupon(
            _pay_date(p.end, payment_lag, index.calendar),
            p.start,
            p.end,
            n,
            index,
            spread,
            gearing,
            day_count,
            None,
            use_index_dates,
        )
        for p, n in zip(periods, ns)
    ]


def overnight_leg(
    schedule: Schedule,
    notional: Union[float, Sequence[float]],
    index: OvernightIndex,
    spread: float = 0.0,
    payment_lag: int = 0,
    lookback: int = 0,
    lockout: int = 0,
    observation_shift: bool = False,
    spread_compounding: str = "exclusive",
    method: str = "compounded",
    day_count: Optional[str] = None,
) -> list[Cashflow]:
    """Overnight coupons; SOFR and SONIA swaps typically use ``payment_lag=2``."""
    periods = schedule.periods()
    ns = _notionals(notional, len(periods))
    return [
        OvernightCoupon(
            _pay_date(p.end, payment_lag, index.calendar),
            p.start,
            p.end,
            n,
            index,
            spread,
            lookback,
            lockout,
            observation_shift,
            spread_compounding,
            method,
            day_count,
        )
        for p, n in zip(periods, ns)
    ]
