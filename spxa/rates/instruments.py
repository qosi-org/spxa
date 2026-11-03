r"""Money-market instruments: deposits and forward rate agreements.

:class:`Deposit`
    Lend :math:`N` at :math:`t_s`, receive :math:`N(1 + r\tau)` at
    :math:`t_e`: :math:`\mathrm{PV} = N\bigl[(1 + r\tau)P(t_e) - P(t_s)\bigr]`,
    so the par rate is :math:`(P(t_s)/P(t_e) - 1)/\tau`.
:class:`FRA`
    Cash-settled at the start of the period with the market discounting
    convention: the payer receives
    :math:`N\tau(L - K)/(1 + \tau L)` at :math:`t_s`, where :math:`L` is
    the term rate fixed for :math:`[t_s, t_e]`. Valued with the projected
    forward :math:`F` (or the published fixing),

    .. math:: \mathrm{PV} = N\frac{\tau(F - K)}{1 + \tau F}P_{\text{disc}}(t_s).

    With one curve this equals :math:`N[P(t_s) - (1 + K\tau)P(t_e)]`.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional

from spxa.market.calendars import get_calendar
from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, add_tenor
from spxa.market.daycount import DayCount
from spxa.market.schedule import CalendarLike, adjust
from spxa.rates.indices import FixingHistory, IborIndex

__all__ = ["FRA", "Deposit"]


def _spot_and_end(
    trade_date: DateLike,
    tenor: str,
    calendar: CalendarLike,
    spot_lag: int,
    convention: str,
    end_of_month: bool,
) -> tuple[_dt.date, _dt.date]:
    cal = get_calendar(calendar)
    spot = cal.add_business_days(cal.next_business_day(trade_date), spot_lag)
    eom = end_of_month and spot == cal.previous_business_day(_month_end(spot))
    raw = add_tenor(_month_end(spot) if eom else spot, tenor, end_of_month_rule=eom)
    return spot, adjust(raw, cal, convention)


def _month_end(d: _dt.date) -> _dt.date:
    nxt = _dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - _dt.timedelta(days=1)


@dataclass(frozen=True)
class Deposit:
    """Unsecured term deposit (the lender's view)."""

    start: _dt.date
    end: _dt.date
    rate: float
    day_count: str = "ACT/360"
    notional: float = 1.0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")

    @classmethod
    def from_tenor(
        cls,
        trade_date: DateLike,
        tenor: str,
        rate: float,
        calendar: CalendarLike = "TARGET",
        spot_lag: int = 2,
        convention: str = "MF",
        end_of_month: bool = True,
        day_count: str = "ACT/360",
        notional: float = 1.0,
    ) -> "Deposit":
        """Deposit starting ``spot_lag`` business days after ``trade_date``."""
        start, end = _spot_and_end(trade_date, tenor, calendar, spot_lag, convention, end_of_month)
        return cls(start, end, rate, day_count, notional)

    @property
    def accrual(self) -> float:
        return DayCount(self.day_count).year_fraction(self.start, self.end)

    @property
    def maturity(self) -> _dt.date:
        return self.end

    def implied_rate(self, curve: YieldCurve) -> float:
        """Par rate :math:`(P(t_s)/P(t_e) - 1)/\\tau`."""
        return (
            float(curve.discount(self.start)) / float(curve.discount(self.end)) - 1.0
        ) / self.accrual

    def pv(self, curve: YieldCurve) -> float:
        p0 = float(curve.discount(self.start))
        p1 = float(curve.discount(self.end))
        return self.notional * ((1.0 + self.rate * self.accrual) * p1 - p0)


@dataclass(frozen=True)
class FRA:
    """Forward rate agreement on a term rate.

    Parameters
    ----------
    start, end : date
        Accrual period (the settlement date is ``start``).
    strike : float
        Contract rate :math:`K`.
    index : IborIndex, optional
        Index whose fixing settles the contract; supplies the day count.
    day_count : str
        Used when no index is given.
    payer : bool
        ``True`` to pay the fixed rate (gain when rates rise).
    """

    start: _dt.date
    end: _dt.date
    strike: float
    index: Optional[IborIndex] = None
    day_count: str = "ACT/360"
    notional: float = 1.0
    payer: bool = True

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")

    @classmethod
    def from_index(
        cls,
        index: IborIndex,
        trade_date: DateLike,
        start_tenor: str,
        strike: float,
        notional: float = 1.0,
        payer: bool = True,
    ) -> "FRA":
        """An ``m x (m + tenor)`` FRA: starts ``start_tenor`` after spot, ends at the index maturity."""
        spot = index.value_date(trade_date)
        start = adjust(add_tenor(spot, start_tenor), index.cal, index.convention)
        return cls(
            start, index.maturity_date(start), strike, index, index.day_count, notional, payer
        )

    @property
    def accrual(self) -> float:
        dc = self.index.dc if self.index is not None else DayCount(self.day_count)
        return dc.year_fraction(self.start, self.end)

    @property
    def maturity(self) -> _dt.date:
        return self.end

    @property
    def fixing_date(self) -> Optional[_dt.date]:
        return self.index.fixing_date(self.start) if self.index is not None else None

    def forward_rate(
        self, projection: YieldCurve, fixings: Optional[FixingHistory] = None
    ) -> float:
        """Published fixing if available, else :math:`(P(t_s)/P(t_e) - 1)/\\tau`."""
        fd = self.fixing_date
        if fd is not None and fixings is not None:
            f = fixings.get(fd)
            if f is not None:
                return float(f)
        ref = projection.reference_date
        if fd is not None and ref is not None and fd < ref:
            raise KeyError(f"missing past {self.index.name if self.index else ''} fixing for {fd}")
        p0 = float(projection.discount(self.start))
        p1 = float(projection.discount(self.end))
        return (p0 / p1 - 1.0) / self.accrual

    implied_rate = forward_rate

    def settlement_amount(self, fixing: float) -> float:
        """Amount paid at ``start`` to the payer given the fixing."""
        tau = self.accrual
        sign = 1.0 if self.payer else -1.0
        return sign * self.notional * tau * (fixing - self.strike) / (1.0 + tau * fixing)

    def pv(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
    ) -> float:
        ref = discount.reference_date
        if ref is not None and self.start <= ref:
            return 0.0  # already settled
        F = self.forward_rate(projection if projection is not None else discount, fixings)
        return self.settlement_amount(F) * float(discount.discount(self.start))
