r"""Interest-rate swaps.

A :class:`Swap` exchanges a fixed leg for a floating leg. For the payer of
fixed,

.. math:: \mathrm{PV} = \mathrm{PV}_{\text{float}} - \mathrm{PV}_{\text{fixed}}
          = (S - K)\,A, \qquad A = \sum_i N_i\tau_i P(t^{\text{pay}}_i),

where :math:`S = \mathrm{PV}_{\text{float}}/A` is the par rate. The par
spread is the margin over the floating index that makes the swap worth
zero, :math:`(\mathrm{PV}_{\text{fixed}} - \mathrm{PV}_{\text{float}})/A_{\text{float}}`.

:func:`ois_swap` builds overnight-indexed swaps (both legs on the same
schedule, compounded in arrears with a payment lag, as for SOFR and €STR
OIS); :func:`vanilla_swap` builds fixed-versus-IBOR swaps (EUR market
standard: annual 30/360 fixed against the index tenor).

A :class:`BasisSwap` exchanges two floating legs, e.g. EURIBOR 3M plus a
spread against EURIBOR 6M, or €STR against EURIBOR 3M (:func:`basis_swap`).

**Amortising and accreting swaps.** The ``notional`` of :func:`ois_swap`
and :func:`vanilla_swap` may be a number, a sequence with one entry per
period (when both legs share a schedule), or a function of the accrual
start date, such as :func:`step_notional` or :func:`linear_notional`.
Every coupon is linear in its notional, so a swap whose notional steps
down by :math:`\Delta N_j` at the dates :math:`T_j` is exactly the sum of
bullet swaps of notional :math:`\Delta N_j` maturing at :math:`T_j`, and its
par rate is the annuity-weighted average of their par rates.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace
from bisect import bisect_right
from typing import Callable, Optional, Sequence, Union

from spxa.market.calendars import get_calendar
from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, add_tenor, to_date
from spxa.market.schedule import Schedule, adjust
from spxa.rates.cashflows import (
    Cashflow,
    FixedCoupon,
    SimpleCashflow,
    fixed_leg,
    ibor_leg,
    overnight_leg,
)
from spxa.rates.indices import EURIBOR6M, SOFR, FixingHistory, IborIndex, OvernightIndex
from spxa.rates.pricing import leg_annuity, leg_pv

__all__ = [
    "BasisSwap",
    "NotionalLike",
    "Swap",
    "basis_swap",
    "linear_notional",
    "ois_swap",
    "step_notional",
    "vanilla_swap",
]

NotionalLike = Union[float, Sequence[float], Callable[[_dt.date], float]]


def step_notional(
    dates: Sequence[DateLike], values: Sequence[float]
) -> Callable[[_dt.date], float]:
    """Piecewise-constant notional: ``values[j]`` from ``dates[j]`` (inclusive) to ``dates[j+1]``.

    Dates before ``dates[0]`` raise; ``dates`` must be increasing.
    """
    ds = [to_date(d) for d in dates]
    vs = [float(v) for v in values]
    if len(ds) != len(vs) or not ds:
        raise ValueError("need one value per date")
    if any(b <= a for a, b in zip(ds, ds[1:])):
        raise ValueError("dates must be strictly increasing")

    def f(d: _dt.date) -> float:
        j = bisect_right(ds, to_date(d)) - 1
        if j < 0:
            raise ValueError(f"{d} is before the first notional date {ds[0]}")
        return vs[j]

    return f


def linear_notional(
    start: DateLike, end: DateLike, initial: float, final: float = 0.0
) -> Callable[[_dt.date], float]:
    """Notional moving linearly in calendar days from ``initial`` at ``start`` to ``final`` at ``end``.

    Evaluated at each accrual start; ``final > initial`` gives an accreting swap.
    """
    s0, e0 = to_date(start), to_date(end)
    span = (e0 - s0).days
    if span <= 0:
        raise ValueError("end must be after start")

    def f(d: _dt.date) -> float:
        x = (to_date(d) - s0).days / span
        return float(initial) + (float(final) - float(initial)) * min(max(x, 0.0), 1.0)

    return f


def _leg_notional(notional: NotionalLike, schedule: Schedule) -> "float | list[float]":
    if callable(notional):
        return [float(notional(p.start)) for p in schedule.periods()]
    if isinstance(notional, (int, float)):
        return float(notional)
    return [float(x) for x in notional]


@dataclass(frozen=True)
class Swap:
    """Fixed-for-floating swap.

    Parameters
    ----------
    fixed, floating : list of cashflows
        The two legs (amounts positive).
    payer : bool
        ``True`` if we pay the fixed leg.
    """

    fixed: Sequence[Cashflow]
    floating: Sequence[Cashflow]
    payer: bool = True
    _meta: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.fixed or not self.floating:
            raise ValueError("both legs must have cashflows")
        if not all(isinstance(c, FixedCoupon) for c in self.fixed):
            raise ValueError("the fixed leg must consist of FixedCoupon")

    @property
    def sign(self) -> float:
        return 1.0 if self.payer else -1.0

    @property
    def fixed_rate(self) -> float:
        rates = {c.rate for c in self.fixed if isinstance(c, FixedCoupon)}
        if len(rates) != 1:
            raise ValueError("fixed leg has several rates")
        return rates.pop()

    @property
    def start(self) -> _dt.date:
        return min(
            c.start
            for c in list(self.fixed) + list(self.floating)
            if not isinstance(c, SimpleCashflow)
        )

    @property
    def maturity(self) -> _dt.date:
        return max(c.payment_date for c in list(self.fixed) + list(self.floating))

    def fixed_leg_pv(
        self, discount: YieldCurve, valuation_date: Optional[DateLike] = None
    ) -> float:
        return leg_pv(self.fixed, discount, valuation_date=valuation_date)

    def floating_leg_pv(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        return leg_pv(self.floating, discount, projection, fixings, valuation_date)

    def annuity(self, discount: YieldCurve, valuation_date: Optional[DateLike] = None) -> float:
        """Fixed-leg annuity :math:`\\sum N\\tau P`."""
        return leg_annuity(self.fixed, discount, valuation_date)

    def pv(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        fl = self.floating_leg_pv(discount, projection, fixings, valuation_date)
        fx = self.fixed_leg_pv(discount, valuation_date)
        return self.sign * (fl - fx)

    def par_rate(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        """Fixed rate that makes the swap worth zero."""
        return self.floating_leg_pv(discount, projection, fixings, valuation_date) / self.annuity(
            discount, valuation_date
        )

    def with_fixed_rate(self, rate: float) -> "Swap":
        """Copy of the swap with every fixed coupon at ``rate``."""
        fixed = [replace(c, rate=rate) for c in self.fixed if isinstance(c, FixedCoupon)]
        return Swap(fixed, self.floating, self.payer, dict(self._meta))

    def pv01(self, discount: YieldCurve, valuation_date: Optional[DateLike] = None) -> float:
        """Value of one basis point on the fixed rate, :math:`10^{-4}A` (positive)."""
        return 1e-4 * self.annuity(discount, valuation_date)

    def dv01(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
        bump: float = 1e-4,
    ) -> float:
        """Change in PV for a one-basis-point parallel rise of all curves.

        Zero rates (continuously compounded) of the discount and projection
        curves are shifted together by ``bump`` up and down; the central
        difference is scaled to one basis point.
        """
        proj = projection if projection is not None else discount

        def pv_at(h: float) -> float:
            return self.pv(discount.shifted(h), proj.shifted(h), fixings, valuation_date)

        return (pv_at(bump) - pv_at(-bump)) / (2.0 * bump) * 1e-4

    def par_spread(
        self,
        discount: YieldCurve,
        projection: Optional[YieldCurve] = None,
        fixings: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        """Margin over the floating index (added to every coupon) for zero PV."""
        fl = self.floating_leg_pv(discount, projection, fixings, valuation_date)
        fx = self.fixed_leg_pv(discount, valuation_date)
        return (fx - fl) / leg_annuity(self.floating, discount, valuation_date)


def _dates(
    trade_date: DateLike, tenor: str, calendar: str, spot_lag: int, forward_start: Optional[str]
) -> tuple[_dt.date, _dt.date]:
    cal = get_calendar(calendar)
    start = cal.add_business_days(cal.next_business_day(trade_date), spot_lag)
    if forward_start is not None:
        start = adjust(add_tenor(start, forward_start), cal, "MF")
    return start, add_tenor(start, tenor)


def ois_swap(
    trade_date: DateLike,
    tenor: str,
    fixed_rate: float,
    index: OvernightIndex = SOFR,
    notional: NotionalLike = 1.0,
    payer: bool = True,
    frequency: str = "1Y",
    payment_lag: int = 2,
    spot_lag: int = 2,
    fixed_day_count: Optional[str] = None,
    forward_start: Optional[str] = None,
    spread: float = 0.0,
) -> Swap:
    """Overnight-indexed swap starting ``spot_lag`` business days after ``trade_date``.

    Both legs share an annual (by default) schedule on the index calendar,
    modified following, rolled from the unadjusted start; the floating leg
    compounds the index in arrears. Swaps shorter than ``frequency`` have a
    single period.
    """
    start, end = _dates(trade_date, tenor, index.calendar, spot_lag, forward_start)
    sch = Schedule(start, end, frequency, index.calendar, "MF", stub="short_back")
    dc = fixed_day_count or index.day_count
    n = _leg_notional(notional, sch)
    fx = fixed_leg(sch, n, fixed_rate, dc, payment_lag, index.calendar)
    fl = overnight_leg(sch, n, index, spread=spread, payment_lag=payment_lag)
    return Swap(fx, fl, payer, {"type": "OIS", "index": index.name, "tenor": tenor})


def vanilla_swap(
    trade_date: DateLike,
    tenor: str,
    fixed_rate: float,
    index: IborIndex = EURIBOR6M,
    notional: NotionalLike = 1.0,
    payer: bool = True,
    fixed_tenor: str = "1Y",
    fixed_day_count: str = "30/360",
    float_tenor: Optional[str] = None,
    spot_lag: Optional[int] = None,
    forward_start: Optional[str] = None,
    spread: float = 0.0,
    use_index_dates: bool = True,
) -> Swap:
    """Fixed versus term-rate swap starting at the index spot date.

    Both schedules are generated backward from the maturity on the index
    calendar (modified following, end-of-month); the floating coupons fix
    ``index.fixing_days`` before each period start and are projected over
    the index's own value and maturity dates (which differ from the coupon
    dates when a start date was rolled); ``use_index_dates=False`` projects
    over the coupon periods instead.
    """
    lag = index.fixing_days if spot_lag is None else spot_lag
    start, end = _dates(trade_date, tenor, index.calendar, lag, forward_start)
    eom = index.end_of_month
    fx_sch = Schedule(start, end, fixed_tenor, index.calendar, "MF", end_of_month=eom)
    fl_sch = Schedule(
        start, end, float_tenor or index.tenor, index.calendar, "MF", end_of_month=eom
    )
    if not callable(notional) and not isinstance(notional, (int, float)):
        if len(fx_sch.periods()) != len(fl_sch.periods()):
            raise ValueError(
                "legs have different schedules: pass the notional as a function of the "
                "accrual start date (e.g. step_notional)"
            )
    fx = fixed_leg(fx_sch, _leg_notional(notional, fx_sch), fixed_rate, fixed_day_count)
    fl = ibor_leg(
        fl_sch,
        _leg_notional(notional, fl_sch),
        index,
        spread=spread,
        use_index_dates=use_index_dates,
    )
    return Swap(fx, fl, payer, {"type": "IRS", "index": index.name, "tenor": tenor})


@dataclass(frozen=True)
class BasisSwap:
    r"""Floating-for-floating swap: pay ``first``, receive ``second``.

    The spread is quoted on the first leg; the par spread is
    :math:`(\mathrm{PV}_2 - \mathrm{PV}_1)/A_1` with :math:`A_1` the
    first leg's annuity. Each leg is projected from its own curve.
    """

    first: Sequence[Cashflow]
    second: Sequence[Cashflow]

    def __post_init__(self) -> None:
        if not self.first or not self.second:
            raise ValueError("both legs must have cashflows")

    def leg_pvs(
        self,
        discount: YieldCurve,
        projection1: Optional[YieldCurve] = None,
        projection2: Optional[YieldCurve] = None,
        fixings1: Optional[FixingHistory] = None,
        fixings2: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> tuple[float, float]:
        return (
            leg_pv(self.first, discount, projection1, fixings1, valuation_date),
            leg_pv(self.second, discount, projection2, fixings2, valuation_date),
        )

    def pv(
        self,
        discount: YieldCurve,
        projection1: Optional[YieldCurve] = None,
        projection2: Optional[YieldCurve] = None,
        fixings1: Optional[FixingHistory] = None,
        fixings2: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        p1, p2 = self.leg_pvs(
            discount, projection1, projection2, fixings1, fixings2, valuation_date
        )
        return p2 - p1

    def par_spread(
        self,
        discount: YieldCurve,
        projection1: Optional[YieldCurve] = None,
        projection2: Optional[YieldCurve] = None,
        fixings1: Optional[FixingHistory] = None,
        fixings2: Optional[FixingHistory] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> float:
        """Spread to add to the first leg for zero PV (on top of any existing spread)."""
        p1, p2 = self.leg_pvs(
            discount, projection1, projection2, fixings1, fixings2, valuation_date
        )
        return (p2 - p1) / leg_annuity(self.first, discount, valuation_date)

    @property
    def maturity(self) -> _dt.date:
        return max(c.payment_date for c in list(self.first) + list(self.second))


def _float_leg(
    sch: Schedule,
    notional: float,
    index: "IborIndex | OvernightIndex",
    spread: float,
    payment_lag: int,
    use_index_dates: bool,
) -> list[Cashflow]:
    if isinstance(index, IborIndex):
        return ibor_leg(
            sch,
            notional,
            index,
            spread=spread,
            payment_lag=payment_lag,
            use_index_dates=use_index_dates,
        )
    return overnight_leg(sch, notional, index, spread=spread, payment_lag=payment_lag)


def basis_swap(
    trade_date: DateLike,
    tenor: str,
    first: "IborIndex | OvernightIndex",
    second: "IborIndex | OvernightIndex",
    spread: float = 0.0,
    notional: float = 1.0,
    first_tenor: Optional[str] = None,
    second_tenor: Optional[str] = None,
    spot_lag: int = 2,
    payment_lag: int = 0,
    forward_start: Optional[str] = None,
    use_index_dates: bool = True,
) -> BasisSwap:
    """Basis swap paying ``first`` + ``spread`` against ``second``.

    IBOR legs pay at their index tenor and overnight legs quarterly unless
    ``first_tenor``/``second_tenor`` say otherwise; dates follow the
    calendar of the first index.
    """
    cal = first.calendar
    start, end = _dates(trade_date, tenor, cal, spot_lag, forward_start)

    def leg(idx: "IborIndex | OvernightIndex", t: Optional[str], s: float) -> list[Cashflow]:
        per = t or (idx.tenor if isinstance(idx, IborIndex) else "3M")
        sch = Schedule(start, end, per, idx.calendar, "MF", end_of_month=True)
        return _float_leg(sch, notional, idx, s, payment_lag, use_index_dates)

    return BasisSwap(leg(first, first_tenor, spread), leg(second, second_tenor, 0.0))
