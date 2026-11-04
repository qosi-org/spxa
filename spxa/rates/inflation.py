r"""Consumer price indices: monthly fixings, publication lag and reference CPI.

A price index is published once a month. Inflation-linked instruments use it
with a lag of :math:`L` months (three for US TIPS, UK linkers since 2005, and
euro-area HICP swaps). With interpolation, the reference CPI on day :math:`t` of
a month with :math:`D` days is

.. math::

    \mathrm{RefCPI}(d) = \mathrm{CPI}_{m-L}
        + \frac{t - 1}{D}\bigl(\mathrm{CPI}_{m-L+1} - \mathrm{CPI}_{m-L}\bigr),

so the first of each month carries the CPI of :math:`L` months earlier
(US Treasury, Canadian RRBs, HICP inflation-linked bonds). Without
interpolation (zero-coupon inflation swaps on HICP ex-tobacco, UK RPI swaps)
the whole month uses :math:`\mathrm{CPI}_{m-L}`. The index ratio of a bond is
its reference CPI over the reference CPI on its dated date.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from spxa.market.curves import InterpolatedDiscountCurve
from spxa.market.dates import DateLike, add_months, days_in_month, to_date

__all__ = [
    "CPIFixings",
    "InflationCurve",
    "InflationIndex",
    "Seasonality",
    "bootstrap_inflation_curve",
    "month_start",
]


def month_start(d: DateLike) -> _dt.date:
    """First day of the month of ``d``."""
    x = to_date(d)
    return x.replace(day=1)


@dataclass
class CPIFixings:
    """Published monthly index levels, keyed by the first day of the reference month."""

    name: str
    levels: dict[_dt.date, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data, self.levels = self.levels, {}
        for d, x in data.items():
            self.add(d, x)

    @classmethod
    def from_pairs(cls, name: str, pairs: Iterable[tuple[DateLike, float]]) -> "CPIFixings":
        out = cls(name)
        for d, x in pairs:
            out.add(d, x)
        return out

    @classmethod
    def from_series(cls, name: str, first_month: DateLike, values: Iterable[float]) -> "CPIFixings":
        """Consecutive monthly levels starting at ``first_month``."""
        out, m = cls(name), month_start(first_month)
        for x in values:
            out.add(m, x)
            m = add_months(m, 1)
        return out

    def add(self, month: DateLike, level: float) -> None:
        x = float(level)
        if not (math.isfinite(x) and x > 0):
            raise ValueError("index levels must be positive and finite")
        self.levels[month_start(month)] = x

    def get(self, month: DateLike) -> Optional[float]:
        return self.levels.get(month_start(month))

    def __getitem__(self, month: DateLike) -> float:
        x = self.get(month)
        if x is None:
            raise KeyError(f"no {self.name} fixing for {month_start(month):%Y-%m}")
        return x

    def __contains__(self, month: object) -> bool:
        try:
            return month_start(month) in self.levels  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    @property
    def last_month(self) -> _dt.date:
        """The latest published reference month."""
        if not self.levels:
            raise ValueError("no fixings")
        return max(self.levels)


@dataclass(frozen=True)
class InflationIndex:
    """A price index as used by inflation-linked instruments.

    ``lag`` is the observation lag in months; ``interpolated`` selects daily
    linear interpolation between consecutive lagged months; ``digits``
    rounds reference CPIs and index ratios (five for US TIPS and Canadian
    RRBs; ``None`` keeps full precision).
    """

    name: str
    fixings: CPIFixings
    lag: int = 3
    interpolated: bool = True
    digits: Optional[int] = None

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("the lag must be non-negative")

    def _round(self, x: float) -> float:
        return x if self.digits is None else round(x, self.digits)

    def reference_months(self, d: DateLike) -> tuple[_dt.date, _dt.date, float]:
        """The two lagged months used on ``d`` and the weight on the second."""
        x = to_date(d)
        m0 = add_months(month_start(x), -self.lag)
        if not self.interpolated:
            return m0, m0, 0.0
        return m0, add_months(m0, 1), (x.day - 1) / days_in_month(x.year, x.month)

    def reference_cpi(self, d: DateLike) -> float:
        """Reference CPI on date ``d`` from the published fixings."""
        m0, m1, w = self.reference_months(d)
        lo = self.fixings[m0]
        val = lo if w == 0.0 else lo + w * (self.fixings[m1] - lo)
        return self._round(val)

    def index_ratio(self, d: DateLike, base_date: DateLike) -> float:
        """Reference CPI on ``d`` over the reference CPI on ``base_date``."""
        return self._round(self.reference_cpi(d) / self.reference_cpi(base_date))

    def is_fixed(self, d: DateLike) -> bool:
        """Whether every fixing needed on ``d`` has been published."""
        m0, m1, w = self.reference_months(d)
        return m0 in self.fixings and (w == 0.0 or m1 in self.fixings)


def _months_between(a: _dt.date, b: _dt.date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


@dataclass(frozen=True)
class Seasonality:
    """Multiplicative monthly seasonality of a price index.

    ``factors[k]`` is the seasonal log change from month ``k`` (January = 0)
    to the next. The factors are demeaned so that they sum to zero over a year:
    seasonality then moves intra-year fixings but leaves fixings a whole number
    of years from the base month untouched.
    """

    factors: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.factors) != 12:
            raise ValueError("need twelve monthly factors")
        mean = sum(self.factors) / 12.0
        object.__setattr__(self, "factors", tuple(float(f) - mean for f in self.factors))

    def adjustment(self, month: DateLike, base_month: DateLike) -> float:
        """Seasonal factor of ``month`` relative to ``base_month``."""
        a, b = month_start(base_month), month_start(month)
        n = _months_between(a, b)
        sign, lo = (1.0, a) if n >= 0 else (-1.0, b)
        total = sum(self.factors[(lo.month - 1 + k) % 12] for k in range(abs(n)))
        return math.exp(sign * total)


class InflationCurve:
    r"""Forward fixings of a price index.

    The curve holds index levels by reference (fixing) month: published
    fixings up to ``base_month`` and bootstrapped pillar levels beyond. Between
    pillars, :math:`\log I` is linear in the month count, multiplied by a
    :class:`Seasonality` adjustment relative to ``base_month``.
    :meth:`reference_cpi` then applies the index's lag and interpolation to the
    forward fixings, exactly as to published ones.
    """

    def __init__(
        self,
        index: InflationIndex,
        base_month: DateLike,
        pillar_months: Iterable[DateLike],
        pillar_levels: Iterable[float],
        seasonality: Optional[Seasonality] = None,
    ) -> None:
        self.index, self.seasonality = index, seasonality
        self.base_month = month_start(base_month)
        self.base_level = index.fixings[self.base_month]
        months = [month_start(m) for m in pillar_months]
        levels = [float(x) for x in pillar_levels]
        if len(months) != len(levels) or any(x <= 0 for x in levels):
            raise ValueError("need one positive level per pillar month")
        n = [_months_between(self.base_month, m) for m in months]
        if any(k <= 0 for k in n) or any(b <= a for a, b in zip(n, n[1:])):
            raise ValueError("pillar months must increase after the base month")
        self.pillar_months = tuple(months)
        self._n = [0] + n
        self._log = [math.log(self.base_level)] + [math.log(x) for x in levels]

    def _trend(self, n: int) -> float:
        ns, ys = self._n, self._log
        if n >= ns[-1]:  # flat inflation beyond the last pillar
            slope = (ys[-1] - ys[-2]) / (ns[-1] - ns[-2])
            return math.exp(ys[-1] + slope * (n - ns[-1]))
        for k in range(1, len(ns)):
            if n <= ns[k]:
                w = (n - ns[k - 1]) / (ns[k] - ns[k - 1])
                return math.exp(ys[k - 1] + w * (ys[k] - ys[k - 1]))
        raise AssertionError("unreachable")

    def fixing(self, month: DateLike) -> float:
        """Published fixing if available, otherwise the forward fixing."""
        m = month_start(month)
        if m <= self.base_month:
            return self.index.fixings[m]
        level = self._trend(_months_between(self.base_month, m))
        if self.seasonality is not None:
            level *= self.seasonality.adjustment(m, self.base_month)
        return level

    def reference_cpi(self, d: DateLike) -> float:
        """Forward reference CPI on ``d`` (lag and interpolation of the index; no rounding)."""
        m0, m1, w = self.index.reference_months(d)
        lo = self.fixing(m0)
        return lo if w == 0.0 else lo + w * (self.fixing(m1) - lo)

    def zero_coupon_rate(
        self, start: DateLike, maturity: DateLike, years: Optional[float] = None
    ) -> float:
        """Fair ZCIS rate :math:`(I(M)/I(S))^{1/n} - 1`; ``years`` defaults to whole years between the dates."""
        s, m = to_date(start), to_date(maturity)
        n = years if years is not None else _months_between(s, m) / 12.0
        return (self.reference_cpi(m) / self.reference_cpi(s)) ** (1.0 / n) - 1.0

    def real_curve(
        self, nominal: Any, today: DateLike, years: int = 30
    ) -> InterpolatedDiscountCurve:
        r"""Real discount curve :math:`P_r(0,T) = P_n(0,T)\,\hat I(T)/I(0)`, where :math:`\hat I` is
        the forward reference CPI: the value today, in today's money, of one unit of
        index-linked notional paid at :math:`T`. Nodes sit on the first of every month."""
        t0 = to_date(today)
        i0 = self.reference_cpi(t0)
        dates = [add_months(month_start(t0), k) for k in range(1, 12 * years + 1)]
        dfs = [float(nominal.discount(d)) * self.reference_cpi(d) / i0 for d in dates]
        return InterpolatedDiscountCurve(dates, dfs, reference_date=t0, day_count=nominal.day_count)


def bootstrap_inflation_curve(
    index: InflationIndex,
    start: DateLike,
    quotes: dict[int, float],
    seasonality: Optional[Seasonality] = None,
) -> InflationCurve:
    """Forward fixings from zero-coupon inflation swap rates.

    ``quotes`` maps whole-year maturities to fair rates of swaps starting on
    ``start`` and paying :math:`I(M)/I(S) - 1` against :math:`(1+K)^n - 1`. The
    index must be non-interpolated (the market convention for HICP and UK RPI
    swaps), so each quote fixes one month, :math:`n` years after the base month.
    With interpolation, use whole-month starts. Seasonality is then exact at the
    pillars because it nets to zero over a year.
    """
    s = to_date(start)
    m0, m1, w = index.reference_months(s)
    if w != 0.0:
        raise ValueError("bootstrap from the first of a month or a non-interpolated index")
    base = index.fixings[m0]
    years = sorted(quotes)
    months = [add_months(m0, 12 * n) for n in years]
    levels = [base * (1.0 + quotes[n]) ** n for n in years]
    return InflationCurve(index, m0, months, levels, seasonality)
