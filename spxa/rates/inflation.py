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
from typing import Iterable, Optional

from spxa.market.dates import DateLike, add_months, days_in_month, to_date

__all__ = ["CPIFixings", "InflationIndex", "month_start"]


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
