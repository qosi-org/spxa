r"""Interest-rate indices and fixing histories.

An index is a set of conventions: the day count of its accrual, the
calendar on which it fixes, and for term rates the tenor, spot lag and
business-day convention. Fixings themselves live in a :class:`FixingHistory`
so that the same index can be used with different data sets.

Overnight indices (risk-free rates)
    The fixing for business day :math:`d_i` applies to the overnight
    period :math:`[d_i, d_{i+1})`, with :math:`d_{i+1}` the next business
    day; it accrues :math:`\tau_i = \mathrm{dc}(d_i, d_{i+1})`.
    Presets: SOFR (ACT/360, US SIFMA calendar), €STR (ACT/360, TARGET),
    SONIA (ACT/365F, UK).
Term (IBOR-style) indices
    Fixed ``fixing_days`` business days before the value date; the
    accrual runs from the value date to the value date plus the tenor,
    adjusted with the index convention. Presets: EURIBOR 1M/3M/6M/12M
    (ACT/360, TARGET, T-2, modified following, end-of-month).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Optional, Union

from spxa.market.calendars import Calendar, get_calendar
from spxa.market.dates import DateLike, Tenor, add_tenor, to_date
from spxa.market.daycount import DayCount
from spxa.market.schedule import adjust

__all__ = [
    "ESTR",
    "EURIBOR1M",
    "EURIBOR3M",
    "EURIBOR6M",
    "EURIBOR12M",
    "SOFR",
    "SONIA",
    "FixingHistory",
    "IborIndex",
    "OvernightIndex",
    "available_indices",
    "get_index",
]


@dataclass(frozen=True)
class OvernightIndex:
    """Overnight risk-free-rate index.

    Parameters
    ----------
    name, currency : str
    day_count : str
        Accrual convention of the overnight rate.
    calendar : str
        Fixing calendar.
    publication_lag : int
        Business days after the fixing date on which the rate is published
        (informational; SOFR, €STR and SONIA publish the next business day).
    """

    name: str
    currency: str
    day_count: str = "ACT/360"
    calendar: str = "WEEKENDS"
    publication_lag: int = 1

    @property
    def cal(self) -> Calendar:
        return get_calendar(self.calendar)

    @property
    def dc(self) -> DayCount:
        return DayCount(self.day_count)

    def is_fixing_date(self, d: DateLike) -> bool:
        return self.cal.is_business_day(d)

    def next_fixing_date(self, d: DateLike) -> _dt.date:
        """Business day after ``d`` (the end of the overnight period starting at ``d``)."""
        return self.cal.next_business_day(to_date(d), include_self=False)

    def fixing_dates(self, start: DateLike, end: DateLike) -> list[_dt.date]:
        """Business days in ``[start, end)``: the fixings that accrue over the period."""
        cal = self.cal
        d, e = cal.next_business_day(start), to_date(end)
        out = []
        while d < e:
            out.append(d)
            d = cal.next_business_day(d, include_self=False)
        return out

    def accrual(self, d: DateLike) -> float:
        """Accrual fraction of the overnight period starting at fixing date ``d``."""
        d = to_date(d)
        return self.dc.year_fraction(d, self.next_fixing_date(d))


@dataclass(frozen=True)
class IborIndex:
    """Term-rate index (EURIBOR-style).

    Parameters
    ----------
    name, currency : str
    tenor : str
        Index tenor (``"3M"``, ``"6M"``, ...).
    day_count, calendar : str
    fixing_days : int
        Spot lag: the fixing is ``fixing_days`` business days before the value date.
    convention : str
        Business-day convention of the maturity date.
    end_of_month : bool
        End-of-month rule for the maturity date.
    """

    name: str
    currency: str
    tenor: str
    day_count: str = "ACT/360"
    calendar: str = "TARGET"
    fixing_days: int = 2
    convention: str = "MF"
    end_of_month: bool = True

    def __post_init__(self) -> None:
        Tenor(self.tenor)
        if self.fixing_days < 0:
            raise ValueError("fixing_days must be non-negative")

    @property
    def cal(self) -> Calendar:
        return get_calendar(self.calendar)

    @property
    def dc(self) -> DayCount:
        return DayCount(self.day_count)

    def value_date(self, fixing_date: DateLike) -> _dt.date:
        """Start of the accrual period for a fixing on ``fixing_date``."""
        return self.cal.add_business_days(self.cal.next_business_day(fixing_date), self.fixing_days)

    def fixing_date(self, value_date: DateLike) -> _dt.date:
        """Fixing date for an accrual period starting on ``value_date``."""
        return (
            self.cal.add_business_days(value_date, -self.fixing_days)
            if self.fixing_days
            else (self.cal.next_business_day(value_date))
        )

    def maturity_date(self, value_date: DateLike) -> _dt.date:
        """End of the accrual period starting on ``value_date``."""
        v = to_date(value_date)
        eom = self.end_of_month and v == self.cal.previous_business_day(_last_of_month(v))
        raw = add_tenor(_last_of_month(v) if eom else v, self.tenor, end_of_month_rule=eom)
        return adjust(raw, self.cal, self.convention)

    def accrual(self, value_date: DateLike) -> float:
        """Accrual fraction of the period starting on ``value_date``."""
        v = to_date(value_date)
        return self.dc.year_fraction(v, self.maturity_date(v))


def _last_of_month(d: _dt.date) -> _dt.date:
    nxt = _dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - _dt.timedelta(days=1)


Index = Union[OvernightIndex, IborIndex]

SOFR = OvernightIndex("SOFR", "USD", "ACT/360", "US-SIFMA")
ESTR = OvernightIndex("ESTR", "EUR", "ACT/360", "TARGET")
SONIA = OvernightIndex("SONIA", "GBP", "ACT/365F", "UK")
EURIBOR1M = IborIndex("EURIBOR1M", "EUR", "1M")
EURIBOR3M = IborIndex("EURIBOR3M", "EUR", "3M")
EURIBOR6M = IborIndex("EURIBOR6M", "EUR", "6M")
EURIBOR12M = IborIndex("EURIBOR12M", "EUR", "12M")

_REGISTRY: dict[str, Index] = {
    i.name: i for i in (SOFR, ESTR, SONIA, EURIBOR1M, EURIBOR3M, EURIBOR6M, EURIBOR12M)
}
_ALIASES = {"€STR": "ESTR", "EURIBOR1Y": "EURIBOR12M"}


def available_indices() -> list[str]:
    return sorted(_REGISTRY)


def get_index(name: str) -> Index:
    """Look up a preset index by name (case-insensitive)."""
    key = name.upper().replace(" ", "").replace("-", "")
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown index {name!r}; available: {available_indices()}") from None


@dataclass
class FixingHistory:
    """Published fixings of one index, keyed by fixing date (rates as decimals)."""

    index: str
    fixings: dict[_dt.date, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = self.fixings
        self.fixings = {}
        for d, r in data.items():
            self.add(d, r)

    @classmethod
    def from_pairs(cls, index: str, pairs: Iterable[tuple[DateLike, float]]) -> "FixingHistory":
        h = cls(index)
        for d, r in pairs:
            h.add(d, r)
        return h

    def add(self, d: DateLike, rate: float) -> None:
        r = float(rate)
        if not math.isfinite(r):
            raise ValueError("fixings must be finite")
        self.fixings[to_date(d)] = r

    def get(self, d: DateLike) -> Optional[float]:
        return self.fixings.get(to_date(d))

    def __getitem__(self, d: DateLike) -> float:
        v = self.get(d)
        if v is None:
            raise KeyError(f"no {self.index} fixing for {to_date(d)}")
        return v

    def __contains__(self, d: object) -> bool:
        try:
            return to_date(d) in self.fixings  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __len__(self) -> int:
        return len(self.fixings)

    def __iter__(self) -> Iterator[_dt.date]:
        return iter(sorted(self.fixings))

    @property
    def last_date(self) -> Optional[_dt.date]:
        return max(self.fixings) if self.fixings else None

    def as_mapping(self) -> Mapping[_dt.date, float]:
        return dict(self.fixings)
