r"""Day-count conventions.

A day-count convention turns a pair of dates into an accrual fraction of a
year. The conventions follow the 2006 ISDA Definitions (section 4.16) and
the ICMA Rule 251:

=====================  ==========================================================
Name                   Year fraction
=====================  ==========================================================
``ACT/360``            :math:`(d_2 - d_1)/360`
``ACT/365F``           :math:`(d_2 - d_1)/365`
``ACT/364``            :math:`(d_2 - d_1)/364`
``ACT/ACT ISDA``       days falling in leap years / 366 + days in other years / 365
``ACT/ACT ICMA``       :math:`\text{days}/(f \cdot \text{days in reference period})`,
                       split into notional periods for long stubs
``30/360``             Bond basis (ISDA 4.16(f))
``30/360 US``          30/360 with the US end-of-February adjustments (SIA)
``30E/360``            Eurobond basis (ISDA 4.16(g))
``30E/360 ISDA``       ISDA 4.16(h): month-ends are 30, except a February
                       termination date
``BUS/252``            business days / 252 (requires a calendar)
=====================  ==========================================================

All 30/360 variants share
:math:`\tfrac{360(Y_2 - Y_1) + 30(M_2 - M_1) + (D_2 - D_1)}{360}` and differ
only in how :math:`D_1, D_2` are adjusted.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Callable, Optional, Protocol

from spxa.market.dates import (
    DateLike,
    add_months,
    days_in_year,
    is_end_of_month,
    to_date,
)

__all__ = [
    "DayCount",
    "available_day_counts",
    "day_count",
    "year_fraction",
]


class _BusinessDayCounter(Protocol):
    def business_days_between(self, start: DateLike, end: DateLike) -> int: ...


def _is_last_of_feb(d: _dt.date) -> bool:
    return d.month == 2 and is_end_of_month(d)


def _thirty_360(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


def _days_bond_basis(a: _dt.date, b: _dt.date, **_: Any) -> int:
    d1, d2 = a.day, b.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return _thirty_360(a.year, a.month, d1, b.year, b.month, d2)


def _days_30_360_us(a: _dt.date, b: _dt.date, eom: bool = True, **_: Any) -> int:
    d1, d2 = a.day, b.day
    if eom and _is_last_of_feb(a):
        if _is_last_of_feb(b):
            d2 = 30
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30
    if d1 == 31:
        d1 = 30
    return _thirty_360(a.year, a.month, d1, b.year, b.month, d2)


def _days_30e_360(a: _dt.date, b: _dt.date, **_: Any) -> int:
    d1, d2 = min(a.day, 30), min(b.day, 30)
    return _thirty_360(a.year, a.month, d1, b.year, b.month, d2)


def _days_30e_360_isda(
    a: _dt.date, b: _dt.date, termination: Optional[_dt.date] = None, **_: Any
) -> int:
    d1 = 30 if is_end_of_month(a) else a.day
    if is_end_of_month(b) and not (b.month == 2 and termination is not None and b == termination):
        d2 = 30
    else:
        d2 = b.day
    return _thirty_360(a.year, a.month, d1, b.year, b.month, d2)


def _actual(a: _dt.date, b: _dt.date, **_: Any) -> int:
    return (b - a).days


def _yf_act_act_isda(a: _dt.date, b: _dt.date, **_: Any) -> float:
    if a == b:
        return 0.0
    if a.year == b.year:
        return (b - a).days / days_in_year(a.year)
    first = (_dt.date(a.year + 1, 1, 1) - a).days / days_in_year(a.year)
    last = (b - _dt.date(b.year, 1, 1)).days / days_in_year(b.year)
    return first + (b.year - a.year - 1) + last


def _yf_act_act_icma(
    a: _dt.date,
    b: _dt.date,
    ref_start: Optional[_dt.date] = None,
    ref_end: Optional[_dt.date] = None,
    frequency: Optional[int] = None,
    end_of_month: bool = False,
    **_: Any,
) -> float:
    if a == b:
        return 0.0
    if frequency is None or frequency <= 0 or 12 % frequency:
        raise ValueError("ACT/ACT ICMA needs a coupon frequency dividing 12 (1, 2, 3, 4, 6, 12)")
    rs = ref_start if ref_start is not None else a
    re_ = ref_end if ref_end is not None else b
    months = 12 // frequency
    ref_days = (re_ - rs).days
    if ref_days <= 0:
        raise ValueError("reference period must have positive length")
    # Long first stub: the part before the reference period accrues against
    # the notional period(s) that precede it.
    if a < rs:
        prev = add_months(rs, -months, end_of_month)
        return _yf_act_act_icma(a, rs, prev, rs, frequency, end_of_month) + _yf_act_act_icma(
            rs, b, rs, re_, frequency, end_of_month
        )
    # Long final stub: the part after the reference period accrues against
    # the notional period(s) that follow it.
    if b > re_:
        nxt = add_months(re_, months, end_of_month)
        return _yf_act_act_icma(a, re_, rs, re_, frequency, end_of_month) + _yf_act_act_icma(
            re_, b, re_, nxt, frequency, end_of_month
        )
    return (b - a).days / (frequency * ref_days)


def _yf_bus_252(
    a: _dt.date, b: _dt.date, calendar: Optional[_BusinessDayCounter] = None, **_: Any
) -> float:
    if calendar is None:
        raise ValueError("BUS/252 needs a business-day calendar (calendar=...)")
    return calendar.business_days_between(a, b) / 252.0


# name -> (day counter or None, year-fraction function)
_DayFn = Callable[..., int]
_YfFn = Callable[..., float]


def _from_days(days: _DayFn, basis: float) -> _YfFn:
    def yf(a: _dt.date, b: _dt.date, **kw: Any) -> float:
        return days(a, b, **kw) / basis

    return yf


_CONVENTIONS: dict[str, tuple[Optional[_DayFn], _YfFn]] = {
    "ACT/360": (_actual, _from_days(_actual, 360.0)),
    "ACT/365F": (_actual, _from_days(_actual, 365.0)),
    "ACT/364": (_actual, _from_days(_actual, 364.0)),
    "ACT/ACT ISDA": (_actual, _yf_act_act_isda),
    "ACT/ACT ICMA": (_actual, _yf_act_act_icma),
    "30/360": (_days_bond_basis, _from_days(_days_bond_basis, 360.0)),
    "30/360 US": (_days_30_360_us, _from_days(_days_30_360_us, 360.0)),
    "30E/360": (_days_30e_360, _from_days(_days_30e_360, 360.0)),
    "30E/360 ISDA": (_days_30e_360_isda, _from_days(_days_30e_360_isda, 360.0)),
    "BUS/252": (None, _yf_bus_252),
}

_ALIASES = {
    "ACT/360": "ACT/360",
    "A/360": "ACT/360",
    "ACTUAL/360": "ACT/360",
    "ACT/365": "ACT/365F",
    "ACT/365F": "ACT/365F",
    "ACT/365 FIXED": "ACT/365F",
    "ACTUAL/365 FIXED": "ACT/365F",
    "A/365F": "ACT/365F",
    "ACTUAL/365F": "ACT/365F",
    "ACT/364": "ACT/364",
    "ACT/ACT": "ACT/ACT ISDA",
    "ACT/ACT ISDA": "ACT/ACT ISDA",
    "ACTUAL/ACTUAL ISDA": "ACT/ACT ISDA",
    "ACT/ACT ICMA": "ACT/ACT ICMA",
    "ACT/ACT ISMA": "ACT/ACT ICMA",
    "ACTUAL/ACTUAL ICMA": "ACT/ACT ICMA",
    "30/360": "30/360",
    "30/360 BOND": "30/360",
    "30/360 BOND BASIS": "30/360",
    "30/360 ISDA": "30/360",
    "30/360 US": "30/360 US",
    "30U/360": "30/360 US",
    "30/360 SIA": "30/360 US",
    "30E/360": "30E/360",
    "30/360 ICMA": "30E/360",
    "EUROBOND BASIS": "30E/360",
    "30E/360 ISDA": "30E/360 ISDA",
    "30E/360 GERMAN": "30E/360 ISDA",
    "BUS/252": "BUS/252",
}


def _normalise(name: str) -> str:
    return re.sub(r"[\s/_\-]", "", str(name).upper())


_LOOKUP = {_normalise(k): v for k, v in _ALIASES.items()}


def _canonical(name: str) -> str:
    try:
        return _LOOKUP[_normalise(name)]
    except KeyError:
        raise ValueError(
            f"unknown day-count convention {name!r}; available: {available_day_counts()}"
        ) from None


def available_day_counts() -> list[str]:
    """Canonical names of the supported conventions."""
    return list(_CONVENTIONS)


class DayCount:
    """A day-count convention.

    Parameters
    ----------
    name : str
        Convention name or alias (case-insensitive), e.g. ``"ACT/360"``,
        ``"Actual/365 Fixed"``, ``"30E/360"``, ``"ACT/ACT ICMA"``.

    Examples
    --------
    >>> DayCount("ACT/ACT ISDA").year_fraction("2003-11-01", "2004-05-01")
    0.4977243...
    """

    __slots__ = ("name",)

    name: str

    def __init__(self, name: "str | DayCount") -> None:
        self.name = name.name if isinstance(name, DayCount) else _canonical(name)

    def day_count(self, start: DateLike, end: DateLike, **kwargs: Any) -> int:
        """Number of accrual days under the convention (calendar days for ACT)."""
        days, _ = _CONVENTIONS[self.name]
        if days is None:
            raise ValueError(f"{self.name} has no calendar-independent day count")
        return days(to_date(start), to_date(end), **_dates(kwargs))

    def year_fraction(self, start: DateLike, end: DateLike, **kwargs: Any) -> float:
        """Accrual fraction from ``start`` to ``end``.

        Keyword arguments, used only by the conventions that need them:

        ``ref_start``, ``ref_end``, ``frequency``, ``end_of_month``
            ACT/ACT ICMA reference coupon period and coupons per year.
        ``termination``
            30E/360 ISDA: the final date of the schedule.
        ``eom``
            30/360 US: apply the end-of-February rules (default ``True``).
        ``calendar``
            BUS/252: an object with ``business_days_between(start, end)``.

        If ``end < start`` the result is the negative of the reversed
        fraction.
        """
        a, b = to_date(start), to_date(end)
        if b < a:
            return -self.year_fraction(b, a, **kwargs)
        _, yf = _CONVENTIONS[self.name]
        return float(yf(a, b, **_dates(kwargs)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = DayCount(other)
            except ValueError:
                return False
        return isinstance(other, DayCount) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"DayCount('{self.name}')"

    def __str__(self) -> str:
        return self.name


def _dates(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    for key in ("ref_start", "ref_end", "termination"):
        if out.get(key) is not None:
            out[key] = to_date(out[key])
    return out


def year_fraction(
    start: DateLike, end: DateLike, convention: "str | DayCount", **kwargs: Any
) -> float:
    """Functional form of :meth:`DayCount.year_fraction`."""
    return DayCount(convention).year_fraction(start, end, **kwargs)


def day_count(start: DateLike, end: DateLike, convention: "str | DayCount", **kwargs: Any) -> int:
    """Functional form of :meth:`DayCount.day_count`."""
    return DayCount(convention).day_count(start, end, **kwargs)
