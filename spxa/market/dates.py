r"""Calendar-date utilities shared by :mod:`spxa.market`, :mod:`spxa.rates` and :mod:`spxa.vol`.

Dates are plain :class:`datetime.date` objects. Every public function accepts
anything :func:`to_date` understands (``date``, ``datetime``, ISO strings
``"YYYY-MM-DD"``, ``numpy.datetime64`` and ``pandas.Timestamp``-like objects
exposing ``.date()``) and returns ``date``.

Month arithmetic follows the conventions used for swap and bond schedules:
adding months clamps to the last day of the target month, and the optional
end-of-month rule maps month-ends to month-ends.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as _dt
import re
from typing import Any, Union

import numpy as np

__all__ = [
    "DateLike",
    "Tenor",
    "add_months",
    "add_tenor",
    "days_between",
    "days_in_month",
    "days_in_year",
    "end_of_month",
    "is_end_of_month",
    "is_leap_year",
    "to_date",
]

DateLike = Union[_dt.date, _dt.datetime, str, np.datetime64, Any]

_TENOR_RE = re.compile(r"(?:[+-]?\d+[DWMY])+")


def to_date(value: DateLike) -> _dt.date:
    """Convert ``value`` to :class:`datetime.date`.

    Accepts ``date``, ``datetime`` (the time is dropped), ISO-8601 strings,
    ``numpy.datetime64`` and objects with a ``date()`` method.

    Raises
    ------
    TypeError
        If the value cannot be interpreted as a date.
    """
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise TypeError(f"cannot parse date string {value!r}") from exc
    if isinstance(value, np.datetime64):
        days = value.astype("datetime64[D]").astype(np.int64)
        return _dt.date(1970, 1, 1) + _dt.timedelta(days=int(days))
    date_method = getattr(value, "date", None)
    if callable(date_method):
        out = date_method()
        if isinstance(out, _dt.date):
            return out
    raise TypeError(f"cannot convert {type(value).__name__} to a date")


def is_leap_year(year: int) -> bool:
    """Gregorian leap-year rule."""
    return _calendar.isleap(year)


def days_in_year(year: int) -> int:
    """365 or 366."""
    return 366 if is_leap_year(year) else 365


def days_in_month(year: int, month: int) -> int:
    """Number of days in ``month`` of ``year``."""
    return _calendar.monthrange(year, month)[1]


def end_of_month(d: DateLike) -> _dt.date:
    """Last calendar day of the month containing ``d``."""
    d = to_date(d)
    return d.replace(day=days_in_month(d.year, d.month))


def is_end_of_month(d: DateLike) -> bool:
    """Whether ``d`` is the last calendar day of its month."""
    d = to_date(d)
    return d.day == days_in_month(d.year, d.month)


def days_between(start: DateLike, end: DateLike) -> int:
    """Actual number of days from ``start`` to ``end`` (negative if reversed)."""
    return (to_date(end) - to_date(start)).days


def add_months(d: DateLike, months: int, end_of_month_rule: bool = False) -> _dt.date:
    """Add calendar months, clamping the day to the target month's length.

    Parameters
    ----------
    d : date-like
        Start date.
    months : int
        Number of months (may be negative).
    end_of_month_rule : bool
        If ``True`` and ``d`` is the last day of its month, the result is the
        last day of the target month (e.g. 28 Feb 2023 + 1M = 31 Mar 2023).

    Examples
    --------
    >>> add_months("2024-01-31", 1)
    datetime.date(2024, 2, 29)
    >>> add_months("2023-02-28", 1, end_of_month_rule=True)
    datetime.date(2023, 3, 31)
    """
    d = to_date(d)
    total = d.year * 12 + (d.month - 1) + int(months)
    year, month = divmod(total, 12)
    month += 1
    last = days_in_month(year, month)
    if end_of_month_rule and is_end_of_month(d):
        return _dt.date(year, month, last)
    return _dt.date(year, month, min(d.day, last))


class Tenor:
    """A period such as ``3M``, ``10Y``, ``2W`` or ``1D``.

    Parameters
    ----------
    spec : str or Tenor
        ``"<n><unit>"`` with unit ``D`` (days), ``W`` (weeks), ``M`` (months)
        or ``Y`` (years); case-insensitive. Composite specs such as ``"1Y6M"``
        are accepted and reduced to months when possible.
    """

    __slots__ = ("n", "unit")

    n: int
    unit: str

    def __init__(self, spec: Union[str, "Tenor"], unit: str | None = None) -> None:
        if isinstance(spec, Tenor):
            self.n, self.unit = spec.n, spec.unit
            return
        if unit is not None:
            self.n, self.unit = int(spec), unit.upper()
            if self.unit not in "DWMY" or len(self.unit) != 1:
                raise ValueError(f"invalid tenor unit {unit!r}")
            return
        text = re.sub(r"\s+", "", str(spec)).upper()
        if not _TENOR_RE.fullmatch(text):
            raise ValueError(f"invalid tenor {spec!r}")
        parts = re.findall(r"([+-]?\d+)([DWMY])", text)
        if len(parts) == 1:
            self.n, self.unit = int(parts[0][0]), parts[0][1].upper()
            return
        units = {u.upper() for _, u in parts}
        if units <= {"M", "Y"}:
            self.n = sum(int(a) * (12 if b.upper() == "Y" else 1) for a, b in parts)
            self.unit = "M"
        elif units <= {"D", "W"}:
            self.n = sum(int(a) * (7 if b.upper() == "W" else 1) for a, b in parts)
            self.unit = "D"
        else:
            raise ValueError(f"cannot combine day/week and month/year units in {spec!r}")

    @property
    def months(self) -> int:
        """Length in months (``Y`` and ``M`` tenors only)."""
        if self.unit == "M":
            return self.n
        if self.unit == "Y":
            return 12 * self.n
        raise ValueError(f"tenor {self} is not expressed in months or years")

    @property
    def days(self) -> int:
        """Length in days (``D`` and ``W`` tenors only)."""
        if self.unit == "D":
            return self.n
        if self.unit == "W":
            return 7 * self.n
        raise ValueError(f"tenor {self} is not expressed in days or weeks")

    @property
    def is_month_based(self) -> bool:
        return self.unit in ("M", "Y")

    def approx_years(self) -> float:
        """Nominal length in years (``D``/365, ``W``·7/365, ``M``/12, ``Y``)."""
        if self.is_month_based:
            return self.months / 12.0
        return self.days / 365.0

    def __neg__(self) -> "Tenor":
        return Tenor(-self.n, self.unit)

    def __mul__(self, k: int) -> "Tenor":
        return Tenor(self.n * int(k), self.unit)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = Tenor(other)
            except ValueError:
                return False
        if not isinstance(other, Tenor):
            return NotImplemented
        if self.is_month_based and other.is_month_based:
            return self.months == other.months
        if not self.is_month_based and not other.is_month_based:
            return self.days == other.days
        return False

    def __hash__(self) -> int:
        return hash(("M", self.months) if self.is_month_based else ("D", self.days))

    def __repr__(self) -> str:
        return f"Tenor('{self}')"

    def __str__(self) -> str:
        return f"{self.n}{self.unit}"


def add_tenor(d: DateLike, tenor: Union[str, Tenor], end_of_month_rule: bool = False) -> _dt.date:
    """Unadjusted ``d + tenor`` (calendar arithmetic, no business-day roll).

    Month and year tenors use :func:`add_months`; day and week tenors add
    calendar days.
    """
    t = Tenor(tenor)
    if t.is_month_based:
        return add_months(d, t.months, end_of_month_rule)
    return to_date(d) + _dt.timedelta(days=t.days)
