r"""Business-day calendars.

Built-in calendars (look them up with :func:`get_calendar`):

``WEEKENDS``
    Saturdays and Sundays only.
``TARGET``
    The Eurosystem TARGET2/T2 calendar: 1 January, Good Friday, Easter
    Monday, 1 May, 25 and 26 December (the Easter and May/Boxing Day
    closures from 2000), and 31 December in 1998, 1999 and 2001.
``US-FED``
    Federal Reserve Bank holidays (SOFR is not published on these). A
    holiday on a Sunday moves to Monday; one on a Saturday is not observed.
    Juneteenth from 2022.
``US-SIFMA``
    U.S. Government Securities business days following the SIFMA
    recommendations used for SOFR-linked instruments: the federal holidays
    plus Good Friday; Saturday holidays move to Friday except New Year's
    Day. SIFMA sometimes recommends only an early close on Good Friday;
    such years can be added with ``extra_business_days``.
``UK``
    England & Wales bank holidays (London), including substitute days and
    the one-off holidays of 1999, 2002, 2011, 2012, 2020, 2022 and 2023.

Calendars can be combined with :class:`JointCalendar` and extended with
ad-hoc holidays or business days.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Callable, Iterable, Optional, Sequence

from spxa.market.dates import DateLike, to_date

__all__ = [
    "Calendar",
    "JointCalendar",
    "available_calendars",
    "easter_sunday",
    "get_calendar",
]

_ONE_DAY = _dt.timedelta(days=1)


def easter_sunday(year: int) -> _dt.date:
    """Gregorian Easter Sunday (anonymous Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return _dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """``n``-th ``weekday`` (Mon=0) of the month; ``n = -1`` is the last one."""
    if n > 0:
        first = _dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + _dt.timedelta(days=offset + 7 * (n - 1))
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - _ONE_DAY
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: _dt.date, saturday_to_friday: bool = True) -> Optional[_dt.date]:
    """US observance: Sunday -> Monday, Saturday -> Friday (or not observed)."""
    if d.weekday() == 6:
        return d + _ONE_DAY
    if d.weekday() == 5:
        return d - _ONE_DAY if saturday_to_friday else None
    return d


# ----------------------------------------------------------------------
# Holiday rules: year -> set of holiday dates (weekends are added separately)
# ----------------------------------------------------------------------


def _target_holidays(y: int) -> set[_dt.date]:
    out = {_dt.date(y, 1, 1), _dt.date(y, 12, 25)}
    if y >= 2000:
        e = easter_sunday(y)
        out |= {e - 2 * _ONE_DAY, e + _ONE_DAY, _dt.date(y, 5, 1), _dt.date(y, 12, 26)}
    if y in (1998, 1999, 2001):
        out.add(_dt.date(y, 12, 31))
    return out


def _us_federal(y: int, saturday_to_friday: bool, new_year_saturday: bool) -> set[_dt.date]:
    fixed = [_dt.date(y, 7, 4), _dt.date(y, 11, 11), _dt.date(y, 12, 25)]
    if y >= 2022:
        fixed.append(_dt.date(y, 6, 19))
    out = {o for d in fixed if (o := _observed(d, saturday_to_friday)) is not None}
    # New Year's Day of y (Saturday: observed on 31 Dec of y - 1 only if requested)
    ny = _observed(_dt.date(y, 1, 1), new_year_saturday)
    if ny is not None and ny.year == y:
        out.add(ny)
    nxt = _observed(_dt.date(y + 1, 1, 1), new_year_saturday)
    if nxt is not None and nxt.year == y:
        out.add(nxt)
    out |= {
        _nth_weekday(y, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(y, 2, 0, 3),  # Washington's Birthday
        _nth_weekday(y, 5, 0, -1),  # Memorial Day
        _nth_weekday(y, 9, 0, 1),  # Labor Day
        _nth_weekday(y, 10, 0, 2),  # Columbus Day
        _nth_weekday(y, 11, 3, 4),  # Thanksgiving
    }
    return out


def _us_fed_holidays(y: int) -> set[_dt.date]:
    return _us_federal(y, saturday_to_friday=False, new_year_saturday=False)


def _us_sifma_holidays(y: int) -> set[_dt.date]:
    out = _us_federal(y, saturday_to_friday=True, new_year_saturday=False)
    out.add(easter_sunday(y) - 2 * _ONE_DAY)
    return out


_UK_SPECIAL = {
    _dt.date(1999, 12, 31),  # Millennium
    _dt.date(2002, 6, 3),  # Golden Jubilee
    _dt.date(2011, 4, 29),  # Royal Wedding
    _dt.date(2012, 6, 5),  # Diamond Jubilee
    _dt.date(2022, 6, 3),  # Platinum Jubilee
    _dt.date(2022, 9, 19),  # State funeral of Queen Elizabeth II
    _dt.date(2023, 5, 8),  # Coronation of King Charles III
}
_UK_EARLY_MAY_MOVED = {1995: _dt.date(1995, 5, 8), 2020: _dt.date(2020, 5, 8)}
_UK_SPRING_MOVED = {
    2002: _dt.date(2002, 6, 4),
    2012: _dt.date(2012, 6, 4),
    2022: _dt.date(2022, 6, 2),
}


def _uk_holidays(y: int) -> set[_dt.date]:
    e = easter_sunday(y)
    out = {e - 2 * _ONE_DAY, e + _ONE_DAY}
    ny = _dt.date(y, 1, 1)
    out.add(ny + _dt.timedelta(days={5: 2, 6: 1}.get(ny.weekday(), 0)))
    out.add(_UK_EARLY_MAY_MOVED.get(y, _nth_weekday(y, 5, 0, 1)))
    out.add(_UK_SPRING_MOVED.get(y, _nth_weekday(y, 5, 0, -1)))
    out.add(_nth_weekday(y, 8, 0, -1))
    xmas = _dt.date(y, 12, 25)
    wd = xmas.weekday()
    if wd == 5:  # Saturday: Christmas -> Mon 27, Boxing Day -> Tue 28
        out |= {_dt.date(y, 12, 27), _dt.date(y, 12, 28)}
    elif wd == 6:  # Sunday: Boxing Day Mon 26, Christmas -> Tue 27
        out |= {_dt.date(y, 12, 26), _dt.date(y, 12, 27)}
    elif wd == 4:  # Friday: Boxing Day (Saturday) -> Mon 28
        out |= {xmas, _dt.date(y, 12, 28)}
    else:
        out |= {xmas, _dt.date(y, 12, 26)}
    out |= {d for d in _UK_SPECIAL if d.year == y}
    return out


_RULES: dict[str, Callable[[int], set[_dt.date]]] = {
    "WEEKENDS": lambda y: set(),
    "TARGET": _target_holidays,
    "US-FED": _us_fed_holidays,
    "US-SIFMA": _us_sifma_holidays,
    "UK": _uk_holidays,
}
_CAL_ALIASES = {
    "WEEKENDS": "WEEKENDS",
    "WEEKENDSONLY": "WEEKENDS",
    "NONE": "WEEKENDS",
    "TARGET": "TARGET",
    "TARGET2": "TARGET",
    "EUR": "TARGET",
    "USFED": "US-FED",
    "FED": "US-FED",
    "FRB": "US-FED",
    "USSIFMA": "US-SIFMA",
    "SIFMA": "US-SIFMA",
    "USGOV": "US-SIFMA",
    "SOFR": "US-SIFMA",
    "USD": "US-SIFMA",
    "UK": "UK",
    "GBLO": "UK",
    "LONDON": "UK",
    "GBP": "UK",
}


@lru_cache(maxsize=4096)
def _holidays_for(rule: str, year: int) -> frozenset[_dt.date]:
    return frozenset(d for d in _RULES[rule](year) if d.weekday() < 5)


def available_calendars() -> list[str]:
    """Names of the built-in calendars."""
    return list(_RULES)


class Calendar:
    """A business-day calendar.

    Parameters
    ----------
    name : str
        Built-in calendar name or alias (``"TARGET"``, ``"US-SIFMA"``,
        ``"SOFR"``, ``"US-FED"``, ``"UK"``, ``"WEEKENDS"``).
    extra_holidays, extra_business_days : iterable of date-like, optional
        Ad-hoc adjustments applied on top of the rule-based holidays.
    """

    def __init__(
        self,
        name: str = "WEEKENDS",
        extra_holidays: Iterable[DateLike] = (),
        extra_business_days: Iterable[DateLike] = (),
    ) -> None:
        key = "".join(ch for ch in str(name).upper() if ch.isalnum())
        if key not in _CAL_ALIASES:
            raise ValueError(f"unknown calendar {name!r}; available: {available_calendars()}")
        self.rule = _CAL_ALIASES[key]
        self.extra_holidays = frozenset(to_date(d) for d in extra_holidays)
        self.extra_business_days = frozenset(to_date(d) for d in extra_business_days)

    @property
    def name(self) -> str:
        return self.rule

    # -- queries --------------------------------------------------------
    def is_holiday(self, d: DateLike) -> bool:
        """Whether ``d`` is a weekend day or a holiday."""
        return not self.is_business_day(d)

    def is_business_day(self, d: DateLike) -> bool:
        d = to_date(d)
        if d in self.extra_business_days:
            return True
        if d in self.extra_holidays or d.weekday() >= 5:
            return False
        return d not in _holidays_for(self.rule, d.year)

    def holidays(
        self, start: DateLike, end: DateLike, include_weekends: bool = False
    ) -> list[_dt.date]:
        """Holidays in ``[start, end]`` (weekdays only unless ``include_weekends``)."""
        a, b = to_date(start), to_date(end)
        out = []
        d = a
        while d <= b:
            if not self.is_business_day(d) and (include_weekends or d.weekday() < 5):
                out.append(d)
            d += _ONE_DAY
        return out

    def business_days_between(self, start: DateLike, end: DateLike) -> int:
        """Business days in ``[start, end)``; negative if ``end < start``."""
        a, b = to_date(start), to_date(end)
        if b < a:
            return -self.business_days_between(b, a)
        n = 0
        d = a
        while d < b:
            n += self.is_business_day(d)
            d += _ONE_DAY
        return n

    def next_business_day(self, d: DateLike, include_self: bool = True) -> _dt.date:
        d = to_date(d)
        if not include_self:
            d += _ONE_DAY
        while not self.is_business_day(d):
            d += _ONE_DAY
        return d

    def previous_business_day(self, d: DateLike, include_self: bool = True) -> _dt.date:
        d = to_date(d)
        if not include_self:
            d -= _ONE_DAY
        while not self.is_business_day(d):
            d -= _ONE_DAY
        return d

    def add_business_days(self, d: DateLike, n: int) -> _dt.date:
        """Move ``n`` business days (``n = 0`` rolls a holiday forward)."""
        d = to_date(d)
        if n == 0:
            return self.next_business_day(d)
        step = _ONE_DAY if n > 0 else -_ONE_DAY
        remaining = abs(n)
        while remaining:
            d += step
            if self.is_business_day(d):
                remaining -= 1
        return d

    def with_holidays(self, *dates: DateLike) -> "Calendar":
        """Copy with additional holidays."""
        return self._copy(extra_holidays=self.extra_holidays | {to_date(x) for x in dates})

    def with_business_days(self, *dates: DateLike) -> "Calendar":
        """Copy with additional business days."""
        return self._copy(
            extra_business_days=self.extra_business_days | {to_date(x) for x in dates}
        )

    def _copy(self, **changes: frozenset[_dt.date]) -> "Calendar":
        cal = Calendar.__new__(Calendar)
        cal.rule = self.rule
        cal.extra_holidays = changes.get("extra_holidays", self.extra_holidays)
        cal.extra_business_days = changes.get("extra_business_days", self.extra_business_days)
        return cal

    def __repr__(self) -> str:
        return f"Calendar('{self.rule}')"


class JointCalendar(Calendar):
    """Combination of calendars.

    Parameters
    ----------
    calendars : sequence of Calendar or str
    rule : {"holiday_if_any", "holiday_if_all"}
        ``"holiday_if_any"`` (default, e.g. a USD/EUR swap) makes a day a
        business day only if it is one in every calendar.
    """

    def __init__(self, calendars: Sequence["Calendar | str"], rule: str = "holiday_if_any") -> None:
        if rule not in ("holiday_if_any", "holiday_if_all"):
            raise ValueError("rule must be 'holiday_if_any' or 'holiday_if_all'")
        if not calendars:
            raise ValueError("need at least one calendar")
        self.calendars = tuple(c if isinstance(c, Calendar) else Calendar(c) for c in calendars)
        self.join_rule = rule
        self.rule = "+".join(c.name for c in self.calendars)
        self.extra_holidays = frozenset()
        self.extra_business_days = frozenset()

    def is_business_day(self, d: DateLike) -> bool:
        d = to_date(d)
        if d in self.extra_business_days:
            return True
        if d in self.extra_holidays:
            return False
        flags = [c.is_business_day(d) for c in self.calendars]
        return all(flags) if self.join_rule == "holiday_if_any" else any(flags)

    def _copy(self, **changes: frozenset[_dt.date]) -> "Calendar":
        cal = JointCalendar(self.calendars, self.join_rule)
        cal.extra_holidays = changes.get("extra_holidays", self.extra_holidays)
        cal.extra_business_days = changes.get("extra_business_days", self.extra_business_days)
        return cal

    def __repr__(self) -> str:
        return f"JointCalendar({[c.name for c in self.calendars]}, rule='{self.join_rule}')"


def get_calendar(spec: "str | Calendar | Sequence[str | Calendar]") -> Calendar:
    """Resolve a calendar from a name, a ``"A+B"`` string, a sequence or a Calendar."""
    if isinstance(spec, Calendar):
        return spec
    if isinstance(spec, str):
        parts = [p for p in spec.split("+") if p.strip()]
        return Calendar(parts[0]) if len(parts) == 1 else JointCalendar(parts)
    return JointCalendar(list(spec))
