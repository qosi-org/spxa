r"""Business-day conventions and coupon schedules.

Business-day conventions (ISDA 2006 section 4.12):

==========  ==================================================================
``F``       Following: the next business day.
``MF``      Modified following: following, unless that crosses into the next
            month, in which case preceding.
``P``       Preceding: the previous business day.
``MP``      Modified preceding: preceding, unless that crosses into the
            previous month, in which case following.
``U``       Unadjusted.
==========  ==================================================================

:class:`Schedule` generates unadjusted and adjusted coupon dates between an
effective and a termination date. Regular dates are anchored at one end and
computed as ``anchor ± k·tenor`` (never iteratively, so month-end clamping
does not drift). The stub, if the tenor does not divide the period, sits at
the opposite end:

* ``short_front`` / ``long_front``: dates rolled backward from the
  termination date (the market default for swaps);
* ``short_back`` / ``long_back``: dates rolled forward from the effective
  date;
* explicit ``first_date`` / ``next_to_last_date`` stubs.

With ``end_of_month=True`` and a month-end anchor every regular date is a
month end.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from spxa.market.calendars import Calendar, get_calendar
from spxa.market.dates import DateLike, Tenor, add_tenor, is_end_of_month, to_date
from spxa.market.daycount import DayCount

__all__ = [
    "AccrualPeriod",
    "Schedule",
    "adjust",
    "available_business_day_conventions",
]

_CONVENTIONS = {
    "F": "F",
    "FOLLOWING": "F",
    "MF": "MF",
    "MODIFIEDFOLLOWING": "MF",
    "MODFOLLOWING": "MF",
    "P": "P",
    "PRECEDING": "P",
    "MP": "MP",
    "MODIFIEDPRECEDING": "MP",
    "U": "U",
    "NONE": "U",
    "UNADJUSTED": "U",
}

CalendarLike = Union[str, Calendar]


def _bdc(name: str) -> str:
    key = "".join(ch for ch in str(name).upper() if ch.isalnum())
    try:
        return _CONVENTIONS[key]
    except KeyError:
        raise ValueError(
            f"unknown business-day convention {name!r}; use one of F, MF, P, MP, U"
        ) from None


def available_business_day_conventions() -> list[str]:
    """Canonical business-day convention codes."""
    return ["F", "MF", "P", "MP", "U"]


def adjust(d: DateLike, calendar: CalendarLike = "WEEKENDS", convention: str = "MF") -> _dt.date:
    """Roll ``d`` to a business day of ``calendar`` according to ``convention``."""
    d = to_date(d)
    conv = _bdc(convention)
    if conv == "U":
        return d
    cal = get_calendar(calendar)
    if cal.is_business_day(d):
        return d
    if conv in ("F", "MF"):
        out = cal.next_business_day(d)
        if conv == "MF" and out.month != d.month:
            out = cal.previous_business_day(d)
        return out
    out = cal.previous_business_day(d)
    if conv == "MP" and out.month != d.month:
        out = cal.next_business_day(d)
    return out


@dataclass(frozen=True)
class AccrualPeriod:
    """One coupon period of a :class:`Schedule`."""

    start: _dt.date
    end: _dt.date
    unadjusted_start: _dt.date
    unadjusted_end: _dt.date
    ref_start: _dt.date
    ref_end: _dt.date
    is_regular: bool

    def year_fraction(self, day_count: "str | DayCount", frequency: Optional[int] = None) -> float:
        """Accrual fraction; ACT/ACT ICMA uses this period's reference dates."""
        dc = DayCount(day_count)
        if dc.name == "ACT/ACT ICMA":
            return dc.year_fraction(
                self.start,
                self.end,
                ref_start=self.ref_start,
                ref_end=self.ref_end,
                frequency=frequency,
            )
        return dc.year_fraction(self.start, self.end)


class Schedule:
    """Coupon schedule.

    Parameters
    ----------
    effective, termination : date-like
        Unadjusted start and end of the schedule.
    tenor : str, Tenor or None
        Coupon period (``"3M"``, ``"6M"``, ``"1Y"``, ``"28D"`` ...). ``None``
        gives a single period.
    calendar : str or Calendar
        Calendar used for adjustment (default ``"WEEKENDS"``).
    convention : str
        Business-day convention for the effective and intermediate dates.
    termination_convention : str, optional
        Convention for the termination date (defaults to ``convention``).
    stub : str
        ``"short_front"`` (default), ``"long_front"``, ``"short_back"``,
        ``"long_back"``; ignored when explicit stub dates are given.
    end_of_month : bool
        Apply the end-of-month rule to month-based tenors when the anchor
        date is a month end.
    first_date, next_to_last_date : date-like, optional
        Explicit end of the front stub / start of the back stub.
    """

    def __init__(
        self,
        effective: DateLike,
        termination: DateLike,
        tenor: "str | Tenor | None" = "6M",
        calendar: CalendarLike = "WEEKENDS",
        convention: str = "MF",
        termination_convention: Optional[str] = None,
        stub: str = "short_front",
        end_of_month: bool = False,
        first_date: Optional[DateLike] = None,
        next_to_last_date: Optional[DateLike] = None,
    ) -> None:
        self.effective = to_date(effective)
        self.termination = to_date(termination)
        if self.termination <= self.effective:
            raise ValueError("termination must be after effective")
        if stub not in ("short_front", "long_front", "short_back", "long_back"):
            raise ValueError(f"invalid stub {stub!r}")
        self.tenor = None if tenor is None else Tenor(tenor)
        if self.tenor is not None and self.tenor.n <= 0:
            raise ValueError("tenor must be positive")
        self.calendar = get_calendar(calendar)
        self.convention = _bdc(convention)
        self.termination_convention = _bdc(termination_convention or convention)
        self.stub = stub
        self.end_of_month = bool(end_of_month) and (
            self.tenor is not None and self.tenor.is_month_based
        )
        self.first_date = None if first_date is None else to_date(first_date)
        self.next_to_last_date = None if next_to_last_date is None else to_date(next_to_last_date)
        self.unadjusted_dates, self._regular = self._generate()
        self.dates = self._adjust_all()

    # ------------------------------------------------------------------
    def _roll(self, anchor: _dt.date, k: int, eom: bool) -> _dt.date:
        assert self.tenor is not None
        if self.tenor.is_month_based:
            return add_tenor(anchor, Tenor(k * self.tenor.months, "M"), eom)
        return anchor + _dt.timedelta(days=k * self.tenor.days)

    def _anchored(self, lo: _dt.date, hi: _dt.date, backward: bool) -> tuple[list[_dt.date], bool]:
        """Regular dates strictly between ``lo`` and ``hi``, anchored at ``hi``
        (backward) or ``lo`` (forward), and whether they tile the interval exactly."""
        anchor = hi if backward else lo
        eom = self.end_of_month and is_end_of_month(anchor)
        out: list[_dt.date] = []
        k = 1
        while True:
            d = self._roll(anchor, -k if backward else k, eom)
            if backward and d <= lo:
                return sorted(out), d == lo
            if not backward and d >= hi:
                return out, d == hi
            out.append(d)
            k += 1

    def _one_tenor(self, a: _dt.date, b: _dt.date, anchor_at_end: bool) -> bool:
        if anchor_at_end:
            return self._roll(b, -1, self.end_of_month and is_end_of_month(b)) == a
        return self._roll(a, 1, self.end_of_month and is_end_of_month(a)) == b

    def _generate(self) -> tuple[list[_dt.date], list[bool]]:
        a, b = self.effective, self.termination
        if self.tenor is None:
            return [a, b], [True]
        fd, ntl = self.first_date, self.next_to_last_date
        for x in (fd, ntl):
            if x is not None and not a < x < b:
                raise ValueError("explicit stub dates must lie strictly inside the schedule")
        if fd is not None and ntl is not None and ntl <= fd:
            raise ValueError("next_to_last_date must be after first_date")

        if fd is not None and ntl is not None:
            inner, exact = self._anchored(fd, ntl, backward=False)
            if not exact:
                raise ValueError(
                    "first_date and next_to_last_date are not a whole number of tenors apart"
                )
            dates = [a, fd, *inner, ntl, b]
            regular = [True] * (len(dates) - 1)
            regular[0] = self._one_tenor(a, fd, anchor_at_end=True)
            regular[-1] = self._one_tenor(ntl, b, anchor_at_end=False)
            return dates, regular
        if fd is not None:
            inner, exact = self._anchored(fd, b, backward=False)
            dates = [a, fd, *inner, b]
            regular = [True] * (len(dates) - 1)
            regular[0] = self._one_tenor(a, fd, anchor_at_end=True)
            regular[-1] = regular[-1] and exact
            return dates, regular
        if ntl is not None:
            inner, exact = self._anchored(a, ntl, backward=True)
            dates = [a, *inner, ntl, b]
            regular = [True] * (len(dates) - 1)
            regular[-1] = self._one_tenor(ntl, b, anchor_at_end=False)
            regular[0] = regular[0] and exact
            return dates, regular

        backward = self.stub in ("short_front", "long_front")
        inner, exact = self._anchored(a, b, backward=backward)
        dates = [a, *inner, b]
        if not exact and self.stub in ("long_front", "long_back") and len(dates) > 2:
            del dates[1 if backward else -2]
        regular = [True] * (len(dates) - 1)
        if not exact:
            regular[0 if backward else -1] = False
        return dates, regular

    def _adjust_all(self) -> list[_dt.date]:
        out = [adjust(d, self.calendar, self.convention) for d in self.unadjusted_dates[:-1]]
        out.append(adjust(self.termination, self.calendar, self.termination_convention))
        for x, y in zip(out, out[1:]):
            if y <= x:
                raise ValueError("adjusted schedule dates are not strictly increasing")
        return out

    # ------------------------------------------------------------------
    def periods(self) -> list[AccrualPeriod]:
        """Accrual periods with ICMA reference periods for the stubs."""
        out = []
        ud, ad = self.unadjusted_dates, self.dates
        n = len(ud) - 1
        for i in range(n):
            ref_s, ref_e = ud[i], ud[i + 1]
            if not self._regular[i] and self.tenor is not None:
                front = i == 0 and (n > 1 or self._front_stub)
                if front:
                    ref_s = self._roll(ref_e, -1, self.end_of_month and is_end_of_month(ref_e))
                else:
                    ref_e = self._roll(ref_s, 1, self.end_of_month and is_end_of_month(ref_s))
            out.append(
                AccrualPeriod(ad[i], ad[i + 1], ud[i], ud[i + 1], ref_s, ref_e, self._regular[i])
            )
        return out

    @property
    def _front_stub(self) -> bool:
        if self.first_date is not None:
            return True
        if self.next_to_last_date is not None:
            return False
        return self.stub in ("short_front", "long_front")

    @property
    def frequency(self) -> Optional[int]:
        """Coupons per year for month-based tenors dividing 12 (else ``None``)."""
        if self.tenor is None or not self.tenor.is_month_based:
            return None
        m = self.tenor.months
        return 12 // m if m <= 12 and 12 % m == 0 else None

    def year_fractions(self, day_count: "str | DayCount") -> list[float]:
        """Accrual fractions of all periods."""
        return [p.year_fraction(day_count, self.frequency) for p in self.periods()]

    @property
    def is_regular(self) -> list[bool]:
        return list(self._regular)

    def __len__(self) -> int:
        return len(self.dates)

    def __iter__(self) -> Iterator[_dt.date]:
        return iter(self.dates)

    def __getitem__(self, i: Any) -> Any:
        return self.dates[i]

    def __repr__(self) -> str:
        return (
            f"Schedule({self.effective} -> {self.termination}, {self.tenor}, "
            f"{self.calendar.name}, {self.convention}, {len(self.dates) - 1} periods)"
        )
