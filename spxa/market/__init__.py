r"""
spxa.market — market conventions and building blocks shared by
:mod:`spxa.rates` and :mod:`spxa.vol`.

Dates and tenors
    :func:`to_date`, :func:`add_months`, :func:`add_tenor`, :class:`Tenor`.
Day counts
    :class:`DayCount`, :func:`year_fraction`, :func:`day_count` (ACT/360,
    ACT/365F, ACT/ACT ISDA and ICMA, the 30/360 family, BUS/252).
Calendars
    :class:`Calendar`, :class:`JointCalendar`, :func:`get_calendar`
    (TARGET, US-FED, US-SIFMA/SOFR, UK, weekends), :func:`easter_sunday`.
Schedules
    :func:`adjust` (F, MF, P, MP, U), :class:`Schedule` with front/back,
    short/long and explicit stubs, the end-of-month rule and accrual periods.
Curves and forwards
    :class:`YieldCurve`, :class:`FlatCurve`, :class:`InterpolatedDiscountCurve`,
    compounding conversions, :class:`EquityForward` with dividend yields,
    repo and discrete (cash or proportional) dividends.
Option formulas
    Black, Bachelier and shifted Black prices and Greeks; Black–Scholes on spot.
Implied volatility
    :func:`implied_black_volatility` to machine precision (normalised
    Jäckel formulation, cancellation-free tails).
"""

from spxa.market.dates import (
    DateLike,
    Tenor,
    add_months,
    add_tenor,
    days_between,
    days_in_month,
    days_in_year,
    end_of_month,
    is_end_of_month,
    is_leap_year,
    to_date,
)
from spxa.market.black import (
    bachelier_greeks,
    bachelier_price,
    black_greeks,
    black_price,
    black_scholes_greeks,
    black_scholes_price,
    shifted_black_greeks,
    shifted_black_price,
)
from spxa.market.calendars import (
    Calendar,
    JointCalendar,
    available_calendars,
    easter_sunday,
    get_calendar,
)
from spxa.market.curves import (
    Compounding,
    Dividend,
    EquityForward,
    FlatCurve,
    InterpolatedDiscountCurve,
    YieldCurve,
    discount_from_rate,
    rate_from_discount,
)
from spxa.market.daycount import DayCount, available_day_counts, day_count, year_fraction
from spxa.market.implied import implied_black_volatility, normalised_black_call
from spxa.market.schedule import (
    AccrualPeriod,
    Schedule,
    adjust,
    available_business_day_conventions,
)

__all__ = [
    # dates
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
    # day counts
    "DayCount",
    "available_day_counts",
    "day_count",
    "year_fraction",
    # calendars
    "Calendar",
    "JointCalendar",
    "available_calendars",
    "easter_sunday",
    "get_calendar",
    # schedules
    "AccrualPeriod",
    "Schedule",
    "adjust",
    "available_business_day_conventions",
    # curves
    "Compounding",
    "Dividend",
    "EquityForward",
    "FlatCurve",
    "InterpolatedDiscountCurve",
    "YieldCurve",
    "discount_from_rate",
    "rate_from_discount",
    # option formulas
    "bachelier_greeks",
    "bachelier_price",
    "black_greeks",
    "black_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "shifted_black_greeks",
    "shifted_black_price",
    # implied volatility
    "implied_black_volatility",
    "normalised_black_call",
]
