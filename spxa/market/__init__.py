r"""
spxa.market — market conventions and building blocks shared by
:mod:`spxa.rates` and :mod:`spxa.vol`.

Dates and tenors
    :func:`to_date`, :func:`add_months`, :func:`add_tenor`, :class:`Tenor`.
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
]
