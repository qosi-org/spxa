r"""Present values of legs on a multi-curve market.

Cashflows are discounted on one curve (typically the OIS curve of the
collateral currency) while floating coupons are projected from their own
index curve, so that

.. math:: \mathrm{PV} = \sum_i A_i\big(P_{\text{proj}}, \text{fixings}\big)\;P_{\text{disc}}(t^{\text{pay}}_i).

Cashflows paid on or before the valuation date are excluded. The annuity
(PV01 per unit rate) of a leg is :math:`\sum_i N_i \tau_i P_{\text{disc}}(t^{\text{pay}}_i)`.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.rates.cashflows import Cashflow, FixedCoupon, IborCoupon, OvernightCoupon
from spxa.rates.indices import FixingHistory

__all__ = ["CashflowRow", "cashflow_table", "leg_annuity", "leg_pv"]


def _valuation(curve: YieldCurve, valuation_date: Optional[DateLike]) -> _dt.date:
    if valuation_date is not None:
        return to_date(valuation_date)
    if curve.reference_date is None:
        raise ValueError("pass valuation_date or use a discount curve with a reference_date")
    return curve.reference_date


def _alive(leg: Sequence[Cashflow], v: _dt.date) -> list[Cashflow]:
    return [c for c in leg if c.payment_date > v]


def leg_pv(
    leg: Sequence[Cashflow],
    discount: YieldCurve,
    projection: Optional[YieldCurve] = None,
    fixings: Optional[FixingHistory] = None,
    valuation_date: Optional[DateLike] = None,
) -> float:
    """Present value of a leg (``projection`` defaults to ``discount``)."""
    v = _valuation(discount, valuation_date)
    proj = projection if projection is not None else discount
    return float(
        sum(
            c.amount(proj, fixings) * float(discount.discount(c.payment_date))
            for c in _alive(leg, v)
        )
    )


def leg_annuity(
    leg: Sequence[Cashflow], discount: YieldCurve, valuation_date: Optional[DateLike] = None
) -> float:
    r""":math:`\sum_i N_i\tau_i P(t^{\text{pay}}_i)` over the coupons of a leg (simple cashflows ignored)."""
    v = _valuation(discount, valuation_date)
    total = 0.0
    for c in _alive(leg, v):
        if isinstance(c, (FixedCoupon, IborCoupon, OvernightCoupon)):
            total += c.notional * c.accrual * float(discount.discount(c.payment_date))
    return total


@dataclass(frozen=True)
class CashflowRow:
    payment_date: _dt.date
    start: Optional[_dt.date]
    end: Optional[_dt.date]
    notional: Optional[float]
    accrual: Optional[float]
    rate: Optional[float]
    amount: float
    discount: float
    pv: float


def cashflow_table(
    leg: Sequence[Cashflow],
    discount: YieldCurve,
    projection: Optional[YieldCurve] = None,
    fixings: Optional[FixingHistory] = None,
    valuation_date: Optional[DateLike] = None,
) -> list[CashflowRow]:
    """Per-cashflow breakdown of :func:`leg_pv`."""
    v = _valuation(discount, valuation_date)
    proj = projection if projection is not None else discount
    rows = []
    for c in _alive(leg, v):
        amt = c.amount(proj, fixings)
        df = float(discount.discount(c.payment_date))
        start: Any = getattr(c, "start", None)
        end: Any = getattr(c, "end", None)
        notional = getattr(c, "notional", None)
        accrual = getattr(c, "accrual", None)
        rate: Optional[float]
        if isinstance(c, FixedCoupon):
            rate = c.rate
        elif isinstance(c, (IborCoupon, OvernightCoupon)):
            rate = c.rate(proj, fixings)
        else:
            rate = None
        rows.append(
            CashflowRow(c.payment_date, start, end, notional, accrual, rate, amt, df, amt * df)
        )
    return rows
