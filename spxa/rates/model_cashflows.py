r"""Swaps as cashflows on the discount curve, for short-rate and LMM pricing.

A one-factor model on the discount curve sees floating coupons through the
deterministic-basis approximation. A coupon fixing over :math:`[s, e]` and
paid at :math:`p` is written as

.. math:: N\rho\,P(t, s) - N\,P(t, p) + c\,P(t, p), \qquad \rho = \frac{P(0, p)}{P(0, e)},

where the residual amount :math:`c` is chosen so that the coupon's value
today is reproduced exactly:
:math:`c = V_0/P(0, p) - N\bigl(P(0, s)/P(0, e) - 1\bigr)`. It collects the
spread between the projection and discount forwards, any quoted spread and
gearing, and the gap between index and accrual periods. For a single curve
with :math:`p = e` it is zero, and the floating leg telescopes to
:math:`N(P(t, t_0) - P(t, t_n))`. Fixed coupons are cashflows already.

The result is a list of times and amounts with
:math:`\sum_i a_i P(0, t_i)` equal to the swap's present value, and
:math:`\sum_i a_i P(t, t_i)` its value at any time before the first
fixing under the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.cashflows import FixedCoupon, IborCoupon, OvernightCoupon
from spxa.rates.pricing import leg_pv
from spxa.rates.swaps import Swap

__all__ = ["ModelCashflows", "swap_cashflows"]


@dataclass(frozen=True)
class ModelCashflows:
    """Amounts ``amounts[i]`` paid at year fractions ``times[i]`` (sorted, merged)."""

    times: np.ndarray
    amounts: np.ndarray

    def pv(self, discount: YieldCurve) -> float:
        return float(np.dot(self.amounts, discount.discount(self.times)))

    def value(self, model: Any, t: float, x: Any) -> Any:
        """``sum a_i P(t, t_i | x)`` for a model with ``discount_bond(t, T, x)``."""
        P = np.asarray(model.discount_bond(t, self.times, np.asarray(x, float)[..., None]), float)
        return P @ self.amounts


def swap_cashflows(
    swap: Swap, discount: YieldCurve, projection: Optional[YieldCurve] = None
) -> ModelCashflows:
    """Payer-signed cashflows of ``swap`` (``-`` fixed, ``+`` floating for a payer swap)."""
    proj = projection if projection is not None else discount
    flows: dict[float, float] = {}

    def add(date: Any, amount: float) -> None:
        t = float(discount.time(date))
        flows[t] = flows.get(t, 0.0) + amount

    sign = swap.sign
    for c in swap.fixed:
        assert isinstance(c, FixedCoupon)
        add(c.payment_date, -sign * c.amount())
    for c in swap.floating:
        if not isinstance(c, (IborCoupon, OvernightCoupon)):
            raise TypeError(f"unsupported floating cashflow {type(c).__name__}")
        N = c.notional
        P_s, P_e, P_p = (float(discount.discount(d)) for d in (c.start, c.end, c.payment_date))
        v0 = leg_pv([c], discount, proj)
        add(c.start, sign * N * P_p / P_e)
        add(c.payment_date, sign * (v0 / P_p - N * P_s / P_e))
    times = np.array(sorted(flows))
    amounts = np.array([flows[t] for t in times])
    keep = amounts != 0.0
    return ModelCashflows(times[keep], amounts[keep])
