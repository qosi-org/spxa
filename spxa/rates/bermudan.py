r"""
Bermudan swaptions.

A Bermudan swaption gives the right, on each of a set of exercise dates
:math:`t_1 < \dots < t_m`, to enter the remaining part of a fixed swap. The
holder who exercises at :math:`t_k` receives the swap whose coupons start on
or after the corresponding period start, so the exercise value is

.. math:: g_k(x) = \Bigl(\sum_i a_i^{(k)} P(t_k, t_i \mid x)\Bigr)^+,

with the payer-signed model cashflows :math:`a^{(k)}` of that co-terminal
swap (see :mod:`spxa.rates.model_cashflows`). Its value is

.. math:: V = \sup_{\tau \in \{t_1, \dots, t_m\}} E\bigl[D(\tau)\,g_\tau\bigr],

which lies above every co-terminal European swaption and below their sum.
:class:`BermudanSwaption` only describes the contract; the engines live in
:mod:`spxa.rates.bermudan_engines` and :mod:`spxa.rates.lsm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.cashflows import IborCoupon
from spxa.rates.model_cashflows import ModelCashflows, coupon_cashflows
from spxa.rates.swaps import Swap
from spxa.rates.swaptions import Swaption

__all__ = ["BermudanSwaption", "bermudan_from_times", "bermudan_swaption", "coterminal_swaptions"]


@dataclass(frozen=True)
class BermudanSwaption:
    """Exercise times and, for each, the payer-signed cashflows of the swap entered."""

    exercise_times: np.ndarray
    underlyings: tuple[ModelCashflows, ...]

    def __post_init__(self) -> None:
        t = np.asarray(self.exercise_times, float)
        if t.ndim != 1 or t.size == 0 or len(self.underlyings) != t.size:
            raise ValueError("need one underlying per exercise time")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("exercise times must be positive and increasing")
        for tk, cf in zip(t, self.underlyings):
            if cf.times.size and cf.times[0] < tk - 1e-12:
                raise ValueError("an underlying pays before its exercise time")

    @property
    def n_exercise(self) -> int:
        return int(np.asarray(self.exercise_times).size)

    @property
    def maturity(self) -> float:
        return float(self.exercise_times[-1])

    def underlying_value(self, k: int, model: Any, x: Any) -> Any:
        r""":math:`\sum_i a^{(k)}_i P(t_k, t_i \mid x)` for a model with ``discount_bond(t, T, x)``."""
        return self.underlyings[k].value(model, float(self.exercise_times[k]), x)

    def exercise_value(self, k: int, model: Any, x: Any) -> Any:
        return np.maximum(self.underlying_value(k, model, x), 0.0)

    def restrict(self, ks: Sequence[int]) -> "BermudanSwaption":
        """The contract with only the exercise dates ``ks`` (one index gives a European swaption)."""
        ks = sorted(ks)
        return BermudanSwaption(
            np.asarray(self.exercise_times)[ks], tuple(self.underlyings[k] for k in ks)
        )

    def underlying_pv(self, k: int, discount: YieldCurve) -> float:
        """Today's value of the swap entered at exercise ``k``."""
        return self.underlyings[k].pv(discount)


def bermudan_swaption(
    swap: Swap,
    discount: YieldCurve,
    projection: Optional[YieldCurve] = None,
    lockout: int = 1,
) -> BermudanSwaption:
    """Co-terminal Bermudan on ``swap``: exercisable on the fixing date of each fixed period start.

    ``lockout`` fixed periods are skipped first (``1`` gives the usual
    ``n``-non-call-1 structure: exercise into the swap from its second
    fixed period). The floating schedule must restart at every exercised
    fixed period start.
    """
    fixed: list[Any] = list(swap.fixed)
    floating: list[Any] = list(swap.floating)
    if not 0 <= lockout < len(fixed):
        raise ValueError("lockout must leave at least one exercise date")
    times, flows = [], []
    for c in fixed[lockout:]:
        start = c.start
        flt = [f for f in floating if f.start >= start]
        if not flt or flt[0].start != start:
            raise ValueError(f"no floating period starts on {start}")
        fix = flt[0].fixing if isinstance(flt[0], IborCoupon) else start
        t = float(discount.time(fix))
        if t <= 0:
            continue
        fx = [f for f in fixed if f.start >= start]
        times.append(t)
        flows.append(coupon_cashflows(fx, flt, swap.sign, discount, projection))
    return BermudanSwaption(np.array(times), tuple(flows))


def bermudan_from_times(
    exercise_times: Sequence[float],
    end: float,
    fixed_rate: float,
    notional: float = 1.0,
    payer: bool = True,
    frequency: int = 1,
) -> BermudanSwaption:
    r"""Single-curve Bermudan on year-fraction dates (useful for trees).

    Fixed coupons :math:`N K/f` are paid every :math:`1/f` years up to ``end``,
    and each exercise time must be a coupon date. The swap entered at
    :math:`t_k` is worth
    :math:`N\bigl(P(t_k, t_k) - P(t_k, T)\bigr) - \sum_j N K\tau_j P(t_k, t_j)`
    to the payer; the notional exchange at :math:`t_k` is a constant.
    """
    tau = 1.0 / frequency
    n = round(end * frequency)
    if abs(n * tau - end) > 1e-12:
        raise ValueError("end must be a whole number of coupon periods")
    pay = tau * np.arange(1, n + 1)
    sign = 1.0 if payer else -1.0
    under = []
    for t in exercise_times:
        if not np.any(np.abs(pay - t) < 1e-12) or t >= end:
            raise ValueError(f"exercise time {t} is not a coupon date before the end")
        p = pay[pay > t + 1e-12]
        amounts = -sign * notional * fixed_rate * tau * np.ones(p.size)
        amounts[-1] -= sign * notional
        under.append(ModelCashflows(np.r_[t, p], np.r_[sign * notional, amounts]))
    return BermudanSwaption(np.asarray(exercise_times, float), tuple(under))


def coterminal_swaptions(swap: Swap, discount: YieldCurve, lockout: int = 1) -> list[Swaption]:
    """The European swaptions into each co-terminal swap of :func:`bermudan_swaption`.

    They share the Bermudan's exercise dates and underlyings exactly, which
    makes them the natural calibration set for a one-factor model.
    """
    fixed: list[Any] = list(swap.fixed)
    floating: list[Any] = list(swap.floating)
    out = []
    for c in fixed[lockout:]:
        flt = [f for f in floating if f.start >= c.start]
        fix = flt[0].fixing if isinstance(flt[0], IborCoupon) else c.start
        if discount.time(fix) <= 0:
            continue
        sub = Swap([f for f in fixed if f.start >= c.start], flt, swap.payer)
        out.append(Swaption(sub, fix))
    return out
