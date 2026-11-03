r"""Discount curves and equity forwards.

Times are year fractions from the curve's reference date. Every method that
takes a time also accepts a date when the curve has a ``reference_date``;
the date is converted with the curve's day count (ACT/365F by default).

Compounding conventions for quoted rates :math:`r` over :math:`t` years:

===================  ==============================================
``"continuous"``     :math:`D = e^{-rt}`
``"simple"``         :math:`D = 1/(1 + rt)`
``"annual"``         :math:`D = (1 + r)^{-t}`
integer :math:`n`    :math:`D = (1 + r/n)^{-nt}` (e.g. 2 = semi-annual)
===================  ==============================================

The equity forward under deterministic rates, a continuous dividend yield
with factor :math:`Q(t)`, cash dividends :math:`d_i` and proportional
dividends :math:`\delta_j` (escrowed-dividend model) is

.. math::
    F(t) = \frac{Q(t)}{P_f(t)}\Bigl(S_0\prod_{j}(1 - \delta_j)
    - \sum_{i} d_i\,\frac{P_f(t_i)}{Q(t_i)}\prod_{j:\,t_j > t_i}(1 - \delta_j)\Bigr),

with all sums and products over dividends paid in :math:`(0, t]` and
:math:`P_f` the funding (repo) discount curve.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount

__all__ = [
    "Compounding",
    "Dividend",
    "EquityForward",
    "FlatCurve",
    "InterpolatedDiscountCurve",
    "YieldCurve",
    "discount_from_rate",
    "rate_from_discount",
]

Compounding = Union[str, int]
TimeLike = Union[float, np.ndarray, Sequence[float], DateLike]


def _comp(c: Compounding) -> Union[str, int]:
    if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
        if c <= 0:
            raise ValueError("compounding frequency must be positive")
        return int(c)
    key = str(c).lower()
    aliases = {
        "continuous": "continuous",
        "cont": "continuous",
        "simple": "simple",
        "annual": 1,
        "semiannual": 2,
        "semi-annual": 2,
        "quarterly": 4,
        "monthly": 12,
    }
    if key not in aliases:
        raise ValueError(f"unknown compounding {c!r}")
    return aliases[key]


def discount_from_rate(rate: Any, t: Any, compounding: Compounding = "continuous") -> Any:
    """Discount factor from a rate quoted with the given compounding."""
    c = _comp(compounding)
    r, t = np.asarray(rate, dtype=float), np.asarray(t, dtype=float)
    if c == "continuous":
        out = np.exp(-r * t)
    elif c == "simple":
        out = 1.0 / (1.0 + r * t)
    else:
        out = (1.0 + r / c) ** (-c * t)
    return float(out) if out.ndim == 0 else out


def rate_from_discount(df: Any, t: Any, compounding: Compounding = "continuous") -> Any:
    """Rate with the given compounding implied by a discount factor (``t > 0``)."""
    c = _comp(compounding)
    df, t = np.asarray(df, dtype=float), np.asarray(t, dtype=float)
    if np.any(t <= 0):
        raise ValueError("rate_from_discount needs t > 0")
    if c == "continuous":
        out = -np.log(df) / t
    elif c == "simple":
        out = (1.0 / df - 1.0) / t
    else:
        out = c * (df ** (-1.0 / (c * t)) - 1.0)
    return float(out) if out.ndim == 0 else out


class YieldCurve(abc.ABC):
    """Base class for deterministic discount curves.

    Subclasses implement :meth:`_discount` on year fractions. Everything
    else (zero and forward rates, date handling) is derived.
    """

    def __init__(
        self, reference_date: Optional[DateLike] = None, day_count: "str | DayCount" = "ACT/365F"
    ) -> None:
        self.reference_date = None if reference_date is None else to_date(reference_date)
        self.day_count = DayCount(day_count)

    @abc.abstractmethod
    def _discount(self, t: np.ndarray) -> np.ndarray:
        """Discount factors at year fractions ``t`` (array)."""

    # -- time handling --------------------------------------------------
    def time(self, x: TimeLike) -> Any:
        """Year fraction(s) of ``x`` (numbers pass through, dates are converted)."""
        if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool):
            return float(x)
        if isinstance(x, np.ndarray) and x.dtype.kind in "fiu":
            return x.astype(float)
        if isinstance(x, (list, tuple)):
            return np.array([self.time(v) for v in x], dtype=float)
        if self.reference_date is None:
            raise ValueError("curve has no reference_date; pass times in years")
        return self.day_count.year_fraction(self.reference_date, to_date(x))

    def _eval(self, t: TimeLike) -> tuple[np.ndarray, bool]:
        tt = np.asarray(self.time(t), dtype=float)
        return tt, tt.ndim == 0

    # -- public API -----------------------------------------------------
    def discount(self, t: TimeLike) -> Any:
        """Discount factor :math:`P(0, t)`."""
        tt, scalar = self._eval(t)
        out = self._discount(np.atleast_1d(tt))
        return float(out[0]) if scalar else out.reshape(tt.shape)

    def zero_rate(self, t: TimeLike, compounding: Compounding = "continuous") -> Any:
        """Zero rate to ``t`` with the given compounding (``t > 0``)."""
        tt, _ = self._eval(t)
        return rate_from_discount(self.discount(tt), tt, compounding)

    def forward_rate(self, t1: TimeLike, t2: TimeLike, compounding: Compounding = "simple") -> Any:
        """Forward rate between ``t1 < t2``: :math:`P(t_1)/P(t_2)` expressed with ``compounding``."""
        a, _ = self._eval(t1)
        b, _ = self._eval(t2)
        if np.any(b <= a):
            raise ValueError("forward_rate needs t2 > t1")
        return rate_from_discount(self.discount(b) / self.discount(a), b - a, compounding)

    def instantaneous_forward(self, t: TimeLike, h: float = 1e-5) -> Any:
        r""":math:`f(t) = -\partial_t \log P(0, t)` (central difference unless overridden)."""
        tt, scalar = self._eval(t)
        lo = np.maximum(tt - h, 0.0)
        hi = tt + h
        out = -(np.log(self.discount(hi)) - np.log(self.discount(lo))) / (hi - lo)
        return float(out) if scalar else out

    def shifted(self, spread: float) -> "YieldCurve":
        """Curve with a parallel continuously compounded zero-rate shift."""
        return _ShiftedCurve(self, spread)


class _ShiftedCurve(YieldCurve):
    def __init__(self, base: YieldCurve, spread: float) -> None:
        super().__init__(base.reference_date, base.day_count)
        self.base, self.spread = base, float(spread)

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return self.base._discount(t) * np.exp(-self.spread * t)


class FlatCurve(YieldCurve):
    """Constant-rate curve.

    Parameters
    ----------
    rate : float
        The rate, quoted with ``compounding``.
    compounding : str or int
        Default continuous.
    """

    def __init__(
        self,
        rate: float,
        compounding: Compounding = "continuous",
        reference_date: Optional[DateLike] = None,
        day_count: "str | DayCount" = "ACT/365F",
    ) -> None:
        super().__init__(reference_date, day_count)
        self.rate = float(rate)
        self.compounding = _comp(compounding)

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(discount_from_rate(self.rate, t, self.compounding), dtype=float)

    def __repr__(self) -> str:
        return f"FlatCurve({self.rate}, {self.compounding!r})"


class InterpolatedDiscountCurve(YieldCurve):
    r"""Curve through discount-factor nodes.

    Interpolation is linear in :math:`\log P` (piecewise-constant
    instantaneous forwards). Beyond the last node the last forward is
    extrapolated flat. A node at :math:`t = 0` with :math:`P = 1` is added
    if missing.
    """

    def __init__(
        self,
        times: Sequence[float],
        discounts: Sequence[float],
        reference_date: Optional[DateLike] = None,
        day_count: "str | DayCount" = "ACT/365F",
    ) -> None:
        super().__init__(reference_date, day_count)
        t = np.asarray(self.time(list(times)), dtype=float)
        p = np.asarray(discounts, dtype=float)
        if t.shape != p.shape or t.ndim != 1 or t.size == 0:
            raise ValueError("times and discounts must be 1-D arrays of equal length")
        if np.any(np.diff(t) <= 0) or t[0] < 0:
            raise ValueError("times must be non-negative and strictly increasing")
        if np.any(p <= 0):
            raise ValueError("discount factors must be positive")
        if t[0] > 0:
            t, p = np.concatenate([[0.0], t]), np.concatenate([[1.0], p])
        elif not math.isclose(p[0], 1.0, abs_tol=1e-14):
            raise ValueError("discount factor at t = 0 must be 1")
        self.times, self.discounts = t, p
        self._logp = np.log(p)

    def _discount(self, t: np.ndarray) -> np.ndarray:
        if np.any(t < 0):
            raise ValueError("negative time")
        if self.times.size == 1:
            return np.ones_like(t)
        logp = np.interp(t, self.times, self._logp)
        beyond = t > self.times[-1]
        if np.any(beyond):
            f_last = -(self._logp[-1] - self._logp[-2]) / (self.times[-1] - self.times[-2])
            logp[beyond] = self._logp[-1] - f_last * (t[beyond] - self.times[-1])
        return np.exp(logp)


@dataclass(frozen=True)
class Dividend:
    """A discrete dividend: ``amount`` paid at ``time`` (years or a date).

    ``kind="cash"`` pays a fixed amount; ``kind="proportional"`` pays
    ``amount`` times the pre-dividend price (``0 <= amount < 1``).
    """

    time: Any
    amount: float
    kind: str = "cash"

    def __post_init__(self) -> None:
        if self.kind not in ("cash", "proportional"):
            raise ValueError("dividend kind must be 'cash' or 'proportional'")
        if self.kind == "proportional" and not 0.0 <= self.amount < 1.0:
            raise ValueError("proportional dividends must lie in [0, 1)")
        if self.kind == "cash" and self.amount < 0:
            raise ValueError("cash dividends must be non-negative")


class EquityForward:
    """Forward price of a stock with dividends and repo.

    Parameters
    ----------
    spot : float
    discount : YieldCurve or float
        Risk-free discount curve (a float is a flat continuous rate).
    dividend_yield : YieldCurve or float, optional
        Continuous dividend yield, as a curve of "discount factors" :math:`Q(t)`.
    repo : YieldCurve or float, optional
        Funding curve used to carry the stock; defaults to ``discount``.
    dividends : sequence of Dividend, optional
        Discrete dividends (escrowed model).
    """

    def __init__(
        self,
        spot: float,
        discount: "YieldCurve | float",
        dividend_yield: "YieldCurve | float | None" = None,
        repo: "YieldCurve | float | None" = None,
        dividends: Sequence[Dividend] = (),
    ) -> None:
        if spot <= 0:
            raise ValueError("spot must be positive")
        self.spot = float(spot)
        self.discount_curve = _as_curve(discount)
        self.dividend_curve = None if dividend_yield is None else _as_curve(dividend_yield)
        self.repo_curve = self.discount_curve if repo is None else _as_curve(repo)
        times = [float(self.discount_curve.time(d.time)) for d in dividends]
        order = sorted(range(len(times)), key=times.__getitem__)
        self.dividends = [dividends[i] for i in order]
        self._div_times = np.array([times[i] for i in order], dtype=float)

    def _escrowed_spot(self, t: float) -> float:
        r"""Time-0 value of one share delivered at ``t`` net of the dividends before it.

        :math:`S\prod_j(1-\delta_j) - \sum_i d_i \frac{P_f(t_i)}{Q(t_i)}
        \prod_{j:\,t_j > t_i}(1-\delta_j)`, products over dividends in
        :math:`(0, t]`: a cash dividend is reduced only by the proportional
        dividends paid after it.
        """
        live = [(d, td) for d, td in zip(self.dividends, self._div_times) if 0 < td <= t]
        value = self.spot
        for d, td in live:
            if d.kind == "cash":
                value -= d.amount * float(self.repo_curve.discount(td)) / self._q(td)
            else:
                value *= 1.0 - d.amount
        return value

    def _q(self, t: float) -> float:
        return 1.0 if self.dividend_curve is None else float(self.dividend_curve.discount(t))

    def forward(self, t: TimeLike) -> Any:
        """Forward price for delivery at ``t``."""
        tt = np.asarray(self.discount_curve.time(t), dtype=float)
        vals = np.array(
            [
                self._escrowed_spot(x) * self._q(x) / float(self.repo_curve.discount(x))
                for x in np.atleast_1d(tt)
            ]
        )
        if np.any(vals <= 0):
            raise ValueError("dividends exceed the spot price: forward is not positive")
        return float(vals[0]) if tt.ndim == 0 else vals.reshape(tt.shape)

    def discount(self, t: TimeLike) -> Any:
        """Risk-free discount factor (for option premiums)."""
        return self.discount_curve.discount(t)

    def dividend_pv(self, t: TimeLike) -> float:
        """Value of the cash dividends paid in ``(0, t]``, discounted on the funding
        curve and divided by the dividend-yield factor (the escrowed amount)."""
        tt = float(self.discount_curve.time(t))
        return float(
            sum(
                d.amount * self.repo_curve.discount(td) / self._q(td)
                for d, td in zip(self.dividends, self._div_times)
                if d.kind == "cash" and 0 < td <= tt
            )
        )


def _as_curve(x: "YieldCurve | float") -> YieldCurve:
    return x if isinstance(x, YieldCurve) else FlatCurve(float(x))
