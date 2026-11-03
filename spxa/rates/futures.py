r"""IMM dates and interest-rate futures with Hull–White convexity.

IMM dates are the third Wednesdays of March, June, September and December
(serial months on request); codes are ``F G H J K M N Q U V X Z`` plus the
year, e.g. ``H24``.

Futures on a term rate
    Daily margining makes the futures rate the risk-neutral expectation of
    the simple rate fixed for :math:`[t_1, t_2]`,
    :math:`R^{\mathrm{fut}} = E^Q[L(t_1; t_1, t_2)]`, which exceeds the
    forward :math:`F = E^{T_2}[L]`. In the Hull–White model
    :math:`dr = (\vartheta(t) - a r)\,dt + \sigma\,dW` (Brigo & Mercurio 2006, §3.3)

    .. math:: 1 + \tau R^{\mathrm{fut}} = (1 + \tau F)\,e^{\Gamma}, \qquad
              \Gamma = \frac{\sigma^2 B_{12}}{2a}\Bigl[B_{12}\bigl(1 - e^{-2at_1}\bigr) + aB_{01}^2\Bigr],

    with :math:`B_{ij} = (1 - e^{-a(t_j - t_i)})/a`; for :math:`a \to 0`
    (Ho–Lee) :math:`\Gamma \to \sigma^2\tau t_1(t_1 + 2\tau)/2`. The
    familiar adjustment of continuously compounded rates (Hull, *Options,
    Futures and Other Derivatives*),
    :math:`\frac{B_{12}}{\tau}\bigl[B_{12}(1 - e^{-2at_1}) + 2aB_{01}^2\bigr]\frac{\sigma^2}{4a}`,
    which tends to :math:`\sigma^2 t_1 t_2/2`, is :func:`hull_white_rate_convexity`.

Futures on an overnight rate
    A three-month SOFR future settles on the rate compounded over its IMM
    reference quarter, :math:`1 + \tau R = \prod_i (1 + r_i\tau_i)`; the
    one-month contract on the arithmetic average. Define each daily rate by
    :math:`1 + r_i\tau_i = 1/P(t_i, t_{i+1})`. Under the :math:`t_N`-forward
    measure the product telescopes, so
    :math:`E^{T_N}[1 + \tau R] = P(0, t_0)/P(0, t_N) = 1 + \tau F`. In any
    Gaussian short-rate model :math:`Y = \log\prod_i 1/P(t_i, t_{i+1})` is
    Gaussian, and changing measure gives
    :math:`E^Q[e^Y] = E^{T_N}[e^Y]\,e^{\mathrm{Cov}(Y, \int_0^{t_N} r)}`. In
    Hull–White :math:`\log 1/P(t_i, t_{i+1}) = B(\delta_i)\,x(t_i)` plus a
    constant, with :math:`\delta_i = t_{i+1} - t_i`, and

    .. math:: c(t; T) = \mathrm{Cov}\Bigl(x(t), \int_0^T x\Bigr)
              = \sigma^2\Bigl[\tfrac12 B(t)^2 + B(T - t)\,B_2(t)\Bigr],
              \qquad B_2(t) = \frac{1 - e^{-2at}}{2a},

    so that, exactly for the daily product,

    .. math:: 1 + \tau R^{\mathrm{fut}} = (1 + \tau F)\,e^{\Gamma_c}, \qquad
              \Gamma_c = \sum_{t_i > 0} B(\delta_i)\,c(t_i; t_N)

    (:func:`hull_white_discrete_compounded_convexity`). Fixings already
    observed (:math:`t_i \le 0`) do not contribute. The continuous
    compounding limit is
    :math:`\Gamma_c = \mathrm{Var}(I) + \mathrm{Cov}(\int_0^{T_s} r, I)` with
    :math:`I = \int_{T_s}^{T_e} r`, that is

    .. math:: \Gamma_c = \sigma^2\Bigl[B(\tau)^2 B_2(T_s) + J(\tau)
              + \tfrac12 B(\tau) B(T_s)^2\Bigr], \qquad
              J(\tau) = \int_0^\tau B(v)^2\,dv,

    which for Ho–Lee is :math:`\sigma^2(\tau^2 T_s + \tau^3/3 + \tau T_s^2/2)`
    (:func:`hull_white_compounded_convexity`). Unlike a term-rate future the
    adjustment does not vanish at :math:`T_s = 0`, since the rate is still
    being fixed. For the averaged contract each day is adjusted separately:
    :math:`E^Q[r_i]\tau_i = (1 + F_i\tau_i)\,e^{B(\delta_i)c(t_i; t_{i+1})} - 1`,
    where :math:`[t_i, t_{i+1}]` runs from the fixing date to the next one.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Optional, Sequence


from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.rates.indices import SOFR, FixingHistory, IborIndex, OvernightIndex
from spxa.rates.overnight import (
    compounded_overnight_rate,
    curve_projection,
    observation_schedule,
    projected_overnight_rate,
)

__all__ = [
    "OvernightFuture",
    "RateFuture",
    "hull_white_compounded_convexity",
    "hull_white_compounding_covariance",
    "hull_white_discrete_compounded_convexity",
    "hull_white_futures_convexity",
    "hull_white_rate_convexity",
    "imm_code",
    "imm_date",
    "imm_from_code",
    "is_imm_date",
    "next_imm_date",
]

_CODES = "FGHJKMNQUVXZ"
_QUARTERLY = (3, 6, 9, 12)


def imm_date(year: int, month: int) -> _dt.date:
    """Third Wednesday of the month."""
    first = _dt.date(year, month, 1)
    return first + _dt.timedelta(days=(2 - first.weekday()) % 7 + 14)


def is_imm_date(d: DateLike, quarterly: bool = True) -> bool:
    d = to_date(d)
    return d == imm_date(d.year, d.month) and (not quarterly or d.month in _QUARTERLY)


def next_imm_date(d: DateLike, quarterly: bool = True, include_self: bool = False) -> _dt.date:
    """First IMM date after ``d`` (or on it with ``include_self``)."""
    d = to_date(d)
    y, m = d.year, d.month
    while True:
        if not quarterly or m in _QUARTERLY:
            c = imm_date(y, m)
            if c > d or (include_self and c == d):
                return c
        m += 1
        if m == 13:
            y, m = y + 1, 1


def imm_code(d: DateLike) -> str:
    """Contract code of an IMM date, e.g. ``"H24"`` for 20 March 2024."""
    d = to_date(d)
    if d != imm_date(d.year, d.month):
        raise ValueError(f"{d} is not an IMM date")
    return f"{_CODES[d.month - 1]}{d.year % 100:02d}"


def imm_from_code(code: str, reference: DateLike) -> _dt.date:
    """IMM date of a code such as ``"Z4"`` or ``"H25"``: the first one on or after ``reference``."""
    code = code.strip().upper()
    if len(code) < 2 or code[0] not in _CODES or not code[1:].isdigit():
        raise ValueError(f"bad IMM code {code!r}")
    month = _CODES.index(code[0]) + 1
    digits = code[1:]
    ref = to_date(reference)
    mod = 10 ** len(digits)
    year = ref.year - ref.year % mod + int(digits)
    while imm_date(year, month) < ref:
        year += mod
    return imm_date(year, month)


def _B(a: float, t: float) -> float:
    return t if a == 0.0 else -math.expm1(-a * t) / a


def hull_white_futures_convexity(t1: float, t2: float, a: float, sigma: float) -> float:
    r""":math:`\Gamma` with :math:`1 + \tau R^{\mathrm{fut}} = (1 + \tau F)e^\Gamma` (times in years)."""
    if not 0.0 <= t1 < t2:
        raise ValueError("need 0 <= t1 < t2")
    if a < 0 or sigma < 0:
        raise ValueError("a and sigma must be non-negative")
    b12, b01 = _B(a, t2 - t1), _B(a, t1)
    v1 = t1 if a == 0.0 else -math.expm1(-2.0 * a * t1) / (2.0 * a)  # (1 - e^{-2 a t1}) / (2a)
    return sigma * sigma * b12 * (b12 * v1 + 0.5 * b01 * b01)


def hull_white_rate_convexity(t1: float, t2: float, a: float, sigma: float) -> float:
    """Hull's futures-minus-forward adjustment for continuously compounded rates."""
    if not 0.0 <= t1 < t2:
        raise ValueError("need 0 <= t1 < t2")
    b12, b01 = _B(a, t2 - t1), _B(a, t1)
    v1 = t1 if a == 0.0 else -math.expm1(-2.0 * a * t1) / (2.0 * a)
    return sigma * sigma * b12 * (0.5 * b12 * v1 + 0.5 * b01 * b01) / (t2 - t1)


@dataclass(frozen=True)
class RateFuture:
    """Futures contract on a term rate (EURIBOR 3M style), quoted ``100 (1 - R)``.

    ``convexity`` is either a fixed rate adjustment (``futures - forward``)
    or ``None`` to use the Hull–White parameters passed to the pricing
    methods.
    """

    start: _dt.date
    end: _dt.date
    price: float
    index: Optional[IborIndex] = None
    day_count: str = "ACT/360"
    convexity: Optional[float] = None

    @classmethod
    def from_imm(
        cls, index: IborIndex, imm: DateLike, price: float, convexity: Optional[float] = None
    ) -> "RateFuture":
        start = to_date(imm)
        return cls(start, index.maturity_date(start), price, index, index.day_count, convexity)

    @property
    def rate(self) -> float:
        return 1.0 - self.price / 100.0

    @property
    def accrual(self) -> float:
        dc = self.index.dc if self.index is not None else DayCount(self.day_count)
        return dc.year_fraction(self.start, self.end)

    @property
    def maturity(self) -> _dt.date:
        return self.end

    def forward_rate(
        self, projection: YieldCurve, fixings: Optional[FixingHistory] = None
    ) -> float:
        if self.index is not None and fixings is not None:
            f = fixings.get(self.index.fixing_date(self.start))
            if f is not None:
                return float(f)
        return (
            float(projection.discount(self.start)) / float(projection.discount(self.end)) - 1.0
        ) / self.accrual

    def convexity_adjustment(
        self, projection: YieldCurve, a: float = 0.0, sigma: float = 0.0
    ) -> float:
        """``futures rate - forward rate``."""
        if self.convexity is not None:
            return self.convexity
        t1 = float(projection.time(self.start))
        t2 = float(projection.time(self.end))
        if t1 <= 0:
            return 0.0
        F = self.forward_rate(projection)
        tau = self.accrual
        g = hull_white_futures_convexity(t1, t2, a, sigma)
        return ((1.0 + tau * F) * math.expm1(g)) / tau

    def implied_rate(self, projection: YieldCurve, a: float = 0.0, sigma: float = 0.0) -> float:
        """Model futures rate: forward plus convexity."""
        return self.forward_rate(projection) + self.convexity_adjustment(projection, a, sigma)

    def implied_price(self, projection: YieldCurve, a: float = 0.0, sigma: float = 0.0) -> float:
        return 100.0 * (1.0 - self.implied_rate(projection, a, sigma))

    def convexity_adjusted_forward(
        self,
        a: float = 0.0,
        sigma: float = 0.0,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
    ) -> float:
        """Forward rate implied by the quoted price (for curve building)."""
        if self.convexity is not None:
            return self.rate - self.convexity
        if t1 is None or t2 is None:
            raise ValueError("pass t1 and t2 (year fractions) for a Hull-White adjustment")
        tau = self.accrual
        g = hull_white_futures_convexity(t1, t2, a, sigma) if t1 > 0 else 0.0
        return ((1.0 + tau * self.rate) * math.exp(-g) - 1.0) / tau


# ----------------------------------------------------------------------
# Overnight-rate futures (SOFR 1M / 3M style)
# ----------------------------------------------------------------------


def _B2(a: float, t: float) -> float:
    return t if a == 0.0 else -math.expm1(-2.0 * a * t) / (2.0 * a)


def _J(a: float, tau: float) -> float:
    r""":math:`\int_0^\tau B(v)^2\,dv = (\tau - 2B(\tau) + B_2(\tau))/a^2`.

    For :math:`a\tau < 0.1` the cancelling closed form is replaced by its
    series :math:`\tau^3\sum_{n\ge3}(-1)^{n+1}(2^{n-1}-2)(a\tau)^{n-3}/n!`.
    """
    x = a * tau
    if x < 0.1:
        total, fact = 0.0, 6.0
        for n in range(3, 22):
            if n > 3:
                fact *= n
            total += (-1) ** (n + 1) * (2.0 ** (n - 1) - 2.0) * x ** (n - 3) / fact
        return tau**3 * total
    return (tau - 2.0 * _B(a, tau) + _B2(a, tau)) / (a * a)


def hull_white_compounding_covariance(t: float, T: float, a: float, sigma: float) -> float:
    r""":math:`c(t; T) = \mathrm{Cov}\bigl(x(t), \int_0^T x\,du\bigr)` in Hull–White, :math:`0 \le t \le T`."""
    if not 0.0 <= t <= T:
        raise ValueError("need 0 <= t <= T")
    if a < 0 or sigma < 0:
        raise ValueError("a and sigma must be non-negative")
    b = _B(a, t)
    return sigma * sigma * (0.5 * b * b + _B(a, T - t) * _B2(a, t))


def hull_white_compounded_convexity(ts: float, te: float, a: float, sigma: float) -> float:
    r""":math:`\Gamma_c` of a continuously compounded overnight future on :math:`[T_s, T_e]`."""
    if not 0.0 <= ts < te:
        raise ValueError("need 0 <= ts < te")
    if a < 0 or sigma < 0:
        raise ValueError("a and sigma must be non-negative")
    tau = te - ts
    b, bs = _B(a, tau), _B(a, ts)
    return sigma * sigma * (b * b * _B2(a, ts) + _J(a, tau) + 0.5 * b * bs * bs)


def hull_white_discrete_compounded_convexity(
    times: Sequence[float], a: float, sigma: float
) -> float:
    r""":math:`\Gamma_c = \sum_{t_i > 0} B(t_{i+1} - t_i)\,c(t_i; t_N)`.

    ``times`` are the model times :math:`t_0 < \dots < t_N` of the daily
    sub-period boundaries; sub-periods starting at or before zero are
    already fixed and contribute nothing.
    """
    t = [float(x) for x in times]
    if len(t) < 2 or any(y <= x for x, y in zip(t, t[1:])):
        raise ValueError("times must be strictly increasing with at least two entries")
    tn = t[-1]
    return sum(
        _B(a, t1 - t0) * hull_white_compounding_covariance(t0, tn, a, sigma)
        for t0, t1 in zip(t, t[1:])
        if t0 > 0
    )


@dataclass(frozen=True)
class OvernightFuture:
    """Futures contract on a compounded (3M SOFR) or averaged (1M SOFR) overnight rate.

    The final settlement price is ``100 (1 - R)`` with ``R`` the rate over
    ``[start, end)``. ``convexity`` is a fixed ``futures - forward``
    adjustment, or ``None`` for the exact Hull–White adjustment of the
    module docstring.
    """

    start: _dt.date
    end: _dt.date
    price: float
    index: OvernightIndex = SOFR
    method: str = "compounded"
    convexity: Optional[float] = None

    def __post_init__(self) -> None:
        if self.method not in ("compounded", "averaged"):
            raise ValueError("method must be 'compounded' or 'averaged'")
        if self.end <= self.start:
            raise ValueError("end must be after start")

    @classmethod
    def from_imm(
        cls,
        imm: DateLike,
        price: float,
        index: OvernightIndex = SOFR,
        convexity: Optional[float] = None,
    ) -> "OvernightFuture":
        """Three-month contract on the quarter from an IMM date to the next one."""
        start = to_date(imm)
        if not is_imm_date(start):
            raise ValueError(f"{start} is not a quarterly IMM date")
        return cls(start, next_imm_date(start), price, index, "compounded", convexity)

    @classmethod
    def monthly(
        cls,
        year: int,
        month: int,
        price: float,
        index: OvernightIndex = SOFR,
        convexity: Optional[float] = None,
    ) -> "OvernightFuture":
        """One-month contract on the arithmetic average over a calendar month."""
        start = _dt.date(year, month, 1)
        end = _dt.date(year + month // 12, month % 12 + 1, 1)
        return cls(start, end, price, index, "averaged", convexity)

    @property
    def rate(self) -> float:
        return 1.0 - self.price / 100.0

    @property
    def accrual(self) -> float:
        return self.index.dc.year_fraction(self.start, self.end)

    @property
    def maturity(self) -> _dt.date:
        return self.end

    def forward_rate(
        self, projection: YieldCurve, fixings: Optional[FixingHistory] = None
    ) -> float:
        """Rate over the period from published and curve-projected fixings (no convexity)."""
        if self.method == "compounded":
            return projected_overnight_rate(self.index, self.start, self.end, projection, fixings)
        return compounded_overnight_rate(
            self.index,
            self.start,
            self.end,
            fixings,
            curve_projection(self.index, projection),
            method="averaged",
        ).rate

    def convexity_adjustment(
        self,
        projection: YieldCurve,
        a: float = 0.0,
        sigma: float = 0.0,
        fixings: Optional[FixingHistory] = None,
    ) -> float:
        """``futures rate - forward rate``, exact in Hull–White unless ``convexity`` is set."""
        if self.convexity is not None:
            return self.convexity
        acc, obs, weights, _ = observation_schedule(self.index, self.start, self.end)
        bounds = [*acc[1:], self.end]
        getter = fixings.get if fixings is not None else (lambda d: None)
        tau = self.accrual
        if self.method == "compounded":
            tn = float(projection.time(self.end))
            g = 0.0
            for d0, d1, o in zip(acc, bounds, obs):
                t0, t1 = float(projection.time(d0)), float(projection.time(d1))
                if t0 > 0 and getter(o) is None:
                    g += _B(a, t1 - t0) * hull_white_compounding_covariance(t0, tn, a, sigma)
            F = self.forward_rate(projection, fixings)
            return (1.0 + tau * F) * math.expm1(g) / tau
        # averaged: each fixing r_o is the rate over [o, next fixing date], weighted by w_i
        dc = self.index.dc
        total = 0.0
        for o, wi in zip(obs, weights):
            t0 = float(projection.time(o))
            if t0 <= 0 or getter(o) is not None:
                continue
            nxt = self.index.next_fixing_date(o)
            t1 = float(projection.time(nxt))
            growth = float(projection.discount(o)) / float(projection.discount(nxt))
            gi = _B(a, t1 - t0) * hull_white_compounding_covariance(t0, t1, a, sigma)
            total += wi / dc.year_fraction(o, nxt) * growth * math.expm1(gi)
        return total / tau

    def implied_rate(
        self,
        projection: YieldCurve,
        a: float = 0.0,
        sigma: float = 0.0,
        fixings: Optional[FixingHistory] = None,
    ) -> float:
        """Model futures rate: forward plus convexity."""
        return self.forward_rate(projection, fixings) + self.convexity_adjustment(
            projection, a, sigma, fixings
        )

    def implied_price(
        self,
        projection: YieldCurve,
        a: float = 0.0,
        sigma: float = 0.0,
        fixings: Optional[FixingHistory] = None,
    ) -> float:
        return 100.0 * (1.0 - self.implied_rate(projection, a, sigma, fixings))
