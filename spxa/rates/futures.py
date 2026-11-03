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
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Optional

from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.rates.indices import FixingHistory, IborIndex

__all__ = [
    "RateFuture",
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
