r"""Calibration instruments for curve construction.

A *helper* wraps a market instrument and its quote. Given a set of named
curves, it returns the model quote, and the bootstrapper drives
:math:`\text{model} - \text{market}` to zero. Every helper has:

``quote``
    The market quote: a rate, a futures price or a basis spread.
``pillar``
    The latest date at which the instrument reads any curve. This is
    where a bootstrapped curve gets a node for it.
``curves``
    Names of the curves it reads, resolved through a mapping at pricing
    time. Single-curve set-ups use one name, ``"discount"``, for
    everything. Multi-curve set-ups name an OIS discount curve and one
    projection curve per index.
``implied_quote(curves)``
    The model quote on a mapping ``{name: YieldCurve}``.
"""

from __future__ import annotations

import abc
import datetime as _dt
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

from spxa.market.curves import YieldCurve
from spxa.rates.cashflows import Cashflow, IborCoupon, OvernightCoupon
from spxa.rates.bonds import FixedRateBond
from spxa.rates.futures import OvernightFuture, RateFuture
from spxa.rates.indices import OvernightIndex
from spxa.rates.instruments import FRA, Deposit
from spxa.rates.swaps import BasisSwap, Swap

__all__ = [
    "CurveHelper",
    "DepositHelper",
    "FRAHelper",
    "FutureHelper",
    "SwapHelper",
    "BasisSwapHelper",
    "last_curve_date",
    "quarterly_futures",
    "BondHelper",
]

Curves = Mapping[str, YieldCurve]


def last_curve_date(leg: Sequence[Cashflow]) -> _dt.date:
    """Latest date a leg reads from any curve (payments and index periods)."""
    out = max(c.payment_date for c in leg)
    for c in leg:
        if isinstance(c, IborCoupon):
            end = c.end
            if c.use_index_dates:
                end = c.index.maturity_date(c.index.value_date(c.fixing))
            out = max(out, end)
        elif isinstance(c, OvernightCoupon):
            out = max(out, c.end)
    return out


class CurveHelper(abc.ABC):
    """A quoted instrument used to build curves."""

    quote: float

    @property
    @abc.abstractmethod
    def pillar(self) -> _dt.date:
        """Latest date the instrument depends on."""

    @property
    @abc.abstractmethod
    def curves(self) -> tuple[str, ...]:
        """Names of the curves read by :meth:`implied_quote`."""

    @abc.abstractmethod
    def implied_quote(self, curves: Curves) -> float:
        """Model quote on the named curves."""

    def residual(self, curves: Curves) -> float:
        """Model minus market quote."""
        return self.implied_quote(curves) - self.quote

    def _get(self, curves: Curves, name: str) -> YieldCurve:
        try:
            return curves[name]
        except KeyError:
            raise KeyError(f"{type(self).__name__} needs curve {name!r}") from None


@dataclass
class DepositHelper(CurveHelper):
    """Simple-rate deposit: quote :math:`(P(t_s)/P(t_e) - 1)/\\tau`."""

    deposit: Deposit
    quote: float = float("nan")
    curve: str = "discount"

    def __post_init__(self) -> None:
        if self.quote != self.quote:
            self.quote = self.deposit.rate

    @property
    def pillar(self) -> _dt.date:
        return self.deposit.end

    @property
    def curves(self) -> tuple[str, ...]:
        return (self.curve,)

    def implied_quote(self, curves: Curves) -> float:
        return self.deposit.implied_rate(self._get(curves, self.curve))


@dataclass
class FRAHelper(CurveHelper):
    """FRA quoted at its forward rate on a projection curve."""

    fra: FRA
    quote: float = float("nan")
    curve: str = "discount"

    def __post_init__(self) -> None:
        if self.quote != self.quote:
            self.quote = self.fra.strike

    @property
    def pillar(self) -> _dt.date:
        return self.fra.end

    @property
    def curves(self) -> tuple[str, ...]:
        return (self.curve,)

    def implied_quote(self, curves: Curves) -> float:
        return self.fra.forward_rate(self._get(curves, self.curve))


@dataclass
class FutureHelper(CurveHelper):
    """Term-rate or overnight future quoted by price, with Hull–White convexity ``(a, sigma)``."""

    future: Union[RateFuture, OvernightFuture]
    quote: float = float("nan")
    curve: str = "discount"
    a: float = 0.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.quote != self.quote:
            self.quote = self.future.price

    @property
    def pillar(self) -> _dt.date:
        return self.future.end

    @property
    def curves(self) -> tuple[str, ...]:
        return (self.curve,)

    def implied_quote(self, curves: Curves) -> float:
        return self.future.implied_price(self._get(curves, self.curve), self.a, self.sigma)


@dataclass
class BondHelper(CurveHelper):
    """Fixed-rate bond quoted by clean price (default) or by yield, settling on ``settle``.

    The model clean price is the forward dirty price at settlement on the
    curve, :math:`\\sum C_i P(t_i)/P(t_s)`, minus accrued interest. The
    model yield is the bond yield of that clean price.
    """

    bond: FixedRateBond
    quote: float
    settle: _dt.date
    curve: str = "discount"
    quote_type: str = "clean"

    def __post_init__(self) -> None:
        if self.quote_type not in ("clean", "yield"):
            raise ValueError("quote_type must be 'clean' or 'yield'")

    @property
    def pillar(self) -> _dt.date:
        return self.bond.cashflows()[-1].payment_date

    @property
    def curves(self) -> tuple[str, ...]:
        return (self.curve,)

    def implied_quote(self, curves: Curves) -> float:
        c = self._get(curves, self.curve)
        clean = self.bond.dirty_price_from_curve(c, self.settle) - self.bond.accrued_interest(
            self.settle
        )
        if self.quote_type == "clean":
            return clean
        return self.bond.yield_from_clean(clean, self.settle)


def quarterly_futures(
    reference_date: Any,
    prices: Sequence[float],
    index: Optional[OvernightIndex] = None,
    curve: str = "discount",
    a: float = 0.0,
    sigma: float = 0.0,
) -> list[FutureHelper]:
    """Helpers for consecutive quarterly 3M overnight futures from the next IMM date.

    ``prices[i]`` is the price of the ``i``-th contract. Contract ``i``
    runs from IMM date ``i`` to IMM date ``i + 1``, so the strip pins
    every forward between its first and last IMM dates.
    """
    from spxa.market.dates import to_date
    from spxa.rates.futures import next_imm_date
    from spxa.rates.indices import SOFR

    idx = index if index is not None else SOFR
    d = next_imm_date(to_date(reference_date))
    out = []
    for p in prices:
        fut = OvernightFuture.from_imm(d, float(p), idx)
        out.append(FutureHelper(fut, float(p), curve, a, sigma))
        d = fut.end
    return out


@dataclass
class SwapHelper(CurveHelper):
    """Fixed-for-floating swap quoted at its par rate.

    ``forward`` names the projection curve of the floating index. ``None``
    means the discount curve itself, as for OIS or single-curve set-ups.
    """

    swap: Swap
    quote: float = float("nan")
    discount: str = "discount"
    forward: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quote != self.quote:
            self.quote = self.swap.fixed_rate

    @property
    def pillar(self) -> _dt.date:
        return max(last_curve_date(self.swap.fixed), last_curve_date(self.swap.floating))

    @property
    def curves(self) -> tuple[str, ...]:
        if self.forward is None or self.forward == self.discount:
            return (self.discount,)
        return (self.discount, self.forward)

    def implied_quote(self, curves: Curves) -> float:
        d = self._get(curves, self.discount)
        f = d if self.forward is None else self._get(curves, self.forward)
        return self.swap.par_rate(d, f)


@dataclass
class BasisSwapHelper(CurveHelper):
    """Floating-floating swap quoted as the par spread on its first leg.

    The model quote is the spread the first leg needs for zero PV: the
    spread its coupons already carry plus :meth:`BasisSwap.par_spread`. It
    does not matter which spread the swap was built with.
    """

    swap: BasisSwap
    quote: float
    first: str
    second: str
    discount: str = "discount"

    @property
    def pillar(self) -> _dt.date:
        return max(last_curve_date(self.swap.first), last_curve_date(self.swap.second))

    @property
    def curves(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.discount, self.first, self.second)))

    @property
    def leg_spread(self) -> float:
        """Spread already carried by the first leg's coupons."""
        spreads = {float(getattr(c, "spread", 0.0)) for c in self.swap.first}
        if len(spreads) != 1:
            raise ValueError("the first leg carries several spreads")
        return spreads.pop()

    def implied_quote(self, curves: Curves) -> float:
        # par_spread is the spread to add on top of the one already on leg 1
        return self.leg_spread + self.swap.par_spread(
            self._get(curves, self.discount),
            self._get(curves, self.first),
            self._get(curves, self.second),
        )
