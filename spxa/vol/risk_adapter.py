r"""Adapter between :mod:`spxa.risk` options and implied-volatility surfaces.

:func:`surface_price` values a :class:`spxa.risk.instruments.EuropeanOption`
from a flat market dictionary in the format of
:class:`spxa.risk.scenarios.MarketScenario`:

* ``market[<id> + "_SURFACE"]``: a :class:`~spxa.vol.surface.VolSurface`
  whose forward curve has a ``spot`` (an
  :class:`~spxa.market.curves.EquityForward`);
* ``market[<id>]`` (optional): the spot :math:`S`. The surface's forwards
  are scaled by :math:`S/S_0`, i.e. carry is proportional;
* against a ``base`` market, the change in ``market[<id> + "_VOL"]`` is a
  parallel implied-volatility shift :math:`\Delta\sigma` and the change in
  ``market["r"]`` a parallel rate shift :math:`\Delta r`, which multiplies
  discount factors by :math:`e^{-\Delta r T}` and forwards by
  :math:`e^{\Delta r T}`.

With :math:`F = F_0(T)\,(S/S_0)\,e^{\Delta r T}` the option is priced with
Black's formula at

``"sticky_strike"``
    :math:`\sigma(K, T) + \Delta\sigma`: the smile stays fixed in strike;
``"sticky_moneyness"``
    :math:`\sigma(\log(K/F), T) + \Delta\sigma`: the smile moves with the
    forward.

Spot delta is the central difference in :math:`S` under the chosen
dynamics; vega is Black's for a parallel shift. Both are multiplied by the
notional, as is the price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from spxa.market.black import black_price
from spxa.risk.instruments import EuropeanOption

__all__ = ["SurfaceQuote", "surface_price", "surface_pricing_fn"]

Market = Mapping[str, Any]


@dataclass(frozen=True)
class SurfaceQuote:
    """Notional-scaled value of a risk option on a surface."""

    instrument_id: str
    price: float
    vol: float
    forward: float
    delta: float
    vega: float


def _shift(market: Market, base: Optional[Market], key: str) -> float:
    if base is None or key not in market or key not in base:
        return 0.0
    return float(market[key]) - float(base[key])


def surface_price(
    instrument: Any,
    market: Market,
    base: Optional[Market] = None,
    dynamics: str = "sticky_strike",
    spot_bump: float = 1e-4,
) -> SurfaceQuote:
    """Value a European option on ``market[<id>_SURFACE]`` (see the module docstring)."""
    if dynamics not in ("sticky_strike", "sticky_moneyness"):
        raise ValueError("dynamics must be 'sticky_strike' or 'sticky_moneyness'")
    if not isinstance(instrument, EuropeanOption):
        raise TypeError(f"{type(instrument).__name__} is not a European option")
    uid = instrument.underlying_ids[0]
    surface = market[f"{uid}_SURFACE"]
    s0 = getattr(surface.forward_curve, "spot", None)
    if s0 is None:
        raise ValueError("the surface's forward curve needs a spot")
    spot = float(market.get(uid, s0))
    K, T = float(instrument.strike), float(instrument.maturity)
    dr, dsig = _shift(market, base, "r"), _shift(market, base, f"{uid}_VOL")
    carry = float(surface.forward(T)) * math.exp(dr * T) / float(s0)
    D = float(surface.discount(T)) * math.exp(-dr * T)
    option = "call" if instrument.is_call else "put"

    def vol(F: float) -> float:
        if dynamics == "sticky_strike":
            sig = float(surface.implied_vol(K, T))
        else:
            sig = float(surface.implied_vol_k(math.log(K / F), T))
        return max(sig + dsig, 0.0)

    def value(S: float) -> float:
        F = S * carry
        return float(black_price(F, K, T, vol(F), D, option))

    F = spot * carry
    sig = vol(F)
    h = spot_bump * spot
    n = float(instrument.notional)
    sd = sig * math.sqrt(T)
    vega = 0.0
    if sd > 0:
        d1 = math.log(F / K) / sd + 0.5 * sd
        vega = D * F * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)
    return SurfaceQuote(
        instrument_id=instrument.instrument_id,
        price=n * value(spot),
        vol=sig,
        forward=F,
        delta=n * (value(spot + h) - value(spot - h)) / (2 * h),
        vega=n * vega,
    )


def surface_pricing_fn(
    book: Union[Any, Sequence[tuple[Any, float]]],
    base_market: Optional[Market] = None,
    dynamics: str = "sticky_strike",
) -> Callable[[Market], float]:
    """``market -> value`` callable for :class:`~spxa.risk.scenarios.StressTestingEngine`.

    ``book`` is one option, ``(option, quantity)`` pairs or an object with
    ``positions``. ``base_market`` anchors the vol and rate shifts.
    """
    if hasattr(book, "positions"):
        positions = list(book.positions)
    elif isinstance(book, (list, tuple)):
        positions = list(book)
    else:
        positions = [(book, 1.0)]

    def value(market: Market) -> float:
        return float(
            sum(q * surface_price(o, market, base_market, dynamics).price for o, q in positions)
        )

    return value
