r"""
Adapter between :mod:`spxa.risk` instruments and the PDE pricers.

:func:`pde_price` values a :class:`spxa.risk.instruments.Instrument` from a
flat market dictionary in the format used by
:class:`spxa.risk.scenarios.MarketScenario`:

* ``market[<underlying id>]``: spot;
* ``market[<underlying id> + "_VOL"]``: Black–Scholes volatility;
* ``market[<underlying id> + "_DIV"]`` (optional): continuous dividend yield;
* ``market["r"]``: continuously compounded short rate;
* ``market["<id1>_<id2>_CORR"]`` (or ``"CORR"``): correlation for
  two-asset baskets.

Supported contracts and the PDE used:

=====================  ======================================================
``EuropeanOption``     :meth:`BlackScholesPDE.european`
``BarrierOption``      :meth:`BlackScholesPDE.barrier` (continuous monitoring)
``AsianOption``        :class:`AsianPDE` (continuous arithmetic average)
``BasketOption``       :class:`TwoAssetBlackScholesPDE` (two underlyings)
=====================  ======================================================

Path-based ``payoff`` methods of the risk instruments monitor barriers and
averages on the simulation grid; the PDE values are the continuous-time
limits. Prices and sensitivities are multiplied by the notional.
:func:`pde_pricing_fn` turns an instrument, a list of
``(instrument, quantity)`` pairs or a :class:`~spxa.risk.portfolio.Portfolio`
into a ``market -> value`` callable for
:class:`~spxa.risk.scenarios.StressTestingEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from spxa.pde.asian import AsianPDE
from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.two_asset import TwoAssetBlackScholesPDE

__all__ = ["PDEQuote", "pde_price", "pde_pricing_fn", "supported_instruments"]

Market = Mapping[str, Any]


@dataclass
class PDEQuote:
    """Notional-scaled PDE value of a risk instrument.

    ``delta``, ``gamma`` and (when requested) ``vega`` are keyed by
    underlying id; vega is per unit of volatility. ``result`` is the raw
    pricer output for one unit of notional.
    """

    instrument_id: str
    price: float
    delta: dict[str, float]
    gamma: dict[str, float]
    vega: Optional[dict[str, float]] = None
    method: str = ""
    result: Any = field(default=None, repr=False)

    def __float__(self) -> float:
        return float(self.price)

    def __story__(self) -> str:
        """Plain-language summary of the quote."""
        lines = [f"{self.instrument_id}: {self.method}", f"  value = {self.price:.8f}"]
        for uid in self.delta:
            extra = f", vega = {self.vega[uid]:.6f}" if self.vega else ""
            lines.append(
                f"  {uid}: delta = {self.delta[uid]:.6f}, gamma = {self.gamma[uid]:.6f}{extra}"
            )
        return "\n".join(lines)


def supported_instruments() -> tuple[str, ...]:
    """Names of the instrument classes :func:`pde_price` accepts."""
    return ("EuropeanOption", "BarrierOption", "AsianOption", "BasketOption")


def _get(market: Market, key: str) -> float:
    if key not in market:
        raise KeyError(f"market data '{key}' is missing")
    return float(market[key])


def _single(market: Market, uid: str) -> tuple[float, float, float, float]:
    return (
        _get(market, uid),
        _get(market, f"{uid}_VOL"),
        float(market.get(f"{uid}_DIV", 0.0)),
        _get(market, "r"),
    )


def _correlation(market: Market, a: str, b: str) -> float:
    for key in (f"{a}_{b}_CORR", f"{b}_{a}_CORR", "CORR"):
        if key in market:
            return float(market[key])
    raise KeyError(f"market data '{a}_{b}_CORR' is missing")


def _price_unit(instrument: Any, market: Market, kwargs: dict[str, Any]) -> tuple[Any, list[str]]:
    kind = type(instrument).__name__
    ids = list(instrument.underlying_ids)
    T = float(instrument.maturity)
    if kind in ("EuropeanOption", "BarrierOption", "AsianOption"):
        s, vol, q, r = _single(market, ids[0])
        option = "call" if instrument.is_call else "put"
        if kind == "EuropeanOption":
            res = BlackScholesPDE(r, vol, q=q).european(instrument.strike, T, s, option, **kwargs)
        elif kind == "BarrierOption":
            res = BlackScholesPDE(r, vol, q=q).barrier(
                instrument.strike,
                T,
                s,
                instrument.barrier,
                barrier_type=instrument.barrier_type,
                option=option,
                rebate=instrument.rebate,
                rebate_timing="expiry",
                **kwargs,
            )
        else:
            res = AsianPDE(r, vol, q=q).price(instrument.strike, T, s, option, **kwargs)
        return res, ids
    if kind == "BasketOption":
        if len(ids) != 2:
            raise NotImplementedError("PDE basket pricing supports exactly two underlyings")
        s1, v1, q1, r = _single(market, ids[0])
        s2, v2, q2, _ = _single(market, ids[1])
        rho = _correlation(market, ids[0], ids[1])
        w1, w2 = (float(w) for w in instrument.weights)
        K = float(instrument.strike)
        sign = 1.0 if instrument.is_call else -1.0
        model = TwoAssetBlackScholesPDE(r, v1, v2, rho, q1, q2)
        res = model.price(
            lambda a, b: np.maximum(sign * (w1 * a + w2 * b - K), 0.0),
            T,
            s1,
            s2,
            label=f"basket {'call' if sign > 0 else 'put'} K={K:g}",
            **kwargs,
        )
        return res, ids
    raise NotImplementedError(
        f"no PDE pricer for {kind}; supported: {', '.join(supported_instruments())}"
    )


def pde_price(
    instrument: Any,
    market: Market,
    vega: bool = False,
    vol_bump: float = 1e-3,
    **kwargs: Any,
) -> PDEQuote:
    """Value a risk instrument by finite differences.

    Parameters
    ----------
    instrument : spxa.risk.instruments.Instrument
    market : mapping
        Market data (see the module docstring).
    vega : bool
        Add vegas by central bump-and-revalue of each ``<id>_VOL``.
    vol_bump : float
        Absolute volatility bump for vega.
    **kwargs
        Resolution and solver options forwarded to the pricer.
    """
    res, ids = _price_unit(instrument, market, kwargs)
    n = float(instrument.notional)
    if isinstance(res.delta, tuple):  # two-asset result
        delta = {ids[0]: n * res.delta[0], ids[1]: n * res.delta[1]}
        gamma = {ids[0]: n * res.gamma[0], ids[1]: n * res.gamma[1]}
    else:
        delta = {ids[0]: n * float(res.delta)}
        gamma = {ids[0]: n * float(res.gamma)}
    vegas = None
    if vega:
        vegas = {}
        for uid in ids:
            key = f"{uid}_VOL"
            up = dict(market, **{key: float(market[key]) + vol_bump})
            dn = dict(market, **{key: float(market[key]) - vol_bump})
            pu = _price_unit(instrument, up, dict(kwargs))[0].price
            pd = _price_unit(instrument, dn, dict(kwargs))[0].price
            vegas[uid] = n * (pu - pd) / (2.0 * vol_bump)
    return PDEQuote(
        instrument_id=instrument.instrument_id,
        price=n * float(res.price),
        delta=delta,
        gamma=gamma,
        vega=vegas,
        method=str(res.method),
        result=res,
    )


def pde_pricing_fn(
    book: Union[Any, Sequence[tuple[Any, float]]],
    **kwargs: Any,
) -> Callable[[Market], float]:
    """``market -> value`` callable for stress testing.

    ``book`` is a single instrument, a sequence of ``(instrument, quantity)``
    pairs or an object with a ``positions`` attribute of such pairs (a
    :class:`~spxa.risk.portfolio.Portfolio`). Keyword arguments are
    forwarded to :func:`pde_price`.
    """
    if hasattr(book, "positions"):
        positions = list(book.positions)
    elif isinstance(book, (list, tuple)):
        positions = list(book)
    else:
        positions = [(book, 1.0)]

    def value(market: Market) -> float:
        return float(sum(q * pde_price(inst, market, **kwargs).price for inst, q in positions))

    return value
