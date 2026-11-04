r"""Adapter between :mod:`spxa.risk` swaps and yield curves.

:func:`curve_price` values a :class:`spxa.risk.instruments.InterestRateSwap`
from a flat market dictionary, the format of
:class:`spxa.risk.scenarios.MarketScenario`:

* ``market[<rate id> + "_CURVE"]`` (optional): a
  :class:`~spxa.market.curves.YieldCurve`;
* ``market["r"]``: the continuously compounded short rate. Without a curve
  the swap is valued on a flat curve at ``r``. With a curve, ``r`` minus its
  value in ``base`` is applied as a parallel zero-rate shift, which is how a
  scenario's ``rate_shifts_bps`` reaches the curve.

The swap pays the fixed rate :math:`K` every ``payment_frequency`` years on
the times :math:`t_k = k\,\delta` (with a short last period ending at the
maturity :math:`T`) and the single-curve floating leg, worth
:math:`1 - P(T)` per unit notional. The payer value is

.. math::
    V = N\Bigl(1 - P(T) - K\sum_k \tau_k P(t_k)\Bigr),

and the receiver value is :math:`-V`. The DV01 is the exact derivative for a
one-basis-point parallel rise in continuously compounded zero rates,
:math:`\partial V/\partial z = N\bigl(T P(T) + K\sum_k \tau_k t_k P(t_k)\bigr)`,
times :math:`10^{-4}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from spxa.market.curves import FlatCurve, YieldCurve

__all__ = ["CurveQuote", "curve_price", "curve_pricing_fn"]

Market = Mapping[str, Any]


@dataclass(frozen=True)
class CurveQuote:
    """Notional-scaled value of a risk swap, with its DV01, par rate and annuity."""

    instrument_id: str
    price: float
    dv01: float
    par_rate: float
    annuity: float


def _schedule(maturity: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    if maturity <= 0 or step <= 0:
        raise ValueError("maturity and payment_frequency must be positive")
    n = int(np.ceil(maturity / step - 1e-9))
    times = np.minimum(step * np.arange(1, n + 1), maturity)
    return times, np.diff(np.r_[0.0, times])


def _curve(instrument: Any, market: Market, base: Optional[Market]) -> YieldCurve:
    key = f"{instrument.underlying_ids[0]}_CURVE"
    if key not in market:
        if "r" not in market:
            raise KeyError(f"market needs '{key}' or 'r'")
        return FlatCurve(float(market["r"]))
    curve: YieldCurve = market[key]
    if base is not None and "r" in market and "r" in base:
        shift = float(market["r"]) - float(base["r"])
        if shift != 0.0:
            curve = curve.shifted(shift)
    return curve


def curve_price(instrument: Any, market: Market, base: Optional[Market] = None) -> CurveQuote:
    """Value an :class:`~spxa.risk.instruments.InterestRateSwap` on a curve.

    ``base`` is the unstressed market; when given, the change in ``r``
    shifts the curve in parallel (see the module docstring).
    """
    if not all(hasattr(instrument, a) for a in ("fixed_rate", "payment_frequency")):
        raise TypeError(f"{type(instrument).__name__} is not an interest-rate swap")
    curve = _curve(instrument, market, base)
    T = float(instrument.maturity)
    times, tau = _schedule(T, float(instrument.payment_frequency))
    p = np.asarray(curve.discount(times), float)
    annuity = float(tau @ p)
    p_T = float(p[-1])
    K = float(instrument.fixed_rate)
    sign = -1.0 if instrument.receive_fixed else 1.0
    n = float(instrument.notional)
    value = sign * n * (1.0 - p_T - K * annuity)
    dvdz = sign * n * (T * p_T + K * float(tau @ (times * p)))
    return CurveQuote(
        instrument_id=instrument.instrument_id,
        price=value,
        dv01=1e-4 * dvdz,
        par_rate=(1.0 - p_T) / annuity,
        annuity=n * annuity,
    )


def curve_pricing_fn(
    book: Union[Any, Sequence[tuple[Any, float]]],
    base_market: Optional[Market] = None,
) -> Callable[[Market], float]:
    """``market -> value`` callable for :class:`~spxa.risk.scenarios.StressTestingEngine`.

    ``book`` is one swap, ``(swap, quantity)`` pairs or an object with
    ``positions``. ``base_market`` anchors the parallel shift of curves.
    """
    if hasattr(book, "positions"):
        positions = list(book.positions)
    elif isinstance(book, (list, tuple)):
        positions = list(book)
    else:
        positions = [(book, 1.0)]

    def value(market: Market) -> float:
        return float(sum(q * curve_price(s, market, base_market).price for s, q in positions))

    return value
