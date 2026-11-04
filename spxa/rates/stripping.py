r"""Stripping caplet volatilities from cap quotes.

Caps are quoted with one flat volatility per maturity. With
:math:`\sigma^{\text{flat}}_n` the quote of the :math:`n`-th cap (all at
the same strike, or all ATM with their own strikes), the caplet
volatilities are bootstrapped as a piecewise-constant function of the
caplet expiry. The caplets that cap :math:`n` adds to cap :math:`n-1`
share one volatility, which is solved so that

.. math:: \sum_{i \in \mathrm{cap}_n} \mathrm{caplet}_i(\sigma_i) =
          \sum_{i \in \mathrm{cap}_n} \mathrm{caplet}_i(\sigma^{\text{flat}}_n).

Caplets already in cap :math:`n-1` keep their stripped vols. Each step is
a one-dimensional root of a monotone function (Brent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from spxa.market.curves import YieldCurve
from spxa.rates.capfloor import CapFloor

__all__ = ["CapletVolCurve", "strip_caplet_vols"]


@dataclass(frozen=True)
class CapletVolCurve:
    """Piecewise-constant caplet vol: ``vols[i]`` applies to expiries in ``(expiries[i-1], expiries[i]]``."""

    expiries: np.ndarray
    vols: np.ndarray

    def __call__(self, T: float, K: float = 0.0) -> float:
        i = int(np.searchsorted(self.expiries, T - 1e-12, side="left"))
        return float(self.vols[min(i, self.vols.size - 1)])


def strip_caplet_vols(
    caps: Sequence[CapFloor],
    flat_vols: Sequence[float],
    discount: YieldCurve,
    projection: Optional[YieldCurve] = None,
    model: str = "black",
    shift: float = 0.0,
    bounds: tuple[float, float] = (1e-6, 5.0),
) -> CapletVolCurve:
    """Bootstrap piecewise-constant caplet vols from caps with increasing maturities.

    Each cap must extend the previous one: its first caplets coincide with
    the previous cap's (same fixing dates) and it adds at least one more.
    The strikes may differ (ATM caps); each caplet is valued at its own
    cap's strike.
    """
    if len(caps) != len(flat_vols) or not caps:
        raise ValueError("need one flat vol per cap")
    edges: list[float] = []
    vols: list[float] = []
    n_prev = 0
    for cap, fv in zip(caps, flat_vols):
        n = len(cap.caplets)
        if n <= n_prev:
            raise ValueError("caps must have increasing numbers of caplets")
        exps = [c.expiry(discount) for c in cap.caplets]
        target = cap.price(discount, projection, float(fv), model, shift)
        known = CapletVolCurve(np.array(edges), np.array(vols)) if vols else None
        old = sum(
            c.price(discount, projection, known(exps[i]) if known else 0.0, model, shift)
            for i, c in enumerate(cap.caplets[:n_prev])
        )
        new = cap.caplets[n_prev:]

        def f(s: float) -> float:
            return old + sum(c.price(discount, projection, s, model, shift) for c in new) - target

        lo, hi = bounds
        if f(lo) > 0 or f(hi) < 0:
            raise ValueError(f"cap {n} cannot be matched: stripped vols would leave {bounds}")
        vols.append(float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-14)))
        edges.append(exps[-1])
        n_prev = n
    return CapletVolCurve(np.array(edges), np.array(vols))
