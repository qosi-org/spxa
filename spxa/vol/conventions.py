r"""Volatility-surface coordinates.

Surfaces are parametrised in **log-forward moneyness** and **total implied
variance**,

.. math:: k = \log\frac{K}{F_T}, \qquad w(k, T) = \sigma_{\mathrm{BS}}^2(k, T)\,T,

the coordinates in which SVI, SSVI and the no-arbitrage conditions are
naturally stated (Gatheral 2004; Gatheral & Jacquier 2014). The undiscounted
call price normalised by the forward is then

.. math:: c(k, w) = \Phi(d_1) - e^{k}\Phi(d_2), \quad
          d_{1,2} = -\frac{k}{\sqrt w} \pm \frac{\sqrt w}{2},

that is :math:`c = e^{k/2}\,b(-k, \sqrt w)` in terms of
:func:`spxa.market.normalised_black_call`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.market.black import black_price
from spxa.market.implied import implied_black_volatility

__all__ = [
    "implied_vol_from_total_variance",
    "log_moneyness",
    "normalised_call_from_total_variance",
    "strike_from_log_moneyness",
    "total_variance",
    "total_variance_from_price",
]


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


def log_moneyness(K: Any, F: Any) -> Any:
    r""":math:`k = \log(K/F)`."""
    K, F = np.asarray(K, float), np.asarray(F, float)
    if np.any(K <= 0) or np.any(F <= 0):
        raise ValueError("strikes and forwards must be positive")
    return _out(np.log(K / F))


def strike_from_log_moneyness(k: Any, F: Any) -> Any:
    r""":math:`K = F e^{k}`."""
    return _out(np.asarray(F, float) * np.exp(np.asarray(k, float)))


def total_variance(sigma: Any, T: Any) -> Any:
    r""":math:`w = \sigma^2 T`."""
    s, t = np.asarray(sigma, float), np.asarray(T, float)
    if np.any(t < 0):
        raise ValueError("T must be non-negative")
    return _out(s * s * t)


def implied_vol_from_total_variance(w: Any, T: Any) -> Any:
    r""":math:`\sigma = \sqrt{w/T}` (``NaN`` for negative variance)."""
    w, t = np.asarray(w, float), np.asarray(T, float)
    if np.any(t <= 0):
        raise ValueError("T must be positive")
    with np.errstate(invalid="ignore"):
        return _out(np.where(w >= 0, np.sqrt(np.maximum(w, 0.0) / t), np.nan))


def normalised_call_from_total_variance(k: Any, w: Any) -> Any:
    r"""Undiscounted call price over the forward, :math:`C/(P F)`, at log-moneyness ``k``."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    if np.any(w < 0):
        raise ValueError("total variance must be non-negative")
    return _out(np.asarray(black_price(1.0, np.exp(k), 1.0, np.sqrt(w))))


def total_variance_from_price(
    price: Any, F: Any, K: Any, T: Any, df: Any = 1.0, option: Any = "call"
) -> Any:
    """Total implied variance of option prices (``NaN`` outside the arbitrage bounds)."""
    vol = np.asarray(implied_black_volatility(price, F, K, T, df, option), float)
    return _out(vol * vol * np.asarray(T, float))
