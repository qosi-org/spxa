r"""The one-factor Cheyette (quasi-Gaussian) model with linear local volatility.

Cheyette (1992), in the form of Andersen & Piterbarg (Vol. II, ch. 13):
:math:`r_t = f(0, t) + x_t` with

.. math::
    dx_t = (y_t - \kappa x_t)\,dt + \sigma_r(t, x_t)\,dW_t, \qquad
    dy_t = \bigl(\sigma_r(t, x_t)^2 - 2\kappa y_t\bigr)\,dt, \qquad x_0 = y_0 = 0.

The HJM volatility :math:`\sigma_r(t, x_t)e^{-\kappa(T-t)}` is separable,
so the two states :math:`(x, y)` carry the whole curve:

.. math::
    P(t, T) = \frac{P(0, T)}{P(0, t)}\exp\bigl(-G(t, T)x_t - \tfrac12 G(t, T)^2 y_t\bigr),
    \qquad G(t, T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}.

Here the local volatility is linear in the state,
:math:`\sigma_r(t, x) = \max(\sigma(t) + \beta x, 0)`, with
:math:`\sigma(t)` piecewise constant. The skew :math:`\beta` raises the
volatility when rates rise (:math:`\beta > 0`, towards lognormal). With
:math:`\beta = 0`, :math:`y` is deterministic and the model is Hull–White
(:meth:`Cheyette.to_hull_white`). :meth:`Cheyette.ybar` is :math:`y` along
:math:`x \equiv 0`, the usual freezing point for approximations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.hull_white import HullWhite

__all__ = ["Cheyette"]


class Cheyette:
    """Cheyette model on ``curve`` with mean reversion ``kappa``, vol levels ``sigma`` and skew ``beta``."""

    def __init__(
        self,
        curve: YieldCurve,
        kappa: float,
        sigma: Any,
        beta: float = 0.0,
        sigma_times: Optional[Sequence[float]] = None,
    ) -> None:
        # HullWhite validates kappa, the levels and the knots, and gives ybar in closed form
        self._hw = HullWhite(curve, kappa, sigma, sigma_times)
        self.curve = curve
        self.kappa = float(kappa)
        self.sigma = self._hw.sigma
        self.sigma_times = self._hw.sigma_times
        self.beta = float(beta)

    def sigma_at(self, t: Any) -> Any:
        """Level :math:`\\sigma(t)` (right-continuous at the knots)."""
        k = np.searchsorted(self.sigma_times, np.asarray(t, float), side="right")
        return self.sigma[k]

    def local_vol(self, t: Any, x: Any) -> Any:
        return np.maximum(self.sigma_at(t) + self.beta * np.asarray(x, float), 0.0)

    def G(self, t: Any, T: Any) -> Any:
        return -np.expm1(-self.kappa * (np.asarray(T, float) - np.asarray(t, float))) / self.kappa

    def discount_bond(self, t: float, T: Any, x: Any, y: Any) -> Any:
        """:math:`P(t, T)` given the states (broadcasts over ``T`` and the states)."""
        TT = np.asarray(T, float)
        g = self.G(t, TT)
        ratio = np.asarray(self.curve.discount(TT), float) / float(self.curve.discount(float(t)))
        return ratio * np.exp(-g * np.asarray(x, float) - 0.5 * g * g * np.asarray(y, float))

    def short_rate(self, t: float, x: Any) -> Any:
        return self.curve.instantaneous_forward(t) + np.asarray(x, float)

    def ybar(self, t: float) -> float:
        """:math:`\\int_0^t e^{-2\\kappa(t-u)}\\sigma(u)^2 du`: :math:`y_t` along :math:`x \\equiv 0`."""
        return float(self._hw.y(t))

    def to_hull_white(self) -> HullWhite:
        if self.beta != 0.0:
            raise ValueError("only the beta = 0 Cheyette model is Hull-White")
        return self._hw
