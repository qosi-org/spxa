r"""Hull–White one-factor model fitted to a discount curve.

In the parametrisation of Andersen & Piterbarg (*Interest Rate Modeling*,
§10.1.2) the short rate is :math:`r_t = f(0, t) + x_t` with

.. math:: dx_t = \bigl(y(t) - a\,x_t\bigr)\,dt + \sigma(t)\,dW_t, \qquad x_0 = 0,

under the risk-neutral measure. Mean reversion :math:`a > 0` is constant
and :math:`\sigma(t)` piecewise constant. With
:math:`B(t, T) = (1 - e^{-a(T - t)})/a` and

.. math:: y(t) = \operatorname{Var}(x_t) = \int_0^t e^{-2a(t - u)}\sigma(u)^2\,du,

zero-coupon bonds are

.. math:: P(t, T) = \frac{P(0, T)}{P(0, t)}\exp\Bigl(-B(t, T)\,x_t - \tfrac12 B(t, T)^2\,y(t)\Bigr),

so the model reproduces the initial curve :math:`P(0, \cdot)` by
construction. Under the :math:`T`-forward measure, :math:`x_t` (for
:math:`t \le T`) is Gaussian with variance :math:`y(t)` and mean
:math:`-B(t, T)\,y(t)`. :math:`\log P(T, S)` is Gaussian with variance
:math:`v^2 = B(T, S)^2\,y(T)`, which gives Black-type zero-coupon bond
options (Jamshidian, 1989).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import norm

from spxa.market.curves import YieldCurve

__all__ = ["HullWhite"]


class HullWhite:
    """Hull–White model on ``curve`` with mean reversion ``a`` and volatility ``sigma``.

    ``sigma`` is a number or a sequence of levels: ``sigma[k]`` applies on
    ``[sigma_times[k-1], sigma_times[k])`` (with ``sigma_times[-1] = 0``),
    and the last level continues for ever, so ``len(sigma_times) ==
    len(sigma) - 1``.
    """

    def __init__(
        self,
        curve: YieldCurve,
        a: float,
        sigma: Any,
        sigma_times: Optional[Sequence[float]] = None,
    ) -> None:
        if a <= 0:
            raise ValueError("mean reversion must be positive")
        levels = np.atleast_1d(np.asarray(sigma, float))
        times = np.asarray([] if sigma_times is None else sigma_times, float)
        if times.size != levels.size - 1:
            raise ValueError("need len(sigma_times) == len(sigma) - 1")
        if np.any(levels <= 0) or np.any(np.diff(np.r_[0.0, times]) <= 0):
            raise ValueError("volatilities must be positive and times increasing from 0")
        self.curve = curve
        self.a = float(a)
        self.sigma = levels
        self.sigma_times = times

    def with_sigma(self, sigma: Any, sigma_times: Optional[Sequence[float]] = None) -> "HullWhite":
        return HullWhite(
            self.curve, self.a, sigma, self.sigma_times if sigma_times is None else sigma_times
        )

    # -- deterministic functions ------------------------------------------
    def B(self, t: Any, T: Any) -> Any:
        """:math:`B(t, T) = (1 - e^{-a(T - t)})/a`."""
        return -np.expm1(-self.a * (np.asarray(T, float) - np.asarray(t, float))) / self.a

    def _pieces(self, t: float) -> list[tuple[float, float, float]]:
        """``(u0, u1, sigma)`` for the volatility pieces on ``[0, t]``."""
        edges = np.r_[0.0, self.sigma_times[self.sigma_times < t], t]
        k = np.searchsorted(self.sigma_times, edges[:-1], side="right")
        return [
            (float(u0), float(u1), float(self.sigma[j]))
            for u0, u1, j in zip(edges[:-1], edges[1:], k)
        ]

    def y(self, t: float) -> float:
        """:math:`\\operatorname{Var}(x_t)`."""
        a = self.a
        return sum(
            s * s * (np.exp(-2 * a * (t - u1)) - np.exp(-2 * a * (t - u0))) / (2 * a)
            for u0, u1, s in self._pieces(float(t))
        )

    def forward_mean(self, t: float, T: float) -> float:
        """:math:`E^T[x_t] = -B(t, T)\\,y(t)` under the :math:`T`-forward measure (``t <= T``)."""
        return -float(self.B(t, T)) * self.y(t)

    def phi(self, t: float) -> float:
        """Deterministic part of the short rate: :math:`f(0, t)`."""
        return float(self.curve.instantaneous_forward(float(t)))

    # -- bonds and options ------------------------------------------------
    def discount_bond(self, t: float, T: Any, x: Any) -> Any:
        """:math:`P(t, T)` given :math:`x_t = x` (broadcasts over ``T`` and ``x``)."""
        TT = np.asarray(T, float)
        b = self.B(t, TT)
        ratio = np.asarray(self.curve.discount(TT), float) / float(self.curve.discount(float(t)))
        return ratio * np.exp(-b * np.asarray(x, float) - 0.5 * b * b * self.y(t))

    def short_rate(self, t: float, x: Any) -> Any:
        return self.phi(t) + np.asarray(x, float)

    def bond_option_vol(self, T: float, S: float) -> float:
        """Standard deviation of :math:`\\log P(T, S)`: :math:`B(T, S)\\sqrt{y(T)}`."""
        return float(self.B(T, S) * np.sqrt(self.y(T)))

    def zcb_option(self, T: float, S: float, K: Any, option: str = "call") -> Any:
        """Option expiring at ``T`` on the zero-coupon bond maturing at ``S``, struck at ``K``."""
        if not 0 < T < S:
            raise ValueError("need 0 < T < S")
        pT, pS = float(self.curve.discount(T)), float(self.curve.discount(S))
        v = self.bond_option_vol(T, S)
        KK = np.asarray(K, float)
        d1 = np.log(pS / (KK * pT)) / v + 0.5 * v
        d2 = d1 - v
        if option == "call":
            out = pS * norm.cdf(d1) - KK * pT * norm.cdf(d2)
        elif option == "put":
            out = KK * pT * norm.cdf(-d2) - pS * norm.cdf(-d1)
        else:
            raise ValueError("option must be 'call' or 'put'")
        return float(out) if out.ndim == 0 else out
