r"""Gaussian Heath–Jarrow–Morton models.

Under the risk-neutral measure the instantaneous forward rates follow

.. math::
    df(t, T) = \sigma(t, T)\cdot\Sigma(t, T)\,dt + \sigma(t, T)\cdot dW_t,
    \qquad \Sigma(t, T) = \int_t^T \sigma(t, s)\,ds,

with a deterministic vector volatility :math:`\sigma(t, T)` and
independent Brownian motions :math:`W`. The drift is the HJM
no-arbitrage condition, so the model fits :math:`f(0, \cdot)` from the
curve by construction. Bonds are lognormal:
:math:`d\log P(t, T) = (r_t - \tfrac12|\Sigma(t, T)|^2)\,dt - \Sigma(t, T)\cdot dW_t`.
Under the :math:`T`-forward measure the forward bond :math:`P(t, S)/P(t, T)`
is a martingale with log-variance

.. math:: v^2(T, S) = \int_0^T \bigl|\Sigma(u, S) - \Sigma(u, T)\bigr|^2\,du

so zero-bond options and caplets are Black formulas. Here
:math:`\Sigma` is computed by Gauss–Legendre quadrature for any
volatility function. :func:`exponential_vol` gives Hull–White
(:math:`\sigma e^{-a(T-t)}`), and :func:`hump_vol` the hump
:math:`(\alpha + \beta(T - t))e^{-\gamma(T-t)}`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve

__all__ = ["GaussianHJM", "exponential_vol", "hump_vol"]

VolFn = Callable[[Any, Any], Any]


def exponential_vol(sigma: float, a: float) -> VolFn:
    """:math:`\\sigma e^{-a(T - t)}`: Hull–White in HJM form."""
    return lambda t, T: sigma * np.exp(-a * (np.asarray(T, float) - t))


def hump_vol(alpha: float, beta: float, gamma: float) -> VolFn:
    """:math:`(\\alpha + \\beta(T - t))e^{-\\gamma(T - t)}`, humped in maturity for ``beta > 0``."""

    def f(t: Any, T: Any) -> Any:
        tau = np.asarray(T, float) - t
        return (alpha + beta * tau) * np.exp(-gamma * tau)

    return f


class GaussianHJM:
    """Gaussian HJM on ``curve`` with independent factor volatilities ``vols[i](t, T)``."""

    def __init__(
        self,
        curve: YieldCurve,
        vols: Sequence[VolFn],
        n_quad: int = 48,
        breakpoints: Sequence[float] = (),
    ) -> None:
        """``breakpoints`` are calendar times where the vols jump; time integrals are split there."""
        if len(vols) < 1:
            raise ValueError("need at least one volatility factor")
        self.breakpoints = np.sort(np.asarray(breakpoints, float))
        self.curve = curve
        self.vols = list(vols)
        self.n_factors = len(self.vols)
        self._x, self._w = np.polynomial.legendre.leggauss(int(n_quad))

    def sigma(self, t: float, T: Any) -> np.ndarray:
        """Volatility vector(s), shape ``(n_factors,) + shape(T)``."""
        return np.array([np.broadcast_to(v(t, T), np.shape(T)) for v in self.vols], float)

    def Sigma(self, t: float, T: Any) -> np.ndarray:
        r""":math:`\Sigma(t, T) = \int_t^T \sigma(t, s)\,ds`, shape ``(n_factors,) + shape(T)``."""
        T_ = np.asarray(T, float)
        half = 0.5 * (T_ - t)
        s = t + half[..., None] * (self._x + 1.0)  # (..., n)
        vals = self.sigma(t, s)  # (k, ..., n)
        return np.asarray(np.sum(vals * self._w, axis=-1) * half)

    def drift(self, t: float, T: Any) -> np.ndarray:
        """HJM drift :math:`\\sigma(t, T)\\cdot\\Sigma(t, T)`."""
        return np.asarray(np.sum(self.sigma(t, T) * self.Sigma(t, T), axis=0))

    def forward_bond_variance(self, T: float, s: float, e: float) -> float:
        """Log-variance of :math:`P(T, e)/P(T, s)`: :math:`\\int_0^T |\\Sigma(u, e) - \\Sigma(u, s)|^2 du`."""
        if T <= 0:
            return 0.0
        edges = np.r_[0.0, self.breakpoints[(self.breakpoints > 0) & (self.breakpoints < T)], T]
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            half = 0.5 * (b - a)
            for xi, wi in zip(self._x, self._w):
                u = a + half * (xi + 1.0)
                d = self.Sigma(u, e) - self.Sigma(u, s)
                total += wi * half * float(np.dot(d, d))
        return total

    def zcb_option(self, T: float, S: float, K: Any, option: str = "call") -> Any:
        from spxa.market.black import black_price

        if not 0 < T < S:
            raise ValueError("need 0 < T < S")
        pT, pS = float(self.curve.discount(T)), float(self.curve.discount(S))
        v = math.sqrt(self.forward_bond_variance(T, T, S))
        return black_price(pS / pT, K, 1.0, v, pT, option)

    def caplet(self, caplet: Any, projection: Optional[YieldCurve] = None) -> float:
        """Caplet or floorlet as a bond option, with the projection basis as a deterministic spread."""
        from spxa.market.black import black_price

        c = caplet.coupon
        crv = self.curve
        proj = projection if projection is not None else crv
        tf = float(crv.time(c.fixing))
        s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
        tau = c.accrual
        Q0 = float(crv.discount(s) / crv.discount(e))
        Kstar = 1.0 + tau * (caplet.strike - (caplet.forward(proj) - (Q0 - 1.0) / tau))
        v = math.sqrt(self.forward_bond_variance(tf, s, e))
        option = "call" if caplet.kind == "cap" else "put"
        if Kstar <= 0 or v == 0.0:
            und = max(Q0 - Kstar, 0.0) if option == "call" else max(Kstar - Q0, 0.0)
        else:
            und = float(black_price(Q0, Kstar, 1.0, v, 1.0, option))
        return float(c.notional * crv.discount(p) * und)

    def cap(self, cap: Any, projection: Optional[YieldCurve] = None) -> float:
        return float(sum(self.caplet(c, projection) for c in cap.caplets))

    @classmethod
    def from_hull_white(cls, model: Any, n_quad: int = 48) -> "GaussianHJM":
        """The HJM form :math:`\\sigma(t)e^{-a(T-t)}` of a (piecewise-constant) Hull–White model."""
        knots = np.asarray(model.sigma_times, float)
        levels = np.asarray(model.sigma, float)
        a = float(model.a)

        def vol(t: Any, T: Any) -> Any:
            lvl = levels[np.searchsorted(knots, np.asarray(t, float), side="right")]
            return lvl * np.exp(-a * (np.asarray(T, float) - t))

        return cls(model.curve, [vol], n_quad, knots)
