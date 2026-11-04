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

    # -- caps and swaptions -----------------------------------------------
    def caplet(self, caplet: Any, projection: Optional[YieldCurve] = None) -> float:
        r"""Caplet or floorlet (a :class:`~spxa.rates.capfloor.Caplet`).

        With fixing time :math:`t_f` and period :math:`[s, e]`, the ratio
        :math:`Q = P(t_f, s)/P(t_f, e)` is lognormal under the
        :math:`e`-forward measure with log-variance
        :math:`(B(t_f, e) - B(t_f, s))^2 y(t_f)`. The coupon pays
        :math:`N\tau(L - K)^+ = N(Q - K^*)^+` with
        :math:`K^* = 1 + \tau(K - b)`, where the deterministic basis
        :math:`b` reconciles the discount-curve forward with the projection
        forward. The payment-date ratio :math:`P(\cdot, p)/P(\cdot, e)` is
        frozen at today's value.
        """
        from spxa.market.black import black_price

        c = caplet.coupon
        proj = projection if projection is not None else self.curve
        crv = self.curve
        tf = float(crv.time(c.fixing))
        s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
        tau = c.accrual
        Q0 = float(crv.discount(s) / crv.discount(e))
        b = caplet.forward(proj) - (Q0 - 1.0) / tau
        Kstar = 1.0 + tau * (caplet.strike - b)
        v = float(abs(self.B(tf, e) - self.B(tf, s)) * np.sqrt(self.y(tf))) if tf > 0 else 0.0
        option = "call" if caplet.kind == "cap" else "put"
        if Kstar <= 0:
            und = max(Q0 - Kstar, 0.0) if option == "call" else 0.0
        elif v == 0.0:
            und = max(Q0 - Kstar, 0.0) if option == "call" else max(Kstar - Q0, 0.0)
        else:
            und = float(black_price(Q0, Kstar, 1.0, v, 1.0, option))
        return float(c.notional * crv.discount(p) * und)

    def cap(self, cap: Any, projection: Optional[YieldCurve] = None) -> float:
        """Cap or floor (a :class:`~spxa.rates.capfloor.CapFloor`): the sum of its optionlets."""
        return float(sum(self.caplet(c, projection) for c in cap.caplets))

    def swaption(
        self, swaption: Any, projection: Optional[YieldCurve] = None, method: str = "jamshidian"
    ) -> float:
        r"""European swaption (a physically settled :class:`~spxa.rates.swaptions.Swaption`).

        At expiry :math:`T` the swap is worth :math:`V(x) = \sum_i a_i P(T, t_i | x)`
        (:func:`~spxa.rates.model_cashflows.swap_cashflows`), and under the
        :math:`T`-forward measure :math:`x_T \sim N(0, y(T))`. If the
        amounts change sign once, :math:`V` is monotone in :math:`x`, and
        Jamshidian's decomposition writes :math:`V^+` as a portfolio of
        zero-coupon bond options struck at :math:`P(T, t_i | x^*)`, where
        :math:`V(x^*) = 0`. ``method='integral'`` integrates
        :math:`V^+` against the Gaussian instead (and is used automatically
        when the sign condition fails).
        """
        from scipy.integrate import quad
        from scipy.optimize import brentq

        from spxa.rates.model_cashflows import swap_cashflows

        if getattr(swaption, "settlement", "physical") != "physical":
            raise ValueError("only physical settlement is supported")
        crv = self.curve
        T = float(crv.time(swaption.expiry))
        cf = swap_cashflows(swaption.swap, crv, projection)
        if T <= 0:
            return max(cf.pv(crv), 0.0)
        live = cf.times > T + 1e-12
        times, amounts = cf.times[live], cf.amounts[live]
        const = float(cf.amounts[~live].sum())  # flows at T: P(T, T) = 1
        sd = np.sqrt(self.y(T))
        pT = float(crv.discount(T))

        def V(x: float) -> float:
            return const + float(np.dot(amounts, self.discount_bond(T, times, x)))

        lo, hi = -12 * sd, 12 * sd
        root = brentq(V, lo, hi, xtol=1e-16, rtol=1e-15, maxiter=200) if V(lo) * V(hi) < 0 else None
        one_sign = (
            amounts.size > 1
            and (np.all(amounts[1:] < 0) or np.all(amounts[1:] > 0))
            and np.sign(amounts[0]) != np.sign(amounts[1])
        )
        if method == "jamshidian" and root is not None and one_sign and const == 0.0:
            K = self.discount_bond(T, times, root)
            increasing = V(root + sd) > 0
            total = 0.0
            for t_i, a_i, k_i in zip(times, amounts, K):
                if increasing:
                    total -= a_i * self.zcb_option(T, t_i, k_i, "put")
                else:
                    total += a_i * self.zcb_option(T, t_i, k_i, "call")
            return float(total)
        if method not in ("jamshidian", "integral"):
            raise ValueError("method must be 'jamshidian' or 'integral'")

        def f(x: float) -> float:
            return max(V(x), 0.0) * np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

        pts = [lo, root, hi] if root is not None else [lo, hi]
        val = sum(
            quad(f, a, b, epsabs=0.0, epsrel=1e-12, limit=200)[0] for a, b in zip(pts[:-1], pts[1:])
        )
        return float(pT * val)
