r"""The G2++ two-factor Gaussian model fitted to a discount curve.

Following Brigo & Mercurio (*Interest Rate Models*, §4.2), the short rate
is :math:`r_t = x_t + y_t + \varphi(t)` with

.. math::
    dx = -a\,x\,dt + \sigma\,dW_1, \qquad dy = -b\,y\,dt + \eta\,dW_2, \qquad
    d\langle W_1, W_2\rangle = \rho\,dt, \qquad x_0 = y_0 = 0.

With :math:`B_c(t, T) = (1 - e^{-c(T - t)})/c` and

.. math::
    V(t, T) = \operatorname{Var}\Bigl(\int_t^T (x_u + y_u)\,du \Bigm| \mathcal F_t\Bigr)
    = \int_t^T \bigl(\sigma^2 B_a(u, T)^2 + \eta^2 B_b(u, T)^2 + 2\rho\sigma\eta B_a(u, T) B_b(u, T)\bigr)\,du,

the curve-consistent bond price is

.. math::
    P(t, T) = \frac{P(0, T)}{P(0, t)}\exp\Bigl(\tfrac12\bigl[V(t, T) - V(0, T) + V(0, t)\bigr]
    - B_a(t, T)\,x_t - B_b(t, T)\,y_t\Bigr),

and :math:`\varphi(t) = f(0, t) + \tfrac12\partial_t V(0, t)`. Under the
:math:`T`-forward measure, :math:`(x_t, y_t)` is Gaussian with the
covariance of the risk-neutral measure and means

.. math:: \mu_x = -\int_0^t e^{-a(t - u)}\bigl(\sigma^2 B_a(u, T) + \rho\sigma\eta B_b(u, T)\bigr)\,du,

and symmetrically for :math:`\mu_y`. :math:`\log P(T, S)` is Gaussian, so
zero-coupon bond options and caplets have Black-type closed forms.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.stats import norm

from spxa.market.curves import YieldCurve

__all__ = ["G2PP"]


def _B(c: float, tau: Any) -> Any:
    return -np.expm1(-c * np.asarray(tau, float)) / c


class G2PP:
    """G2++ on ``curve`` with parameters ``a, sigma, b, eta, rho``."""

    def __init__(
        self, curve: YieldCurve, a: float, sigma: float, b: float, eta: float, rho: float
    ) -> None:
        if min(a, b, sigma, eta) <= 0 or not -1 < rho < 1:
            raise ValueError("need a, b, sigma, eta > 0 and |rho| < 1")
        if abs(a - b) < 1e-10:
            raise ValueError("a and b must differ (the factors would not be identified)")
        self.curve = curve
        self.a, self.sigma, self.b, self.eta, self.rho = (
            float(a),
            float(sigma),
            float(b),
            float(eta),
            float(rho),
        )

    # -- deterministic functions ------------------------------------------
    def V(self, t: float, T: Any) -> Any:
        """:math:`V(t, T)`, the conditional variance of :math:`\\int_t^T (x + y)`."""
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho
        tau = np.asarray(T, float) - t

        def term(c1: float, c2: float) -> Any:
            # int_0^tau B_{c1}(u) B_{c2}(u) du with B_c(u) = (1 - e^{-c u})/c
            return (tau - _B(c1, tau) - _B(c2, tau) + _B(c1 + c2, tau)) / (c1 * c2)

        return s * s * term(a, a) + e * e * term(b, b) + 2 * r * s * e * term(a, b)

    def covariance(self, t: float) -> np.ndarray:
        """Covariance matrix of :math:`(x_t, y_t)`."""
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho
        vx = s * s * _B(2 * a, t)
        vy = e * e * _B(2 * b, t)
        cxy = r * s * e * _B(a + b, t)
        return np.array([[vx, cxy], [cxy, vy]])

    def forward_mean(self, t: float, T: float) -> np.ndarray:
        """Mean of :math:`(x_t, y_t)` under the :math:`T`-forward measure (``t <= T``)."""
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho

        def integral(k: float, c: float) -> float:
            # int_0^t e^{-k(t-u)} B_c(u, T) du
            return float((_B(k, t) - np.exp(-c * (T - t)) * _B(k + c, t)) / c)

        mx = -(s * s * integral(a, a) + r * s * e * integral(a, b))
        my = -(e * e * integral(b, b) + r * s * e * integral(b, a))
        return np.array([mx, my])

    def phi(self, t: float) -> float:
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho
        Ba, Bb = _B(a, t), _B(b, t)
        f = float(self.curve.instantaneous_forward(float(t)))
        return f + 0.5 * s * s * Ba * Ba + 0.5 * e * e * Bb * Bb + r * s * e * Ba * Bb

    # -- bonds and options ------------------------------------------------
    def discount_bond(self, t: float, T: Any, x: Any, y: Any) -> Any:
        """:math:`P(t, T)` given :math:`(x_t, y_t)` (broadcasts)."""
        TT = np.asarray(T, float)
        crv = self.curve
        ratio = np.asarray(crv.discount(TT), float) / float(crv.discount(float(t)))
        adj = 0.5 * (self.V(t, TT) - self.V(0.0, TT) + self.V(0.0, t))
        return ratio * np.exp(
            adj
            - _B(self.a, TT - t) * np.asarray(x, float)
            - _B(self.b, TT - t) * np.asarray(y, float)
        )

    def _log_ratio_sd(self, t: float, w: np.ndarray) -> float:
        """Standard deviation of :math:`w_x x_t + w_y y_t`."""
        return float(np.sqrt(w @ self.covariance(t) @ w))

    def bond_option_vol(self, T: float, S: float) -> float:
        """Standard deviation of :math:`\\log P(T, S)`."""
        return self._log_ratio_sd(T, np.array([_B(self.a, S - T), _B(self.b, S - T)]))

    def zcb_option(self, T: float, S: float, K: Any, option: str = "call") -> Any:
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

    def caplet(self, caplet: Any, projection: Optional[YieldCurve] = None) -> float:
        """Caplet or floorlet: as :meth:`spxa.rates.HullWhite.caplet`, with the two-factor log-variance."""
        from spxa.market.black import black_price

        c = caplet.coupon
        crv = self.curve
        proj = projection if projection is not None else crv
        tf = float(crv.time(c.fixing))
        s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
        tau = c.accrual
        Q0 = float(crv.discount(s) / crv.discount(e))
        Kstar = 1.0 + tau * (caplet.strike - (caplet.forward(proj) - (Q0 - 1.0) / tau))
        w = np.array(
            [_B(self.a, e - tf) - _B(self.a, s - tf), _B(self.b, e - tf) - _B(self.b, s - tf)]
        )
        v = self._log_ratio_sd(tf, w) if tf > 0 else 0.0
        option = "call" if caplet.kind == "cap" else "put"
        if Kstar <= 0 or v == 0.0:
            und = max(Q0 - Kstar, 0.0) if option == "call" else max(Kstar - Q0, 0.0)
        else:
            und = float(black_price(Q0, Kstar, 1.0, v, 1.0, option))
        return float(c.notional * crv.discount(p) * und)

    def cap(self, cap: Any, projection: Optional[YieldCurve] = None) -> float:
        return float(sum(self.caplet(c, projection) for c in cap.caplets))

    # -- swaptions --------------------------------------------------------
    def option_on_cashflows(self, T: float, times: Any, amounts: Any, const: float = 0.0) -> float:
        r"""Value of :math:`\bigl(c + \sum_i a_i P(T, t_i)\bigr)^+` paid at ``T``.

        Brigo & Mercurio's one-dimensional formula (§4.2.3). Under the
        :math:`T`-forward measure, condition on :math:`x_T`. Then
        :math:`y_T` is Gaussian, and for amounts that change sign once the
        payoff is monotone in :math:`y` with a single root :math:`\bar y(x)`.
        The conditional expectation is a sum of normal distribution
        functions, and the remaining integral over :math:`x` is done by
        adaptive quadrature.
        """
        from scipy.integrate import quad
        from scipy.optimize import brentq

        t = np.asarray(times, float)
        a_ = np.asarray(amounts, float)
        crv = self.curve
        pT = float(crv.discount(T))
        A = (
            np.asarray(crv.discount(t), float)
            / pT
            * np.exp(0.5 * (self.V(T, t) - self.V(0.0, t) + self.V(0.0, T)))
        )
        Ba, Bb = _B(self.a, t - T), _B(self.b, t - T)
        mu = self.forward_mean(T, T)
        C = self.covariance(T)
        sx = np.sqrt(C[0, 0])
        beta = C[0, 1] / C[0, 0]
        s = np.sqrt(C[1, 1] - C[0, 1] ** 2 / C[0, 0])

        def conditional(x: float) -> float:
            c = a_ * A * np.exp(-Ba * x)
            m = mu[1] + beta * (x - mu[0])
            g = np.exp(-Bb * m + 0.5 * Bb * Bb * s * s)  # E[e^{-Bb y} | x] / e^{0}

            def V(y: float) -> float:
                return const + float(np.dot(c, np.exp(-Bb * y)))

            lo, hi = m - 40 * s, m + 40 * s
            vlo, vhi = V(lo), V(hi)
            if vlo * vhi > 0:
                return max(const + float(np.dot(c, g)), 0.0)
            yb = brentq(V, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
            d = (yb - m) / s
            dd = (yb - m + Bb * s * s) / s
            if vhi > 0:  # V positive above the root
                return const * norm.sf(d) + float(np.dot(c * g, norm.sf(dd)))
            return const * norm.cdf(d) + float(np.dot(c * g, norm.cdf(dd)))

        def integrand(z: float) -> float:
            return conditional(mu[0] + sx * z) * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

        val = quad(integrand, -10, 10, epsabs=0.0, epsrel=1e-11, limit=400)[0]
        return float(pT * val)

    def swaption(self, swaption: Any, projection: Optional[YieldCurve] = None) -> float:
        """European physically settled swaption (deterministic basis to ``projection``)."""
        from spxa.rates.model_cashflows import swap_cashflows

        if getattr(swaption, "settlement", "physical") != "physical":
            raise ValueError("only physical settlement is supported")
        crv = self.curve
        T = float(crv.time(swaption.expiry))
        cf = swap_cashflows(swaption.swap, crv, projection)
        live = cf.times > T + 1e-12
        return self.option_on_cashflows(
            T, cf.times[live], cf.amounts[live], float(cf.amounts[~live].sum())
        )
