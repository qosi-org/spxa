r"""CIR++: the Cox–Ingersoll–Ross model shifted onto a discount curve.

Brigo & Mercurio (2001) set :math:`r_t = x_t + \varphi(t)`, with
:math:`dx = \kappa(\theta - x)\,dt + \sigma\sqrt{x}\,dW` and
:math:`\varphi(t) = f^M(0, t) - f^{\rm CIR}(0, t)`, where

.. math::
    f^{\rm CIR}(0, t) = \frac{2\kappa\theta(e^{ht} - 1)}{2h + (\kappa + h)(e^{ht} - 1)}
        + x_0\,\frac{4h^2 e^{ht}}{\bigl(2h + (\kappa + h)(e^{ht} - 1)\bigr)^2},
    \qquad h = \sqrt{\kappa^2 + 2\sigma^2}.

Then :math:`e^{-\int_0^t\varphi} = P^M(0, t)/P^{\rm CIR}(0, t)`, and the
bond price is

.. math::
    P(t, T) = \Phi(t, T)\,A(T - t)\,e^{-B(T - t)x_t}, \qquad
    \Phi(t, T) = \frac{P^M(0, T)\,P^{\rm CIR}(0, t)}{P^M(0, t)\,P^{\rm CIR}(0, T)},

with the CIR :math:`A, B` of :func:`spxa.pde.reference.cir_bond`. A call on
the bond maturing at :math:`S` is the scaled CIR option

.. math::
    \mathrm{ZBC}(T, S, K) = \frac{P^M(0, T)}{P^{\rm CIR}(0, T)}\,\Phi(T, S)\,
        \mathrm{ZBC}^{\rm CIR}\bigl(T, S, K/\Phi(T, S)\bigr),

and caplets are bond puts. The short rate stays above :math:`\varphi(t)`,
so :meth:`CIRPlusPlus.phi` reports where the fit permits negative rates.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.pde.reference import _cir_ab, cir_bond, cir_bond_option

__all__ = ["CIRPlusPlus"]


class CIRPlusPlus:
    """CIR++ on ``curve`` with CIR parameters ``kappa, theta, sigma`` and ``x0``."""

    def __init__(
        self, curve: YieldCurve, kappa: float, theta: float, sigma: float, x0: float
    ) -> None:
        if min(kappa, theta, sigma) <= 0 or x0 < 0:
            raise ValueError("need kappa, theta, sigma > 0 and x0 >= 0")
        self.curve = curve
        self.kappa, self.theta, self.sigma, self.x0 = (
            float(kappa),
            float(theta),
            float(sigma),
            float(x0),
        )

    @property
    def feller(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma**2

    def _pcir(self, t: float) -> float:
        return float(cir_bond(self.x0, t, self.kappa, self.theta, self.sigma))

    def cir_forward(self, t: Any) -> Any:
        k, h = self.kappa, math.sqrt(self.kappa**2 + 2 * self.sigma**2)
        e = np.expm1(h * np.asarray(t, float))
        den = 2 * h + (k + h) * e
        return 2 * k * self.theta * e / den + self.x0 * 4 * h**2 * (e + 1) / den**2

    def phi(self, t: Any) -> Any:
        """Deterministic shift :math:`f^M(0, t) - f^{\\rm CIR}(0, t)`."""
        return self.curve.instantaneous_forward(t) - self.cir_forward(t)

    def Phi(self, t: float, T: float) -> float:
        crv = self.curve
        return float(crv.discount(T) * self._pcir(t) / (crv.discount(t) * self._pcir(T)))

    def discount_bond(self, t: float, T: Any, x: Any) -> Any:
        """:math:`P(t, T | x_t = x)`; ``T`` may be an array (with scalar ``x``) or ``x`` may be."""
        Ts = np.atleast_1d(np.asarray(T, float))
        out = []
        for Ti in Ts:
            if Ti <= t:
                out.append(np.ones_like(np.asarray(x, float)))
                continue
            lnA, B = _cir_ab(Ti - t, self.kappa, self.theta, self.sigma)
            out.append(self.Phi(t, float(Ti)) * np.exp(lnA - B * np.asarray(x, float)))
        return out[0] if np.ndim(T) == 0 else np.array(out)

    def zcb_option(self, T: float, S: float, K: float, option: str = "call") -> float:
        if not 0 < T < S or K <= 0:
            raise ValueError("need 0 < T < S and K > 0")
        Phi = self.Phi(T, S)
        scale = float(self.curve.discount(T)) / self._pcir(T) * Phi
        base = cir_bond_option(self.x0, K / Phi, T, S, self.kappa, self.theta, self.sigma, option)
        return scale * base

    def caplet(self, caplet: Any, projection: Optional[YieldCurve] = None) -> float:
        """Caplet or floorlet as :math:`K^*` bond puts (calls), with ``K* = 1 + tau(K - basis)``."""
        c = caplet.coupon
        crv = self.curve
        proj = projection if projection is not None else crv
        s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
        tau = c.accrual
        Q0 = float(crv.discount(s) / crv.discount(e))
        Kstar = 1.0 + tau * (caplet.strike - (caplet.forward(proj) - (Q0 - 1.0) / tau))
        if Kstar <= 0:
            raise ValueError("strike below the basis: K* <= 0")
        option = "put" if caplet.kind == "cap" else "call"
        und = Kstar * self.zcb_option(s, e, 1.0 / Kstar, option)
        return float(c.notional * crv.discount(p) / crv.discount(e) * und)

    def cap(self, cap: Any, projection: Optional[YieldCurve] = None) -> float:
        return float(sum(self.caplet(c, projection) for c in cap.caplets))

    # -- swaptions --------------------------------------------------------
    def forward_measure_law(self, T: float) -> tuple[float, float, float]:
        r"""``(scale, df, nc)`` with :math:`x_T = \text{scale}\cdot\chi^2(\text{df}, \text{nc})` under :math:`Q^T`.

        With :math:`\rho = 2h/(\sigma^2(e^{hT} - 1))` and
        :math:`\psi = (\kappa + h)/\sigma^2`, the scale is
        :math:`1/(2(\rho + \psi))`, the degrees of freedom are
        :math:`4\kappa\theta/\sigma^2`, and the non-centrality is
        :math:`2\rho^2 x_0 e^{hT}/(\rho + \psi)` (Brigo & Mercurio, §3.2.3).
        """
        k, s = self.kappa, self.sigma
        h = math.sqrt(k**2 + 2 * s**2)
        rho = 2 * h / (s**2 * math.expm1(h * T))
        psi = (k + h) / s**2
        return (
            1 / (2 * (rho + psi)),
            4 * k * self.theta / s**2,
            2 * rho**2 * self.x0 * math.exp(h * T) / (rho + psi),
        )

    def swaption(
        self, swaption: Any, projection: Optional[YieldCurve] = None, method: str = "jamshidian"
    ) -> float:
        r"""European physically settled swaption.

        Bond prices decrease in :math:`x \ge 0`. So when the swap amounts
        change sign once, Jamshidian's decomposition applies with the root
        :math:`x^*` of :math:`V(x) = \sum_i a_i P(T, t_i | x)`. If
        :math:`V` keeps one sign on :math:`[0, \infty)`, the option is the
        swap or nothing. ``method='integral'`` integrates :math:`V^+`
        against the :math:`Q^T` law of :meth:`forward_measure_law`.
        """
        from scipy.integrate import quad
        from scipy.optimize import brentq
        from scipy.stats import ncx2

        from spxa.rates.model_cashflows import swap_cashflows

        if method not in ("jamshidian", "integral"):
            raise ValueError("method must be 'jamshidian' or 'integral'")
        if getattr(swaption, "settlement", "physical") != "physical":
            raise ValueError("only physical settlement is supported")
        crv = self.curve
        T = float(crv.time(swaption.expiry))
        cf = swap_cashflows(swaption.swap, crv, projection)
        if T <= 0:
            return max(cf.pv(crv), 0.0)
        live = cf.times > T + 1e-12
        times, amounts = cf.times[live], cf.amounts[live]
        const = float(cf.amounts[~live].sum())
        scale, df, nc = self.forward_measure_law(T)
        hi = scale * float(ncx2.ppf(1 - 1e-15, df, nc))

        def V(x: float) -> float:
            return const + float(np.dot(amounts, self.discount_bond(T, times, x)))

        root = (
            brentq(V, 0.0, hi, xtol=1e-16, rtol=1e-15, maxiter=200) if V(0.0) * V(hi) < 0 else None
        )
        pT = float(crv.discount(T))
        if method == "jamshidian":
            if root is None:
                return max(cf.pv(crv), 0.0) if V(0.0) > 0 and V(hi) > 0 else 0.0
            K = self.discount_bond(T, times, root)
            increasing = V(root + 1e-3 * hi) > 0
            total = 0.0
            for t_i, a_i, k_i in zip(times, amounts, K):
                if increasing:
                    total -= a_i * self.zcb_option(T, float(t_i), float(k_i), "put")
                else:
                    total += a_i * self.zcb_option(T, float(t_i), float(k_i), "call")
            return float(total)

        def f(x: float) -> float:
            return max(V(x), 0.0) * float(ncx2.pdf(x / scale, df, nc)) / scale

        pts = [0.0, root, hi] if root is not None else [0.0, hi]
        val = sum(
            quad(f, a, b, epsabs=0.0, epsrel=1e-12, limit=400)[0] for a, b in zip(pts[:-1], pts[1:])
        )
        return float(pT * val)
