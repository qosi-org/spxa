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

    def discount_bond(self, t: float, T: float, x: Any) -> Any:
        if T <= t:
            return np.ones_like(np.asarray(x, float))
        lnA, B = _cir_ab(T - t, self.kappa, self.theta, self.sigma)
        return self.Phi(t, T) * np.exp(lnA - B * np.asarray(x, float))

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
