r"""CMS coupons, caplets and floorlets by static replication.

A coupon paying the swap rate :math:`S_T` (fixed at :math:`T`) at
:math:`T_p` is worth :math:`V = A_0\,E^A[S_T\,\alpha(S_T)]` under the
annuity measure, where :math:`\alpha(S) \approx P(T, T_p)/A(T)` is an
*annuity mapping* function (terminal swap-rate model). Two mappings are
provided:

* ``'standard'`` (Hagan, *Convexity conundrums*, 2003): with :math:`n`
  fixed periods at frequency :math:`m` and the payment
  :math:`\delta` periods after the start,

  .. math:: G(S) = \frac{S\,(1 + S/m)^{-\delta}}{1 - (1 + S/m)^{-n}}, \qquad
            \alpha(S) = \frac{P(T_p)}{A_0}\,\frac{G(S)}{G(S_0)};

* ``'linear'`` (linear TSR, Andersen–Piterbarg):
  :math:`\alpha(S) = P(T_p)/A_0 + a\,(S - S_0)`, with slope
  :math:`a = \frac{P(T_p)}{A_0}G'(S_0)/G(S_0)` taken from the standard
  mapping unless given. With this mapping :math:`E^A[\alpha(S_T)] = P(T_p)/A_0`
  holds exactly, so CMS cap-floor parity is exact.

For a twice-differentiable payoff :math:`f`, Carr–Madan gives

.. math:: E^A[f(S_T)] = f(S_0) + \int_{-\infty}^{S_0} f''(K)\,P(K)\,dK + \int_{S_0}^{\infty} f''(K)\,C(K)\,dK,

where :math:`C, P` are undiscounted annuity-measure (swaption) prices.
Kinks contribute point terms: for the caplet
:math:`f = (S - K)^+\alpha(S)` this is :math:`\alpha(K)C(K)`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad

__all__ = ["CMSReplication"]

Pricer = Callable[[float, str], float]


class CMSReplication:
    """Static replication of CMS payoffs on one swap rate.

    Parameters
    ----------
    S0, T
        Forward swap rate and fixing time.
    annuity, discount_pay
        :math:`A_0` and :math:`P(0, T_p)`.
    price
        ``price(K, option)``: undiscounted annuity-measure swaption price
        (e.g. Black or SABR with unit annuity).
    n, m, delta
        Fixed periods, frequency and payment delay in periods (standard mapping).
    mapping, slope
        ``'standard'`` or ``'linear'``; ``slope`` overrides the linear TSR slope.
    lower, upper
        Integration range for the strike (the lower end is the rate floor,
        e.g. 0 or ``-shift``).
    """

    def __init__(
        self,
        S0: float,
        T: float,
        annuity: float,
        discount_pay: float,
        price: Pricer,
        n: int,
        m: int = 1,
        delta: float = 0.0,
        mapping: str = "standard",
        slope: Optional[float] = None,
        lower: float = 0.0,
        upper: Optional[float] = None,
    ) -> None:
        if mapping not in ("standard", "linear"):
            raise ValueError("mapping must be 'standard' or 'linear'")
        self.S0, self.T, self.A0, self.P = float(S0), float(T), float(annuity), float(discount_pay)
        self.price, self.n, self.m, self.delta = price, int(n), int(m), float(delta)
        self.mapping = mapping
        self.lower = float(lower)
        self.upper = float(upper) if upper is not None else max(10 * abs(S0), 0.5)
        g0, dg0 = self._G(self.S0), self._dG(self.S0)
        self.slope = self.P / self.A0 * dg0 / g0 if slope is None else float(slope)

    def _G(self, S: float) -> float:
        x = 1.0 + S / self.m
        if abs(S) < 1e-12:
            return self.m / self.n * x ** (-self.delta)
        return S * x ** (-self.delta) / (-np.expm1(-self.n * np.log1p(S / self.m)))

    def _dG(self, S: float) -> float:
        h = 1e-5 * max(abs(S), 1e-3)
        return (self._G(S + h) - self._G(S - h)) / (2 * h)

    def alpha(self, S: float) -> float:
        """Annuity mapping :math:`\\alpha(S)`."""
        if self.mapping == "linear":
            return self.P / self.A0 + self.slope * (S - self.S0)
        return self.P / self.A0 * self._G(S) / self._G(self.S0)

    def _d(self, f: Callable[[float], float], S: float, k: int) -> float:
        h = 1e-4 * max(abs(S), 1e-3)
        if k == 1:
            return (f(S + h) - f(S - h)) / (2 * h)
        return (f(S + h) - 2 * f(S) + f(S - h)) / (h * h)

    def _integral(self, f2: Callable[[float], float], a: float, b: float, option: str) -> float:
        if b <= a:
            return 0.0
        val, _ = quad(
            lambda k: f2(k) * self.price(k, option), a, b, limit=400, epsabs=1e-13, epsrel=1e-9
        )
        return float(val)

    def expectation(self, f: Callable[[float], float]) -> float:
        """:math:`E^A[f(S_T)]` for a smooth payoff."""

        def f2(k: float) -> float:
            return self._d(f, k, 2)

        return (
            f(self.S0)
            + self._integral(f2, self.lower, self.S0, "put")
            + self._integral(f2, self.S0, self.upper, "call")
        )

    def coupon(self) -> float:
        """Value of the CMS coupon paying :math:`S_T` at :math:`T_p` (unit notional and accrual)."""
        return self.A0 * self.expectation(lambda s: s * self.alpha(s))

    def convexity_adjustment(self) -> float:
        """:math:`E^{T_p}[S_T] - S_0`."""
        return self.coupon() / self.P - self.S0

    def caplet(self, K: float) -> float:
        """CMS caplet :math:`(S_T - K)^+` paid at :math:`T_p`."""

        def f2(x: float) -> float:
            return 2 * self._d(self.alpha, x, 1) + (x - K) * self._d(self.alpha, x, 2)

        return self.A0 * (
            self.alpha(K) * self.price(K, "call") + self._integral(f2, K, self.upper, "call")
        )

    def floorlet(self, K: float) -> float:
        """CMS floorlet :math:`(K - S_T)^+` paid at :math:`T_p`."""

        def f2(x: float) -> float:
            return -2 * self._d(self.alpha, x, 1) + (K - x) * self._d(self.alpha, x, 2)

        return self.A0 * (
            self.alpha(K) * self.price(K, "put") + self._integral(f2, self.lower, K, "put")
        )
