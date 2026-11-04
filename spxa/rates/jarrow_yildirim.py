r"""The Jarrow–Yildirim model of nominal rates, real rates and inflation.

Under the nominal risk-neutral measure, following Jarrow & Yildirim (2003) and
Mercurio (2005):

.. math::

    dn_t &= (\theta_n(t) - a_n n_t)\,dt + \sigma_n\,dW^n_t, \\
    dr_t &= (\theta_r(t) - \rho_{rI}\sigma_r\sigma_I - a_r r_t)\,dt + \sigma_r\,dW^r_t, \\
    \frac{dI_t}{I_t} &= (n_t - r_t)\,dt + \sigma_I\,dW^I_t,

with correlations :math:`\rho_{nr}, \rho_{nI}, \rho_{rI}`. Both short rates are
Hull–White: :math:`\theta_n` fits the nominal curve, and :math:`\theta_r` fits
the real curve under the real measure. The shift
:math:`-\rho_{rI}\sigma_r\sigma_I` is the change of numeraire from real to
nominal money.

The forward index :math:`I(0)P_r(0,T)/P_n(0,T)` does not depend on the model,
so zero-coupon inflation swaps are priced off the curves. A year-on-year
swaplet paying :math:`I(T_2)/I(T_1) - 1` at :math:`T_2` is worth
:math:`P_n(0,T_1)\,E^{T_1}_n[P_r(T_1,T_2)] - P_n(0,T_2)`. The real bond is
lognormal, and moving from the real to the nominal :math:`T_1`-forward measure
shifts the mean of :math:`r_{T_1}`. This gives the convexity factor
:math:`e^{C(T_1,T_2)}`:

.. math::

    C = B_r(T_1,T_2)\Bigl[\rho_{rI}\sigma_r\sigma_I B_r(0,T_1)
        + \rho_{nr}\sigma_n\sigma_r \int_0^{T_1} e^{-a_r u}B_n(0,u)\,du
        - \tfrac12\sigma_r^2 B_r(0,T_1)^2\Bigr],

with :math:`B_x(t,T) = (1 - e^{-a_x(T-t)})/a_x`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spxa.market.curves import YieldCurve

__all__ = ["JarrowYildirim"]


def _B(a: float, tau: Any) -> Any:
    return -np.expm1(-a * np.asarray(tau, float)) / a


class JarrowYildirim:
    """Jarrow–Yildirim model on a nominal and a real discount curve; ``cpi0`` is today's index."""

    def __init__(
        self,
        nominal: YieldCurve,
        real: YieldCurve,
        cpi0: float,
        a_n: float,
        sigma_n: float,
        a_r: float,
        sigma_r: float,
        sigma_i: float,
        rho_nr: float = 0.0,
        rho_ni: float = 0.0,
        rho_ri: float = 0.0,
    ) -> None:
        if a_n <= 0 or a_r <= 0:
            raise ValueError("mean reversions must be positive")
        if min(sigma_n, sigma_r, sigma_i) < 0 or cpi0 <= 0:
            raise ValueError("volatilities must be non-negative and the index positive")
        corr = np.array([[1.0, rho_nr, rho_ni], [rho_nr, 1.0, rho_ri], [rho_ni, rho_ri, 1.0]])
        if np.linalg.eigvalsh(corr)[0] < -1e-12:
            raise ValueError("the correlation matrix is not positive semidefinite")
        self.nominal, self.real, self.cpi0 = nominal, real, float(cpi0)
        self.a_n, self.sigma_n, self.a_r, self.sigma_r = (
            float(a_n),
            float(sigma_n),
            float(a_r),
            float(sigma_r),
        )
        self.sigma_i = float(sigma_i)
        self.rho_nr, self.rho_ni, self.rho_ri = float(rho_nr), float(rho_ni), float(rho_ri)
        self.correlation = corr

    # -- curves ---------------------------------------------------------------
    def _t(self, x: Any) -> float:
        return float(self.nominal.time(x))

    def forward_index(self, T: Any) -> float:
        """Model-independent forward CPI :math:`I(0)P_r(0,T)/P_n(0,T)`."""
        t = self._t(T)
        return self.cpi0 * float(self.real.discount(t)) / float(self.nominal.discount(t))

    def zero_coupon_swap_rate(self, T: Any) -> float:
        """Fair zero-coupon inflation swap rate for maturity ``T`` (in years)."""
        t = self._t(T)
        return (self.forward_index(t) / self.cpi0) ** (1.0 / t) - 1.0

    # -- year on year ---------------------------------------------------------
    def yoy_convexity(self, T1: Any, T2: Any) -> float:
        """The exponent :math:`C(T_1, T_2)` of the year-on-year convexity factor."""
        t1, t2 = self._t(T1), self._t(T2)
        if not 0 <= t1 < t2:
            raise ValueError("need 0 <= T1 < T2")
        an, ar = self.a_n, self.a_r
        br = float(_B(ar, t1))
        j = (br - float(_B(an + ar, t1))) / an  # int_0^T1 e^{-a_r u} B_n(0,u) du
        inner = (
            self.rho_ri * self.sigma_r * self.sigma_i * br
            + self.rho_nr * self.sigma_n * self.sigma_r * j
            - 0.5 * self.sigma_r**2 * br * br
        )
        return float(_B(ar, t2 - t1)) * inner

    def yoy_forward(self, T1: Any, T2: Any) -> float:
        r""":math:`E^{T_2}_n[I(T_2)/I(T_1)]`, the forward ratio including convexity."""
        t1, t2 = self._t(T1), self._t(T2)
        Pn, Pr = self.nominal.discount, self.real.discount
        ratio = (
            float(Pn(t1)) * float(Pr(t2)) / (float(Pr(t1)) * float(Pn(t2)))
            if t1 > 0
            else float(Pr(t2) / Pn(t2))
        )
        return ratio * math.exp(self.yoy_convexity(t1, t2))

    def yoy_swaplet(self, T1: Any, T2: Any) -> float:
        """Present value of :math:`I(T_2)/I(T_1) - 1` paid at :math:`T_2`."""
        t2 = self._t(T2)
        return float(self.nominal.discount(t2)) * (self.yoy_forward(T1, t2) - 1.0)

    def yoy_swap_rate(self, times: Any) -> float:
        """Fair fixed rate of a year-on-year swap with payment times ``times`` (first period from 0)."""
        t = np.concatenate([[0.0], np.asarray([self._t(x) for x in np.atleast_1d(times)], float)])
        flt = sum(self.yoy_swaplet(a, b) for a, b in zip(t[:-1], t[1:]))
        ann = sum((b - a) * float(self.nominal.discount(b)) for a, b in zip(t[:-1], t[1:]))
        return float(flt / ann)
