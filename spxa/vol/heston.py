r"""
Heston stochastic volatility in forward coordinates.

.. math::
    \frac{dF_t}{F_t} = \sqrt{v_t}\,dW^1_t, \qquad
    dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^2_t, \qquad
    d\langle W^1, W^2\rangle_t = \rho\,dt.

Vanilla prices come from the characteristic function of
:math:`\log(F_T/F_0)` in the Albrecher et al. ("little trap") form with
Lewis's single integral (:func:`spxa.pde.reference.lewis_call_price`). The
model is the stochastic-volatility component of :mod:`spxa.vol.slv`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spxa.market.implied import implied_black_volatility
from spxa.pde.reference import heston_cf, lewis_call_price

__all__ = ["Heston"]


class Heston:
    """Heston model with initial variance ``v0``, mean reversion ``kappa`` to ``theta``,
    vol-of-variance ``xi`` and spot/variance correlation ``rho``."""

    def __init__(self, v0: float, kappa: float, theta: float, xi: float, rho: float) -> None:
        if min(v0, theta) < 0 or kappa <= 0 or xi <= 0:
            raise ValueError("need v0, theta >= 0 and kappa, xi > 0")
        if not -1 < rho < 1:
            raise ValueError("rho must lie in (-1, 1)")
        self.v0, self.kappa, self.theta = float(v0), float(kappa), float(theta)
        self.xi, self.rho = float(xi), float(rho)

    def __repr__(self) -> str:
        return (
            f"Heston(v0={self.v0:g}, kappa={self.kappa:g}, theta={self.theta:g}, "
            f"xi={self.xi:g}, rho={self.rho:g})"
        )

    @property
    def feller(self) -> bool:
        r""":math:`2\kappa\theta \ge \xi^2`: zero is unattainable for the variance."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def scaled(self, eta: float) -> "Heston":
        r"""The same model with vol-of-variance :math:`\eta\,\xi` (the SLV mixing fraction)."""
        return Heston(self.v0, self.kappa, self.theta, max(eta, 1e-12) * self.xi, self.rho)

    def mean_variance(self, t: Any) -> Any:
        r""":math:`E[v_t] = \theta + (v_0 - \theta)e^{-\kappa t}`."""
        return self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * np.asarray(t, float))

    def expected_total_variance(self, T: Any) -> Any:
        r""":math:`E\int_0^T v_t\,dt = \theta T + (v_0 - \theta)(1 - e^{-\kappa T})/\kappa`."""
        T = np.asarray(T, float)
        return self.theta * T + (self.v0 - self.theta) * (1 - np.exp(-self.kappa * T)) / self.kappa

    def cf(self, T: float) -> Any:
        r"""Characteristic function of :math:`\log(F_T/F_0)`, a martingale-normalised log-return."""
        return heston_cf(T, self.v0, self.kappa, self.theta, self.xi, self.rho)

    def price(self, K: Any, T: float, F: float = 1.0, df: float = 1.0, option: str = "call") -> Any:
        """Vanilla prices on forward ``F`` with discount factor ``df``."""
        if T <= 0:
            raise ValueError("T must be positive")
        cf = self.cf(T)
        Ks = np.atleast_1d(np.asarray(K, float))
        out = np.array([df * lewis_call_price(cf, F, float(k), T, 0.0, 0.0, option) for k in Ks])
        return float(out[0]) if np.ndim(K) == 0 else out.reshape(np.shape(K))

    def implied_vol(self, K: Any, T: float, F: float = 1.0) -> Any:
        """Black implied volatilities (out-of-the-money options are inverted)."""
        Ks = np.asarray(K, float)
        calls = np.asarray(self.price(Ks, T, F), float)
        puts = calls - (F - Ks)
        otm = np.where(Ks >= F, calls, puts)
        kind = np.where(Ks >= F, "call", "put")
        vol = np.asarray(implied_black_volatility(otm, F, Ks, T, 1.0, kind), float)
        return float(vol) if vol.ndim == 0 else vol

    def atm_short_vol(self) -> float:
        r""":math:`\sqrt{v_0}`, the short-maturity limit of the at-the-money implied volatility."""
        return math.sqrt(self.v0)
