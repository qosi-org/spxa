r"""
Forward market model for backward-looking rates (Lyashenko & Mercurio, 2019).

After the end of LIBOR, floating legs pay compounded overnight rates
:math:`R_i`, known only at the *end* :math:`T_{i+1}` of their accrual period
:math:`[T_i, T_{i+1}]`. The forward market model extends the LMM to these
rates by letting each forward keep moving through its accrual period with a
volatility that decays to zero,

.. math::
    dR_i = \sigma_i(t)\,g_i(t)\,(R_i + \delta)\,dW_i, \qquad
    g_i(t) = \min\Bigl(1, \frac{(T_{i+1} - t)^+}{T_{i+1} - T_i}\Bigr),

so that :math:`R_i(t)` equals the forward-looking forward before
:math:`T_i` and the realised compounded rate at :math:`T_{i+1}`. A caplet on
:math:`R_i` paid at :math:`T_{i+1}` is a Black formula with variance
:math:`\int_0^{T_{i+1}}\sigma_i^2 g_i^2\,dt`; for a flat volatility this is
:math:`\sigma^2(T_i + \tau_i/3)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spxa.market.black import black_price
from spxa.rates.lmm import LiborMarketModel

__all__ = ["ForwardMarketModel"]

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


class ForwardMarketModel(LiborMarketModel):
    """:class:`LiborMarketModel` whose forwards accrue with decaying volatility."""

    def decay(self, t: float) -> np.ndarray:
        """:math:`g_i(t)` for every forward."""
        start, end = self.times[:-1], self.times[1:]
        return np.clip((end - t) / (end - start), 0.0, 1.0)

    def sigma(self, t: float) -> np.ndarray:
        tau = np.maximum(self.times[:-1] - t, 0.0)
        return self.k * self.vol(tau) * self.decay(t)

    def covariance(self, t0: float, t1: float) -> np.ndarray:
        r""":math:`\rho_{ij}\int_{t_0}^{t_1}\sigma_i g_i\sigma_j g_j\,dt`: closed form before the
        accrual periods start, Gauss–Legendre on the pieces where a decay is active."""
        start = self.times[:-1]
        end = self.times[1:]
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                a = min(t1, start[i], start[j])
                val = 0.0
                if a > t0:
                    val += self.k[i] * self.k[j] * self.vol.covariance(start[i], start[j], t0, a)
                lo, hi = max(t0, a), min(t1, end[i], end[j])
                if hi > lo:
                    # piecewise smooth: split at the accrual starts inside (lo, hi)
                    cuts = sorted({lo, hi, *(c for c in (start[i], start[j]) if lo < c < hi)})
                    for u, w in zip(cuts[:-1], cuts[1:]):
                        ts = 0.5 * (u + w) + 0.5 * (w - u) * _GL_X
                        f = np.array([self.sigma(t)[[i, j]] for t in ts])
                        val += 0.5 * (w - u) * float(np.sum(_GL_W * f[:, 0] * f[:, 1]))
                C[i, j] = C[j, i] = val
        return C * self.correlation

    def caplet_variance(self, i: int) -> float:
        return float(self.covariance(0.0, self.times[i + 1])[i, i])

    def caplet_vol(self, i: int) -> float:
        """Black vol of the caplet on :math:`R_i`, quoted to the payment date :math:`T_{i+1}`."""
        return math.sqrt(self.caplet_variance(i) / self.times[i + 1])

    def caplet(self, i: int, K: Any, notional: float = 1.0, option: str = "call") -> Any:
        """Caplet on the compounded rate :math:`R_i`, fixed and paid at :math:`T_{i+1}`."""
        T = self.times[i + 1]
        F, d = self.forwards[i] + self.shift, self.shift
        df = self.discounts[i + 1] * self.tau[i] * notional
        return black_price(F, np.asarray(K, float) + d, T, self.caplet_vol(i), df, option)
