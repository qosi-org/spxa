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
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.market.black import black_price
from spxa.rates.lmm import LiborMarketModel

__all__ = ["FMMPaths", "ForwardMarketModel", "fmm_paths"]

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


@dataclass(frozen=True)
class FMMPaths:
    """Rates at the tenor dates: ``rates[k]`` (shape ``(n_paths, n)``) at :math:`T_k`, ``k = 0..n``."""

    model: ForwardMarketModel
    rates: np.ndarray

    @property
    def fixings(self) -> np.ndarray:
        r"""Realised compounded rates :math:`R_i(T_{i+1})`, shape ``(n_paths, n)``."""
        n = self.model.n
        return self.rates[np.arange(1, n + 1), :, np.arange(n)].T

    def deflator(self, k: int) -> np.ndarray:
        r""":math:`P(0, T_0)/\prod_{j<k}(1 + \tau_j R_j(T_{j+1}))`."""
        m = self.model
        return m.discounts[0] / np.prod(1.0 + m.tau[:k] * self.fixings[:, :k], axis=1)

    def price(self, values: np.ndarray, k: int) -> tuple[float, float]:
        x = np.asarray(values, float) * self.deflator(k)
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def fmm_paths(
    model: ForwardMarketModel,
    n_paths: int,
    steps_per_period: int = 4,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> FMMPaths:
    r"""Simulate under the spot measure of the backward-looking bank account.

    A rate stays alive until the end of its accrual period, and the drift of
    :math:`R_i` sums over the rates still accruing,
    :math:`\sum_{j = q(t)}^{i}\tau_j C_{ij}(R_j + \delta)/(1 + \tau_j R_j)` with
    :math:`T_{q(t)} < t \le T_{q(t)+1}`; the step uses a predictor–corrector drift.
    """
    m = model
    n, tau, d = m.n, m.tau, m.shift
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    X = np.tile(np.log(m.forwards + d), (n_paths, 1))
    out = np.empty((n + 1, n_paths, n))
    dates = m.times
    grid = [0.0]
    for k in range(n + 1):
        start, end = (0.0 if k == 0 else dates[k - 1]), dates[k]
        if end > start:
            n_sub = max(1, math.ceil(steps_per_period * (end - start) / float(np.mean(tau))))
            grid += list(np.linspace(start, end, n_sub + 1)[1:])
    if dates[0] == 0.0:
        out[0] = np.exp(X) - d
    k = 0 if dates[0] > 0 else 1
    for s in range(len(grid) - 1):
        t0, t1 = grid[s], grid[s + 1]
        q = int(np.searchsorted(dates[1:], t0 + 1e-14, side="left"))
        alive = np.arange(q, n)
        Ca = m.covariance(t0, t1)[np.ix_(alive, alive)]
        w, V = np.linalg.eigh(Ca)
        A = V * np.sqrt(np.maximum(w, 0.0))
        z = rng.standard_normal((half, alive.size))
        if antithetic:
            z = np.concatenate([z, -z])[:n_paths]
        dW = z @ A.T
        low = np.tril(Ca)

        def drift(Xa: np.ndarray) -> np.ndarray:
            R = np.exp(Xa) - d
            g = tau[alive] * (R + d) / (1.0 + tau[alive] * R)
            return (g[:, None, :] * low[None, :, :]).sum(axis=2) - 0.5 * np.diag(Ca)

        Xa = X[:, alive]
        mu0 = drift(Xa)
        X[:, alive] = Xa + 0.5 * (mu0 + drift(Xa + mu0 + dW)) + dW
        while k <= n and abs(t1 - dates[k]) < 1e-12:
            out[k] = np.exp(X) - d
            k += 1
    return FMMPaths(m, out)
