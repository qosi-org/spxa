r"""
Monte Carlo for the LIBOR market model under the spot LIBOR measure.

With :math:`q(t)` the index of the first forward not yet fixed
(:math:`T_{q-1} < t \le T_q`), the numeraire is the discretely rolled bank
account :math:`N(t) = P(t, T_{q(t)})\prod_{j < q(t)}(1 + \tau_j L_j(T_j)) / P(0, T_0)`
and :math:`X_i = \log(L_i + \delta)` follows

.. math::
    dX_i = \Bigl(\sigma_i\sum_{j = q(t)}^{i}\frac{\tau_j\rho_{ij}\sigma_j(L_j + \delta)}{1 + \tau_j L_j}
        - \tfrac12\sigma_i^2\Bigr)dt + \sigma_i\,dW_i.

Each step uses the exact integrated covariance :math:`C = \int\rho\sigma\sigma^{\mathsf T}`
of :meth:`LiborMarketModel.covariance` for the Gaussian increment and the
drift, with the drift either frozen at the start of the step (log-Euler) or
averaged with its value at the predicted end point (predictor–corrector,
Hunter, Jäckel & Joshi 2001). Deflated bonds :math:`P(t, T_k)/N(t)` are then
martingales up to the drift discretisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spxa.rates.lmm import LiborMarketModel

__all__ = ["LMMPaths", "lmm_paths"]


@dataclass(frozen=True)
class LMMPaths:
    """Forward curves at the fixing dates.

    ``forwards[k]`` has shape ``(n_paths, n)``: the forwards at :math:`T_k`
    (already fixed ones hold their fixing).
    """

    model: LiborMarketModel
    forwards: np.ndarray

    @property
    def fixings(self) -> np.ndarray:
        """:math:`L_j(T_j)`, shape ``(n_paths, n)``."""
        n = self.model.n
        return self.forwards[np.arange(n), :, np.arange(n)].T

    def deflator(self, k: int) -> np.ndarray:
        r""":math:`1/N(T_k) = P(0, T_0)/\prod_{j<k}(1 + \tau_j L_j(T_j))`, ``k = 0..n``."""
        m = self.model
        growth = np.prod(1.0 + m.tau[:k] * self.fixings[:, :k], axis=1)
        return m.discounts[0] / growth

    def bond(self, k: int, e: int) -> np.ndarray:
        r""":math:`P(T_k, T_e) = \prod_{j=k}^{e-1}(1 + \tau_j L_j(T_k))^{-1}`."""
        m = self.model
        return 1.0 / np.prod(1.0 + m.tau[k:e] * self.forwards[k][:, k:e], axis=1)

    def price(self, values: np.ndarray, k: int) -> tuple[float, float]:
        """Present value of ``values`` paid at :math:`T_k`, with its standard error."""
        x = np.asarray(values, float) * self.deflator(k)
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def lmm_paths(
    model: LiborMarketModel,
    n_paths: int,
    steps_per_period: int = 4,
    seed: Optional[int] = None,
    antithetic: bool = True,
    predictor_corrector: bool = True,
) -> LMMPaths:
    """Simulate the forwards to every fixing date."""
    m = model
    n, tau, d = m.n, m.tau, m.shift
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    X = np.tile(np.log(m.forwards + d), (n_paths, 1))
    out = np.empty((n, n_paths, n))
    grid = [0.0]
    for k in range(n):
        start, end = (0.0 if k == 0 else m.times[k - 1]), m.times[k]
        if end > start:
            n_sub = max(1, math.ceil(steps_per_period * (end - start) / float(np.mean(tau))))
            grid += list(np.linspace(start, end, n_sub + 1)[1:])
    grid_arr = np.array(grid)
    k = 0
    for s in range(grid_arr.size - 1):
        t0, t1 = float(grid_arr[s]), float(grid_arr[s + 1])
        q = int(np.searchsorted(m.times[:-1], t0 + 1e-14, side="left"))
        C = m.covariance(t0, t1)
        alive = np.arange(q, n)
        Ca = C[np.ix_(alive, alive)]
        w, V = np.linalg.eigh(Ca)
        A = V * np.sqrt(np.maximum(w, 0.0))
        z = rng.standard_normal((half, alive.size))
        if antithetic:
            z = np.concatenate([z, -z])[:n_paths]
        dW = z @ A.T

        def drift(Xa: np.ndarray) -> np.ndarray:
            L = np.exp(Xa) - d
            g = tau[alive] * (L + d) / (1.0 + tau[alive] * L)
            # mu_i = sum_{j = q}^{i} C_ij g_j  (lower-triangular in the alive block)
            return (g[:, None, :] * np.tril(Ca)[None, :, :]).sum(axis=2) - 0.5 * np.diag(Ca)

        Xa = X[:, alive]
        mu0 = drift(Xa)
        if predictor_corrector:
            mu = 0.5 * (mu0 + drift(Xa + mu0 + dW))
        else:
            mu = mu0
        X[:, alive] = Xa + mu + dW
        while k < n and abs(t1 - m.times[k]) < 1e-12:
            out[k] = np.exp(X) - d
            k += 1
    return LMMPaths(m, out)
