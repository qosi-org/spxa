r"""Exact simulation of G2++ under the risk-neutral measure.

The state :math:`(x, y)` and its integrals :math:`(I_x, I_y)` form a
four-dimensional Gaussian with time-homogeneous transitions. Over a step
:math:`h`, with :math:`B_k(h) = (1 - e^{-kh})/k`,

.. math::
    x' = e^{-ah}x + \varepsilon_x, \qquad I_x' = I_x + B_a(h)\,x + \eta_x,

and similarly for :math:`y`. For factors :math:`i, j` with reversion
speeds :math:`k_i, k_j`, volatilities :math:`s_i, s_j` and correlation
:math:`\rho_{ij}`, the innovations have covariances

.. math::
    \operatorname{Cov}(\varepsilon_i, \varepsilon_j) = \rho_{ij}s_is_j B_{k_i + k_j}(h), \qquad
    \operatorname{Cov}(\varepsilon_i, \eta_j) = \rho_{ij}s_is_j\,\frac{B_{k_i}(h) - B_{k_i + k_j}(h)}{k_j},

.. math::
    \operatorname{Cov}(\eta_i, \eta_j) = \rho_{ij}s_is_j\,\frac{h - B_{k_i}(h) - B_{k_j}(h) + B_{k_i + k_j}(h)}{k_ik_j}.

Since :math:`\int_0^t\varphi = -\log P(0, t) + \tfrac12 V(0, t)`, the bank
account discount factor is
:math:`D(t) = P(0, t)\exp\bigl(-\tfrac12 V(0, t) - I_x(t) - I_y(t)\bigr)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spxa.rates.g2pp import G2PP

__all__ = ["G2PPPaths", "g2pp_paths"]


def _B(k: float, h: float) -> float:
    return float(-np.expm1(-k * h) / k)


def step_covariance(model: G2PP, h: float) -> np.ndarray:
    """Covariance of :math:`(\\varepsilon_x, \\varepsilon_y, \\eta_x, \\eta_y)` over a step ``h``."""
    k = (model.a, model.b)
    s = (model.sigma, model.eta)
    rho = np.array([[1.0, model.rho], [model.rho, 1.0]])
    C = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            c = rho[i, j] * s[i] * s[j]
            C[i, j] = c * _B(k[i] + k[j], h)
            C[i, 2 + j] = c * (_B(k[i], h) - _B(k[i] + k[j], h)) / k[j]
            C[2 + i, 2 + j] = (
                c * (h - _B(k[i], h) - _B(k[j], h) + _B(k[i] + k[j], h)) / (k[i] * k[j])
            )
    C[2:, :2] = C[:2, 2:].T
    return C


@dataclass(frozen=True)
class G2PPPaths:
    model: G2PP
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    discount: np.ndarray

    def _row(self, t: float) -> int:
        i = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[i] - t) > 1e-12:
            raise KeyError(f"time {t} was not simulated")
        return i

    def zero_bond(self, t: float, T: Any) -> np.ndarray:
        i = self._row(t)
        TT = np.asarray(T, float)
        x, y = (self.x[i], self.y[i]) if TT.ndim == 0 else (self.x[i][:, None], self.y[i][:, None])
        return np.asarray(self.model.discount_bond(t, TT, x, y), float)

    def price(self, payoff: np.ndarray, t: float) -> tuple[float, float]:
        v = self.discount[self._row(t)] * payoff
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def g2pp_paths(
    model: G2PP, times: Sequence[float], n_paths: int, seed: int = 0, antithetic: bool = True
) -> G2PPPaths:
    """Simulate ``model`` exactly at ``times`` (positive, increasing)."""
    req = np.asarray(times, float)
    if req.size == 0 or req[0] <= 0 or np.any(np.diff(req) <= 0):
        raise ValueError("times must be positive and increasing")
    rng = np.random.default_rng(seed)
    m = (n_paths + 1) // 2 if antithetic else n_paths
    size = 2 * m if antithetic else m
    state = np.zeros((4, size))  # x, y, Ix, Iy
    xs, ys, ds = [np.zeros(n_paths)], [np.zeros(n_paths)], [np.ones(n_paths)]
    prev = 0.0
    for t in req:
        h = t - prev
        L = np.linalg.cholesky(step_covariance(model, h))
        g = rng.standard_normal((4, m))
        if antithetic:
            g = np.concatenate([g, -g], axis=1)
        e = L @ g
        x, y, Ix, Iy = state
        state = np.array(
            [
                np.exp(-model.a * h) * x + e[0],
                np.exp(-model.b * h) * y + e[1],
                Ix + _B(model.a, h) * x + e[2],
                Iy + _B(model.b, h) * y + e[3],
            ]
        )
        P0 = float(model.curve.discount(t))
        xs.append(state[0, :n_paths])
        ys.append(state[1, :n_paths])
        ds.append(
            P0 * np.exp(-0.5 * float(model.V(0.0, t)) - state[2, :n_paths] - state[3, :n_paths])
        )
        prev = t
    return G2PPPaths(model, np.r_[0.0, req], np.array(xs), np.array(ys), np.array(ds))
