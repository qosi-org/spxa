r"""Exact simulation of the Hull–White model under the risk-neutral measure.

Write :math:`x_t = z_t + m(t)`, where :math:`z` is the driftless
Ornstein–Uhlenbeck process :math:`dz = -a z\,dt + \sigma(t)\,dW`,
:math:`z_0 = 0`, and :math:`m(t) = E^Q[x_t]`. Let
:math:`J_t = \int_0^t z_u\,du`. The pair :math:`(z, J)` is Gaussian with
exact transitions over each step. For constant :math:`\sigma` over a step
:math:`h`,

.. math::
    z' = e^{-ah} z + \varepsilon_1, \qquad J' = J + B(h)\,z + \varepsilon_2,

where :math:`\operatorname{Var}\varepsilon_1 = \sigma^2 B_2(h)`,
:math:`\operatorname{Var}\varepsilon_2 = \sigma^2(h - 2B(h) + B_2(h))/a^2` and
:math:`\operatorname{Cov} = \sigma^2(B(h) - B_2(h))/a`, with
:math:`B(h) = (1 - e^{-ah})/a` and :math:`B_2(h) = (1 - e^{-2ah})/(2a)`.
The bank-account discount factor is

.. math:: D(t) = e^{-\int_0^t r_u du} = P(0, t)\exp\bigl(-J_t - \tfrac12\operatorname{Var}(J_t)\bigr),

since :math:`r = f(0, t) + x` and :math:`E[D(t)] = P(0, t)` fixes the
deterministic part. Its derivative gives
:math:`m(t) = \tfrac12\,\tfrac{d}{dt}\operatorname{Var}(J_t)`. The
volatility knots are added to the step grid, so the transitions are exact
for piecewise-constant :math:`\sigma`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spxa.rates.hull_white import HullWhite

__all__ = ["HullWhitePaths", "hull_white_paths", "hull_white_transition"]


@dataclass(frozen=True)
class HullWhitePaths:
    """``x[i]`` and ``discount[i]`` at ``times[i]`` (``times[0] = 0``), one column per path."""

    model: HullWhite
    times: np.ndarray
    x: np.ndarray
    discount: np.ndarray

    def _row(self, t: float) -> int:
        i = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[i] - t) > 1e-12:
            raise KeyError(f"time {t} was not simulated")
        return i

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """``(x_t, D(t))``."""
        i = self._row(t)
        return self.x[i], self.discount[i]

    def short_rate(self, t: float) -> np.ndarray:
        return self.model.short_rate(t, self.at(t)[0])

    def zero_bond(self, t: float, T: Any) -> np.ndarray:
        """:math:`P(t, T)` on every path (shape ``(n_paths,)`` or ``(n_paths, len(T))``)."""
        x = self.at(t)[0]
        TT = np.asarray(T, float)
        return np.asarray(self.model.discount_bond(t, TT, x[:, None] if TT.ndim else x), float)

    def price(self, payoff: np.ndarray, t: float) -> tuple[float, float]:
        """Mean and standard error of :math:`D(t)\\cdot` ``payoff``."""
        v = self.at(t)[1] * payoff
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def hull_white_paths(
    model: HullWhite, times: Sequence[float], n_paths: int, seed: int = 0, antithetic: bool = True
) -> HullWhitePaths:
    """Simulate ``model`` exactly at ``times`` (increasing, positive)."""
    req = np.asarray(times, float)
    if req.size == 0 or req[0] <= 0 or np.any(np.diff(req) <= 0):
        raise ValueError("times must be positive and increasing")
    grid = np.unique(np.r_[0.0, req, model.sigma_times[model.sigma_times < req[-1]]])
    rng = np.random.default_rng(seed)
    m = (n_paths + 1) // 2 if antithetic else n_paths
    z = np.zeros(2 * m if antithetic else m)
    J = np.zeros_like(z)
    a = model.a
    out_x, out_d = [np.zeros(n_paths)], [np.ones(n_paths)]
    for t0, t1 in zip(grid[:-1], grid[1:]):
        h = t1 - t0
        sig = float(model.sigma[np.searchsorted(model.sigma_times, t0, side="right")])
        Bh = -np.expm1(-a * h) / a
        B2 = -np.expm1(-2 * a * h) / (2 * a)
        v1 = sig * sig * B2
        v2 = sig * sig * (h - 2 * Bh + B2) / (a * a)
        c12 = sig * sig * (Bh - B2) / a
        L11 = np.sqrt(v1)
        L21 = c12 / L11
        L22 = np.sqrt(max(v2 - L21 * L21, 0.0))
        g1, g2 = rng.standard_normal(m), rng.standard_normal(m)
        if antithetic:
            g1, g2 = np.concatenate([g1, -g1]), np.concatenate([g2, -g2])
        z, J = np.exp(-a * h) * z + L11 * g1, J + Bh * z + L21 * g1 + L22 * g2
        if np.any(np.abs(req - t1) < 1e-14):
            P0 = float(model.curve.discount(t1))
            out_x.append(z[:n_paths] + model.mean_x(t1))
            out_d.append(P0 * np.exp(-J[:n_paths] - 0.5 * model.integrated_variance(t1)))
    return HullWhitePaths(model, np.r_[0.0, req], np.array(out_x), np.array(out_d))


def hull_white_transition(
    model: HullWhite, t0: float, t1: float, x: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    r"""Exact step of the state from ``t0`` to ``t1``: returns :math:`x_{t_1}` and
    :math:`D(t_0, t_1) = e^{-\int_{t_0}^{t_1} r\,du}` for each start value in ``x``.

    The same :math:`(z, J)` transition as :func:`hull_white_paths`, started
    from :math:`z = x - m(t_0)`; the deterministic part of the discount is
    :math:`P(0, t_1)/P(0, t_0)\,e^{-\frac12(\operatorname{Var}J_{t_1} - \operatorname{Var}J_{t_0})}`.
    """
    z = np.asarray(x, float) - model.mean_x(t0)
    dJ = np.zeros_like(z)
    knots = model.sigma_times[(model.sigma_times > t0) & (model.sigma_times < t1)]
    grid = np.r_[t0, knots, t1]
    a = model.a
    for u0, u1 in zip(grid[:-1], grid[1:]):
        h = u1 - u0
        sig = float(model.sigma[np.searchsorted(model.sigma_times, u0, side="right")])
        Bh = -np.expm1(-a * h) / a
        B2 = -np.expm1(-2 * a * h) / (2 * a)
        v1 = sig * sig * B2
        v2 = sig * sig * (h - 2 * Bh + B2) / (a * a)
        c12 = sig * sig * (Bh - B2) / a
        L11 = np.sqrt(v1)
        L21 = c12 / L11
        L22 = np.sqrt(max(v2 - L21 * L21, 0.0))
        g1, g2 = rng.standard_normal(z.shape), rng.standard_normal(z.shape)
        z, dJ = np.exp(-a * h) * z + L11 * g1, dJ + Bh * z + L21 * g1 + L22 * g2
    c = model.curve
    det = float(c.discount(t1)) / float(c.discount(t0)) if t0 > 0 else float(c.discount(t1))
    var = model.integrated_variance(t1) - (model.integrated_variance(t0) if t0 > 0 else 0.0)
    return z + model.mean_x(t1), det * np.exp(-dJ - 0.5 * var)
