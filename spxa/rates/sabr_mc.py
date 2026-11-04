r"""Monte Carlo for SABR with absorption.

The volatility is simulated exactly,
:math:`\alpha_{t+\Delta} = \alpha_t \exp(\nu\Delta Z - \tfrac12\nu^2\Delta)`,
and the shifted forward by an Euler step

.. math:: F_{t+\Delta} = F_t + \alpha_t (F_t + s)^\beta \bigl(\rho\,\Delta Z + \sqrt{1-\rho^2}\,\Delta W\bigr),

with the volatility frozen at the start of the step. A path that reaches
:math:`-s` stays there: absorption is what makes SABR arbitrage-free for
:math:`\beta < 1`. Discrete monitoring misses crossings inside a step, so
the absorbed mass is biased low at :math:`O(\sqrt{\Delta})`. For the normal
case (:math:`\beta = 0`) a Brownian-bridge test with the frozen
volatility removes that bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spxa.rates.sabr import SABR

__all__ = ["SABRPaths", "sabr_paths"]


@dataclass(frozen=True)
class SABRPaths:
    """Terminal forwards (and volatilities) of simulated SABR paths."""

    F: np.ndarray
    alpha: np.ndarray
    absorbed: np.ndarray
    shift: float

    def call(self, K: Any) -> tuple[np.ndarray, np.ndarray]:
        """Undiscounted calls and their standard errors."""
        KK = np.atleast_1d(np.asarray(K, float))
        pay = np.maximum(self.F[None, :] - KK[:, None], 0.0)
        return pay.mean(axis=1), pay.std(axis=1, ddof=1) / np.sqrt(self.F.size)

    def put(self, K: Any) -> tuple[np.ndarray, np.ndarray]:
        KK = np.atleast_1d(np.asarray(K, float))
        pay = np.maximum(KK[:, None] - self.F[None, :], 0.0)
        return pay.mean(axis=1), pay.std(axis=1, ddof=1) / np.sqrt(self.F.size)


def sabr_paths(
    model: SABR, F0: float, T: float, n_steps: int, n_paths: int, seed: int = 0, bridge: bool = True
) -> SABRPaths:
    """Simulate terminal SABR forwards with absorption at ``-shift`` (antithetic pairs)."""
    if T <= 0 or n_steps < 1 or n_paths < 2:
        raise ValueError("need T > 0, n_steps >= 1, n_paths >= 2")
    rng = np.random.default_rng(seed)
    m = (n_paths + 1) // 2
    dt = T / n_steps
    s, b = model.shift, model.beta
    F = np.full(2 * m, F0 + s)  # work with the shifted forward
    a = np.full(2 * m, model.alpha)
    dead = np.zeros(2 * m, bool)
    rc = np.sqrt(1.0 - model.rho**2)
    for _ in range(n_steps):
        z = rng.standard_normal(m)
        w = rng.standard_normal(m)
        z, w = np.concatenate([z, -z]), np.concatenate([w, -w])
        vol = a * np.maximum(F, 0.0) ** b
        Fn = F + vol * np.sqrt(dt) * (model.rho * z + rc * w)
        hit = Fn <= 0.0
        if bridge and b == 0.0:
            # crossing probability of a Brownian bridge between two positive values
            with np.errstate(divide="ignore", over="ignore"):
                p = np.exp(-2.0 * F * Fn / (vol * vol * dt))
            hit |= (Fn > 0) & (rng.random(2 * m) < p)
        dead |= hit
        F = np.where(dead, 0.0, Fn)
        a = a * np.exp(model.nu * np.sqrt(dt) * z - 0.5 * model.nu**2 * dt)
    F = F[:n_paths] - s
    return SABRPaths(F, a[:n_paths], dead[:n_paths], s)
