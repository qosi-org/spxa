r"""Black–Karasinski model on a trinomial tree.

The short rate is lognormal (Black & Karasinski, 1991):

.. math:: d\log r_t = \bigl(\theta(t) - a\log r_t\bigr)\,dt + \sigma\,dW_t,

so rates stay positive, but bonds have no closed form. The model is built
on the Hull–White trinomial geometry for :math:`x = \log r - \alpha(t)`
(:class:`~spxa.rates.hw_tree.TrinomialShortRateTree`). At each step,
:math:`\alpha_m` solves

.. math:: \sum_j Q_{m,j}\exp\bigl(-e^{\alpha_m + j\Delta x}\Delta t\bigr) = P(0, (m+1)\Delta t)

by Newton's method (the left side is decreasing and convex in
:math:`\alpha_m`), so the tree reprices every zero bond on its grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.hw_tree import TrinomialShortRateTree

__all__ = ["BlackKarasinski", "BlackKarasinskiTree"]


@dataclass(frozen=True)
class BlackKarasinski:
    curve: YieldCurve
    a: float
    sigma: float

    def __post_init__(self) -> None:
        if self.a <= 0 or self.sigma <= 0:
            raise ValueError("need a > 0 and sigma > 0")

    def tree(self, horizon: float, n_steps: int) -> "BlackKarasinskiTree":
        return BlackKarasinskiTree(self, horizon, n_steps)


class BlackKarasinskiTree(TrinomialShortRateTree):
    """Trinomial tree for :class:`BlackKarasinski` fitted to its curve."""

    def __init__(self, model: BlackKarasinski, horizon: float, n_steps: int) -> None:
        self.model = model
        super().__init__(model.curve, model.a, model.sigma, horizon, n_steps)

    def node_rates(self, alpha: float, x: np.ndarray) -> np.ndarray:
        return np.exp(alpha + x)

    def _solve_alpha(self, Q: np.ndarray, x: np.ndarray, target: float) -> float:
        dt = self.dt
        # start from the flat-rate guess r = -log(P_{m+1}/sum Q)/dt
        al = math.log(max(-math.log(target / float(Q.sum())) / dt, 1e-12))
        step = 1.0
        for _ in range(100):
            r = np.exp(al + x)
            e = Q * np.exp(-r * dt)
            done = abs(step) < 1e-8  # quadratic convergence: this step is at roundoff
            step = (float(e.sum()) - target) / -float(np.sum(e * r * dt))
            al -= step
            if done:
                break
        else:  # pragma: no cover - Newton on a convex decreasing function converges
            raise RuntimeError("alpha did not converge")
        return al
