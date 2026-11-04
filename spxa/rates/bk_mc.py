r"""Monte Carlo for Black–Karasinski on the drift fitted by its tree.

The tree fixes :math:`\alpha_m` on its grid (see
:mod:`spxa.rates.black_karasinski`). Here :math:`x = \log r - \alpha` is
simulated exactly as the Ornstein–Uhlenbeck process
:math:`dx = -a x\,dt + \sigma\,dW`:

.. math:: x_{m+1} = e^{-a\Delta t}x_m + \sigma\sqrt{B_2(\Delta t)}\,Z,
          \qquad B_2(h) = \frac{1 - e^{-2ah}}{2a}.

The rate :math:`r_m = e^{\alpha_m + x_m}` is held over each step, as in the
tree, so :math:`D_{m+1} = D_m e^{-r_m\Delta t}`. The paths reproduce the
curve up to the tree's :math:`O(\Delta t)` moment-matching error (inside the
statistical error at 100k paths and 120 steps over six years), and
they price path-dependent payoffs that the tree cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spxa.rates.black_karasinski import BlackKarasinskiTree

__all__ = ["BlackKarasinskiPaths", "black_karasinski_paths"]


@dataclass(frozen=True)
class BlackKarasinskiPaths:
    tree: BlackKarasinskiTree
    x: np.ndarray  # (n_steps + 1, n_paths)
    discount: np.ndarray  # (n_steps + 1, n_paths)

    def short_rate(self, m: int) -> np.ndarray:
        return np.exp(self.tree.alpha[min(m, self.tree.n_steps - 1)] + self.x[m])

    def zero_bond(self, m: int, maturity: float) -> np.ndarray:
        """P(t_m, maturity) on each path, interpolated in x from the tree nodes."""
        w = self.tree.width(m)
        nodes = np.arange(-w, w + 1) * self.tree.dx
        return np.interp(self.x[m], nodes, self.tree.discount_bonds(m, maturity))

    def price(self, payoff: np.ndarray, m: int) -> tuple[float, float]:
        v = payoff * self.discount[m]
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def black_karasinski_paths(
    tree: BlackKarasinskiTree, n_paths: int, seed: int = 0, antithetic: bool = True
) -> BlackKarasinskiPaths:
    a, sigma, dt, n = tree.model.a, tree.model.sigma, tree.dt, tree.n_steps
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((n, half))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)
    decay = math.exp(-a * dt)
    sd = sigma * math.sqrt(-math.expm1(-2 * a * dt) / (2 * a))
    x = np.zeros((n + 1, Z.shape[1]))
    D = np.ones_like(x)
    for m in range(n):
        D[m + 1] = D[m] * np.exp(-np.exp(tree.alpha[m] + x[m]) * dt)
        x[m + 1] = decay * x[m] + sd * Z[m]
    return BlackKarasinskiPaths(tree, x, D)
