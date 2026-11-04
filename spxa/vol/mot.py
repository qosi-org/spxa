r"""
Martingale optimal transport: model-free bounds from two marginals.

Vanilla prices at two expiries :math:`T_1 < T_2` pin down the laws
:math:`\mu` of :math:`X = S_{T_1}/F_{T_1}` and :math:`\nu` of
:math:`Y = S_{T_2}/F_{T_2}` (Breeden–Litzenberger), and absence of arbitrage
makes :math:`(X, Y)` a martingale, :math:`E[Y \mid X] = X`. The price of a
claim :math:`c(X, Y)` consistent with every arbitrage-free model therefore lies in

.. math::
    \Bigl[\inf_{\pi \in \mathcal M(\mu, \nu)} E_\pi[c],\ \sup_{\pi \in \mathcal M(\mu, \nu)} E_\pi[c]\Bigr],

where :math:`\mathcal M(\mu, \nu)` is the set of martingale couplings, non-empty
if and only if :math:`\mu \le_{cx} \nu` (Strassen). On finite supports the
problem is a linear programme. Its dual is the cheapest *superhedge*: static
vanillas :math:`\varphi(X)` and :math:`\psi(Y)` plus a dynamic position
:math:`h(X)` in the underlying from :math:`T_1` to :math:`T_2`, with

.. math:: \varphi(x) + \psi(y) + h(x)(y - x) \ge c(x, y) \quad\text{for all } (x, y),

and the primal and dual values agree (Beiglböck, Henry-Labordère & Penkner).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from spxa.vol.marginals import DiscreteMarginal

__all__ = ["MOTResult", "mot_bounds", "mot_solve", "ot_solve"]

Payoff = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MOTResult:
    r"""Optimal coupling and the dual (super- or sub-) hedge.

    ``phi`` and ``h`` live on the first marginal's points, ``psi`` on the second.
    For an upper bound :math:`\varphi(x) + \psi(y) + h(x)(y - x) \ge c(x, y)`, and
    the reverse for a lower bound.
    """

    value: float
    plan: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    h: np.ndarray
    sense: str
    mu: DiscreteMarginal
    nu: DiscreteMarginal

    @property
    def dual_value(self) -> float:
        return float(self.mu.weights @ self.phi + self.nu.weights @ self.psi)

    def hedge(self, x: Any, y: Any) -> Any:
        """Hedge payoff at points of the supports (index lookup)."""
        i = np.searchsorted(self.mu.points, x)
        j = np.searchsorted(self.nu.points, y)
        return self.phi[i] + self.psi[j] + self.h[i] * (np.asarray(y, float) - np.asarray(x, float))


def _constraints(
    mu: DiscreteMarginal, nu: DiscreteMarginal, martingale: bool
) -> tuple[Any, np.ndarray]:
    n, m = mu.points.size, nu.points.size
    idx = np.arange(n * m)
    rows = coo_matrix((np.ones(n * m), (idx // m, idx)), shape=(n, n * m))
    cols = coo_matrix((np.ones(n * m), (idx % m, idx)), shape=(m, n * m))
    blocks, b = [rows, cols], [mu.weights, nu.weights]
    if martingale:
        d = (nu.points[None, :] - mu.points[:, None]).ravel()
        blocks.append(coo_matrix((d, (idx // m, idx)), shape=(n, n * m)))
        b.append(np.zeros(n))
    return vstack(blocks).tocsr(), np.concatenate(b)


def _solve(
    mu: DiscreteMarginal, nu: DiscreteMarginal, payoff: Payoff, sense: str, martingale: bool
) -> MOTResult:
    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")
    C = np.asarray(payoff(mu.points[:, None], nu.points[None, :]), float)
    C = np.broadcast_to(C, (mu.points.size, nu.points.size))
    sign = -1.0 if sense == "max" else 1.0
    A, b = _constraints(mu, nu, martingale)
    # HiGHS presolve wrongly declares infeasibility when tail weights are tiny (1e-14)
    res = linprog(
        sign * C.ravel(),
        A_eq=A,
        b_eq=b,
        bounds=(0, None),
        method="highs",
        options={
            "presolve": False,
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if res.status != 0:
        raise ValueError(f"transport problem infeasible or failed: {res.message}")
    n, m = mu.points.size, nu.points.size
    y = sign * np.asarray(res.eqlin.marginals, float)
    h = y[n + m :] if martingale else np.zeros(n)
    return MOTResult(
        sign * float(res.fun), res.x.reshape(n, m), y[:n], y[n : n + m], h, sense, mu, nu
    )


def mot_solve(
    mu: DiscreteMarginal, nu: DiscreteMarginal, payoff: Payoff, sense: str = "max"
) -> MOTResult:
    """Extremal martingale coupling of ``mu`` and ``nu`` for ``payoff(x, y)``, with its dual hedge."""
    if not mu.precedes(nu, tol=1e-10):
        raise ValueError("marginals are not in convex order: no martingale coupling exists")
    return _solve(mu, nu, payoff, sense, martingale=True)


def ot_solve(
    mu: DiscreteMarginal, nu: DiscreteMarginal, payoff: Payoff, sense: str = "max"
) -> MOTResult:
    """Classical optimal transport (no martingale constraint), for comparison."""
    return _solve(mu, nu, payoff, sense, martingale=False)


def mot_bounds(mu: DiscreteMarginal, nu: DiscreteMarginal, payoff: Payoff) -> tuple[float, float]:
    """Model-free (lower, upper) price bounds of ``payoff(X, Y)``."""
    return mot_solve(mu, nu, payoff, "min").value, mot_solve(mu, nu, payoff, "max").value
