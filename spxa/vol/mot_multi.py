r"""
Multi-marginal martingale optimal transport for path-dependent claims.

With vanilla smiles at :math:`T_1 < \dots < T_n`, a claim
:math:`c(X_1, \dots, X_n)` on the normalised prices is bounded by optimising
over joint laws with the given marginals that are martingales in their own
filtration, :math:`E[X_{t+1} \mid X_1, \dots, X_t] = X_t`. On grids this is a
linear programme in :math:`\prod_t n_t` variables, with one martingale
constraint per path prefix. Intermediate smiles can only tighten the bounds
of a claim on the first and last dates. A claim on two consecutive dates has
the same bounds as in the two-marginal problem, because any two-step
coupling extends with a Markov kernel.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from spxa.vol.marginals import DiscreteMarginal

__all__ = ["mot_multi"]


def mot_multi(
    marginals: Sequence[DiscreteMarginal],
    payoff: Callable[..., np.ndarray],
    sense: str = "max",
) -> tuple[float, np.ndarray]:
    """Extremal value and plan (an array of shape ``(n_1, ..., n_T)``) for ``payoff(x_1, ..., x_T)``."""
    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")
    if len(marginals) < 2:
        raise ValueError("need at least two marginals")
    for a, b in zip(marginals, marginals[1:]):
        if not a.precedes(b, tol=1e-10):
            raise ValueError("marginals are not in convex order")
    shape = tuple(m.points.size for m in marginals)
    N = int(np.prod(shape))
    idx = np.unravel_index(np.arange(N), shape)
    grids = np.meshgrid(*[m.points for m in marginals], indexing="ij")
    C = np.broadcast_to(np.asarray(payoff(*grids), float), shape).ravel()
    blocks, rhs = [], []
    for t, m in enumerate(marginals):
        blocks.append(coo_matrix((np.ones(N), (idx[t], np.arange(N))), shape=(shape[t], N)))
        rhs.append(m.weights)
    for t in range(len(marginals) - 1):
        prefix = np.ravel_multi_index(idx[: t + 1], shape[: t + 1])
        n_pre = int(np.prod(shape[: t + 1]))
        d = marginals[t + 1].points[idx[t + 1]] - marginals[t].points[idx[t]]
        blocks.append(coo_matrix((d, (prefix, np.arange(N))), shape=(n_pre, N)))
        rhs.append(np.zeros(n_pre))
    sign = -1.0 if sense == "max" else 1.0
    res = linprog(
        sign * C,
        A_eq=vstack(blocks).tocsr(),
        b_eq=np.concatenate(rhs),
        bounds=(0, None),
        method="highs",
        options={
            "presolve": False,
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if res.status != 0:
        raise ValueError(f"multi-marginal problem failed: {res.message}")
    return sign * float(res.fun), res.x.reshape(shape)
