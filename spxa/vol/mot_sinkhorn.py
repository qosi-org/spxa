r"""
Entropic martingale optimal transport (Sinkhorn with a martingale constraint).

Adding the entropy penalty :math:`\varepsilon\,\mathrm{KL}(\pi \,\|\, \mu\otimes\nu)`
to the transport problem gives an optimal plan of Gibbs form,

.. math::
    \pi_{ij} = \mu_i\nu_j\exp\Bigl(\frac{s\,c_{ij} - a_i - b_j - h_i(y_j - x_i)}{\varepsilon}\Bigr),

with :math:`s = +1` for the upper bound and :math:`-1` for the lower. The
potentials are found by block coordinate ascent on the dual. :math:`b` and
:math:`a` have closed forms (log-sum-exp), as in Sinkhorn's algorithm. Each
:math:`h_i` solves a strictly monotone one-dimensional equation, the
martingale condition of row :math:`i`, by safeguarded Newton steps. All
updates are in the log domain. :math:`\varepsilon` is decreased geometrically
with warm starts (ε-scaling). As :math:`\varepsilon \to 0` the value converges to
the linear-programming bound of :mod:`spxa.vol.mot`, with an error of order
:math:`\varepsilon\log(nm)`. Each iteration costs :math:`O(nm)` and needs no
LP solver, so large grids are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import logsumexp

from spxa.vol.marginals import DiscreteMarginal

__all__ = ["SinkhornResult", "mot_sinkhorn"]


@dataclass(frozen=True)
class SinkhornResult:
    """Entropic MOT solution: the plan's value, the plan, potentials and the residuals reached."""

    value: float
    plan: np.ndarray
    a: np.ndarray
    b: np.ndarray
    h: np.ndarray
    eps: float
    iterations: int
    marginal_error: float
    martingale_error: float


def mot_sinkhorn(
    mu: DiscreteMarginal,
    nu: DiscreteMarginal,
    payoff: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sense: str = "max",
    eps: float = 1e-3,
    eps_start: Optional[float] = None,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> SinkhornResult:
    """Entropic martingale transport of ``mu`` to ``nu`` for ``payoff(x, y)``.

    The penalty is scaled from ``eps_start`` (default: the payoff's range) down to
    ``eps``, halving each stage. At the final ``eps`` the iteration stops when the
    second marginal is matched to ``tol``. ``value`` is :math:`E_\\pi[c]` under the
    returned plan.
    """
    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")
    if eps <= 0:
        raise ValueError("eps must be positive")
    keep_x, keep_y = mu.weights > 0, nu.weights > 0
    x, y = mu.points[keep_x], nu.points[keep_y]
    lmu, lnu = np.log(mu.weights[keep_x]), np.log(nu.weights[keep_y])
    C = np.broadcast_to(np.asarray(payoff(x[:, None], y[None, :]), float), (x.size, y.size))
    s = 1.0 if sense == "max" else -1.0
    S = s * C
    D = y[None, :] - x[:, None]
    a, b, h = np.zeros(x.size), np.zeros(y.size), np.zeros(x.size)
    e = float(eps_start) if eps_start is not None else max(float(np.ptp(C)), eps)
    dmax = float(np.abs(D).max())
    it, err = 0, np.inf
    while True:
        e = max(e, eps)
        final = e == eps
        for _ in range(max_iter if final else 50):
            it += 1
            b = e * logsumexp(lmu[:, None] + (S - a[:, None] - h[:, None] * D) / e, axis=0)
            base = lnu[None, :] + (S - b[None, :]) / e
            for _ in range(8):  # Newton on each row's martingale condition
                z = base - h[:, None] * D / e
                w = np.exp(z - z.max(axis=1, keepdims=True))
                w /= w.sum(axis=1, keepdims=True)
                m1 = (w * D).sum(axis=1)
                var = np.maximum((w * D * D).sum(axis=1) - m1 * m1, 1e-300)
                cap = 20.0 * e / dmax  # moves any exponent by at most 20
                step = np.clip(e * m1 / var, -cap, cap)
                h += step
                if np.abs(m1).max() < 1e-14:
                    break
            a = e * logsumexp(base - h[:, None] * D / e, axis=1)
            logpi = lmu[:, None] + lnu[None, :] + (S - a[:, None] - b[None, :] - h[:, None] * D) / e
            err = float(np.abs(np.exp(logsumexp(logpi, axis=0)) - np.exp(lnu)).max())
            if final and err < tol:
                break
        if final:
            break
        e *= 0.5
    plan_k = np.exp(logpi)
    mart = float(np.abs((plan_k * D).sum(axis=1)).max())
    plan = np.zeros((mu.points.size, nu.points.size))
    plan[np.ix_(keep_x, keep_y)] = plan_k
    return SinkhornResult(
        float((plan_k * C).sum()), plan, s * a, s * b, s * h, float(eps), it, err, mart
    )
