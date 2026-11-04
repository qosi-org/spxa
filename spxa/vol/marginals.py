r"""
Discrete marginal laws for martingale optimal transport.

Vanilla prices at one expiry determine the law of :math:`S_T/F_T`
(Breeden–Litzenberger). On an evenly spaced strike grid, the law whose call
function interpolates the quotes linearly puts the second differences of
the prices on the nodes. Two such laws admit a martingale coupling if and
only if they are in convex order (Strassen): equal means and ordered call
prices at every strike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

__all__ = ["DiscreteMarginal", "marginal_from_calls"]


@dataclass(frozen=True)
class DiscreteMarginal:
    """A probability law on finitely many points."""

    points: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        x, p = np.asarray(self.points, float), np.asarray(self.weights, float)
        if x.shape != p.shape or x.ndim != 1:
            raise ValueError("points and weights must be 1-d arrays of the same length")
        if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
            raise ValueError("weights must be non-negative and sum to one")
        object.__setattr__(self, "points", x)
        object.__setattr__(self, "weights", p)

    @property
    def mean(self) -> float:
        return float(self.weights @ self.points)

    def call(self, k: Any) -> Any:
        r""":math:`E[(X - k)^+]`."""
        kk = np.asarray(k, float)
        out = np.maximum(self.points[None, :] - kk.reshape(-1, 1), 0.0) @ self.weights
        return out.reshape(kk.shape) if kk.ndim else float(out[0])

    def expect(self, f: Callable[[np.ndarray], np.ndarray]) -> float:
        return float(self.weights @ f(self.points))

    def precedes(self, other: "DiscreteMarginal", tol: float = 1e-12) -> bool:
        r"""Convex order :math:`\mu \le_{cx} \nu`: equal means and calls ordered at every kink of both."""
        if abs(self.mean - other.mean) > tol:
            return False
        k = np.union1d(self.points, other.points)
        return bool(np.all(self.call(k) <= other.call(k) + tol))


def marginal_from_calls(strikes: Any, calls: Any, floor: Any = None) -> DiscreteMarginal:
    r"""The law on the strike grid whose call prices interpolate ``calls`` linearly.

    ``strikes`` is an evenly spaced grid in forward units and ``calls`` the
    undiscounted prices :math:`E[(X - k)^+]` of a unit-mean law. The ends are
    clamped to :math:`c(k_0) = 1 - k_0` and :math:`c(k_n) = 0`, so the grid must
    cover the support up to negligible mass. Weights are second differences,
    followed by an exponential tilt that restores the unit mean exactly.
    Passing the call prices of an earlier expiry as ``floor`` replaces ``calls``
    by the pointwise maximum, which is still convex. That repairs rounding-level
    calendar violations, so the result dominates the earlier law in convex order.
    """
    k = np.asarray(strikes, float)
    c = np.array(calls, dtype=float)
    h = np.diff(k)
    if k.ndim != 1 or k.size < 3 or np.any(h <= 0) or np.ptp(h) > 1e-9 * h[0]:
        raise ValueError("strikes must be an increasing, evenly spaced grid")
    if floor is not None:
        c = np.maximum(c, np.asarray(floor, float))
    c[0], c[-1] = 1.0 - k[0], 0.0
    ext = np.concatenate([[c[0] + h[0]], c, [0.0]])
    p = np.maximum((ext[:-2] - 2.0 * ext[1:-1] + ext[2:]) / h[0], 0.0)
    p /= p.sum()
    # clipping and truncation leave a tiny mean error; an exponential tilt
    # p_i e^{beta k_i} restores the unit mean exactly and keeps the weights positive
    if not k[0] < 1.0 < k[-1]:
        raise ValueError("the grid must contain the forward 1 in its interior")
    logp = np.log(np.where(p > 0, p, 1e-300))

    def tilt(beta: float) -> np.ndarray:
        z = logp + beta * (k - 1.0)
        w = np.exp(z - z.max())
        return w / w.sum()

    beta = brentq(lambda b: float(tilt(b) @ k) - 1.0, -50.0, 50.0, xtol=1e-15)
    p = tilt(beta)
    p *= p > 1e-300
    return DiscreteMarginal(k, p / p.sum())
