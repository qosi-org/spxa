r"""
Robust (model-free) pricing of forward-start claims and the superhedge as trades.

Claims on :math:`(S_{T_1}, S_{T_2})` are written in forward units
:math:`X = S_{T_1}/F_1` and :math:`Y = S_{T_2}/F_2`. A forward-start call
:math:`(S_{T_2} - kS_{T_1})^+` is :math:`F_2\,(Y - k\,\tfrac{F_1}{F_2}X)^+`.
:func:`robust_bounds` turns the normalised MOT bounds into money. The dual
of :func:`~spxa.vol.mot.mot_solve` is a function on the grid.
:func:`static_replication` writes any function that is piecewise linear on
the grid as cash, a forward and a strip of calls,

.. math:: f(k) = f(k_0) + f'(k_0^+)(k - k_0) + \sum_{i \ge 1} \bigl(f'(k_i^+) - f'(k_i^-)\bigr)(k - k_i)^+,

so the superhedge becomes concrete trades: vanillas on :math:`T_1` and
:math:`T_2`, and :math:`h(X)` forwards entered at :math:`T_1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from spxa.vol.marginals import DiscreteMarginal, surface_marginals
from spxa.vol.mot import MOTResult, mot_solve

__all__ = [
    "StaticReplication",
    "forward_start_call",
    "forward_start_straddle",
    "robust_bounds",
    "static_replication",
    "superhedge_pnl",
]

Payoff = Callable[[np.ndarray, np.ndarray], np.ndarray]


def forward_start_call(k: float, ratio: float = 1.0) -> Payoff:
    r"""Normalised forward-start call :math:`(Y - k\,r\,X)^+` with :math:`r = F_1/F_2`."""
    return lambda x, y: np.maximum(y - k * ratio * x, 0.0)


def forward_start_straddle(k: float = 1.0, ratio: float = 1.0) -> Payoff:
    r"""Normalised forward-start straddle :math:`|Y - k\,r\,X|`."""
    return lambda x, y: np.abs(y - k * ratio * x)


def robust_bounds(
    surface: Any, T1: float, T2: float, payoff: Payoff, strikes: Any, scale: float = 1.0
) -> tuple[float, float, MOTResult, MOTResult]:
    """Model-free price bounds of ``scale * payoff(X, Y)`` given a surface.

    ``scale`` converts normalised units to money, e.g. ``F2 * df(T2)`` for the
    forward-start payoffs above. Returns ``(lower, upper, sub, super)``, where
    the last two are the LP solutions with their hedges.
    """
    mu, nu = surface_marginals(surface, [T1, T2], strikes)
    lo, hi = mot_solve(mu, nu, payoff, "min"), mot_solve(mu, nu, payoff, "max")
    return scale * lo.value, scale * hi.value, lo, hi


@dataclass(frozen=True)
class StaticReplication:
    """``cash + forward * k + sum(calls * (k - strikes)^+)``."""

    cash: float
    forward: float
    strikes: np.ndarray
    calls: np.ndarray

    def __call__(self, k: Any) -> Any:
        kk = np.asarray(k, float)
        out = (
            self.cash
            + self.forward * kk
            + np.maximum(kk[..., None] - self.strikes, 0.0) @ self.calls
        )
        return out

    def price(self, marginal: DiscreteMarginal) -> float:
        """Cost against a law: cash plus forward (unit mean) plus call prices."""
        return float(
            self.cash + self.forward * marginal.mean + marginal.call(self.strikes) @ self.calls
        )


def static_replication(points: Any, values: Any) -> StaticReplication:
    """Cash, forward and call weights reproducing the linear interpolant of ``values`` on ``points``."""
    k, f = np.asarray(points, float), np.asarray(values, float)
    if k.ndim != 1 or k.shape != f.shape or k.size < 2 or np.any(np.diff(k) <= 0):
        raise ValueError("points must be increasing and match values")
    slopes = np.diff(f) / np.diff(k)
    kinks = np.diff(slopes)
    return StaticReplication(float(f[0] - slopes[0] * k[0]), float(slopes[0]), k[1:-1], kinks)


def superhedge_pnl(result: MOTResult, payoff: Payoff, x: Any, y: Any) -> np.ndarray:
    r"""Pathwise hedge minus payoff, :math:`\varphi(x) + \psi(y) + h(x)(y - x) - c(x, y)`.

    Off the grid, the potentials are interpolated linearly, as traded by the
    call strips of :func:`static_replication`. For an upper bound the result is
    non-negative on the grid. Between nodes it can dip by an amount of the order
    of the grid spacing times the payoff's curvature. For a lower bound the
    sign is reversed.
    """
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    phi = np.interp(xx, result.mu.points, result.phi)
    h = np.interp(xx, result.mu.points, result.h)
    psi = np.interp(yy, result.nu.points, result.psi)
    hedge = phi + psi + h * (yy - xx)
    pnl = hedge - payoff(xx, yy)
    return pnl if result.sense == "max" else -pnl
