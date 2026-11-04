r"""Diagnostics for comparing curve interpolation schemes.

Hagan & West (2006) judge a yield-curve interpolation by:

* whether the **forwards are positive**;
* how **smooth** the forwards are, here the roughness
  :math:`\int_0^T f'(t)^2\,dt`;
* how **local** it is: a bump to one input should move the curve only
  near that input. :func:`bump_locality` bumps each node's zero rate and
  reports the furthest node interval that moves;
* how **stable** the forwards are: the largest forward move per unit
  input bump.

:func:`compare_schemes` tabulates these on one set of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spxa.rates.interpolation import InterpolationScheme, PiecewiseCurve

__all__ = ["SchemeDiagnostics", "forward_roughness", "bump_locality", "compare_schemes"]


def forward_roughness(
    curve: PiecewiseCurve, t_max: float, points: int = 20001, t_min: float = 0.0
) -> float:
    r""":math:`\int_{t_{min}}^{t_{max}} f'(t)^2 dt` by finite differences of the analytic forward on a fine grid.

    For schemes with discontinuous forwards (log-linear discounts) the
    integral is infinite. The grid value then grows like ``points`` and is
    only useful as a flag.
    """
    t = np.linspace(t_min, t_max, points)
    f = np.asarray(curve.instantaneous_forward(t), float)
    df = np.diff(f) / np.diff(t)
    return float(np.sum(df**2 * np.diff(t)))


def bump_locality(
    times: Sequence[float],
    rates: Sequence[float],
    scheme: InterpolationScheme,
    bump: float = 1e-4,
    tol: float = 1e-12,
    points_per_interval: int = 50,
) -> tuple[int, float]:
    """``(spread, sensitivity)`` of the forward curve to single-node zero-rate bumps.

    ``spread`` is the largest number of node intervals between a bumped node
    and a point where the forward moved by more than ``tol``. It is 1 when
    a bump only touches the two adjacent intervals. ``sensitivity`` is the
    largest forward move per unit zero-rate bump.
    """
    t = np.asarray(times, float)
    r = np.asarray(rates, float)
    knots = np.concatenate([[0.0], t])
    grid = np.concatenate(
        [
            np.linspace(a, b, points_per_interval, endpoint=False)[1:]
            for a, b in zip(knots, knots[1:])
        ]
    )
    interval = np.searchsorted(knots, grid) - 1  # interval index of each grid point
    base = np.asarray(
        PiecewiseCurve.from_zero_rates(t, r, scheme).instantaneous_forward(grid), float
    )
    spread, sens = 0, 0.0
    for j in range(t.size):
        rb = r.copy()
        rb[j] += bump
        f = np.asarray(
            PiecewiseCurve.from_zero_rates(t, rb, scheme).instantaneous_forward(grid), float
        )
        moved = np.abs(f - base) > tol
        sens = max(sens, float(np.max(np.abs(f - base))) / bump)
        if np.any(moved):
            # node j sits between intervals j and j + 1
            d = np.where(interval[moved] <= j, j - interval[moved] + 1, interval[moved] - j)
            spread = max(spread, int(d.max()))
    return spread, sens


@dataclass(frozen=True)
class SchemeDiagnostics:
    scheme: str
    min_forward: float
    roughness: float
    locality: int
    sensitivity: float


def compare_schemes(
    times: Sequence[float], rates: Sequence[float], schemes: Sequence[InterpolationScheme]
) -> list[SchemeDiagnostics]:
    """Positivity, roughness, locality and bump sensitivity of each scheme on the same nodes."""
    t = np.asarray(times, float)
    out = []
    for s in schemes:
        c = PiecewiseCurve.from_zero_rates(t, rates, s)
        f = np.asarray(c.instantaneous_forward(np.linspace(0.0, t[-1], 20001)), float)
        loc, sens = bump_locality(t, rates, s)
        out.append(
            SchemeDiagnostics(
                repr(s), float(f.min()), forward_roughness(c, float(t[-1])), loc, sens
            )
        )
    return out
