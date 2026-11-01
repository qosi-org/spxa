r"""Grid-convergence analysis and Richardson extrapolation.

If a scheme has error :math:`V_h = V + C h^p + o(h^p)`, then with refinement
ratio :math:`\rho` (:math:`h \to h/\rho`):

* with a known reference, the observed order is
  :math:`p \approx \log(e_h / e_{h/\rho}) / \log\rho`;
* without one, three successive levels give
  :math:`p \approx \log\big(|V_h - V_{h/\rho}| / |V_{h/\rho} - V_{h/\rho^2}|\big)/\log\rho`;
* Richardson extrapolation
  :math:`V^\star = V_{h/\rho} + (V_{h/\rho} - V_h)/(\rho^p - 1)`
  removes the leading error term.

These tools are used throughout the test-suite to *verify* the design order
of every solver, not just its accuracy at one resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np


def observed_order(errors: Sequence[float], ratio: float = 2.0) -> np.ndarray:
    """Observed orders ``log(e_k / e_{k+1}) / log(ratio)`` from errors against a reference."""
    e = np.abs(np.asarray(errors, dtype=float))
    if e.size < 2:
        raise ValueError("need at least two error levels")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e[:-1] / e[1:]) / np.log(ratio)


def observed_order_from_values(values: Sequence[float], ratio: float = 2.0) -> np.ndarray:
    """Observed orders from successive differences of at least three levels."""
    v = np.asarray(values, dtype=float)
    if v.size < 3:
        raise ValueError("need at least three levels")
    d = np.abs(np.diff(v))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(d[:-1] / d[1:]) / np.log(ratio)


def richardson_extrapolate(
    coarse: Any, fine: Any, ratio: float = 2.0, order: float = 2.0
) -> np.ndarray | float:
    """Richardson extrapolation ``fine + (fine - coarse) / (ratio**order - 1)``."""
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    factor = ratio**order - 1.0
    out = (
        np.asarray(fine, dtype=float)
        + (np.asarray(fine, dtype=float) - np.asarray(coarse, dtype=float)) / factor
    )
    return float(out) if out.ndim == 0 else out


@dataclass
class ConvergenceStudy:
    """Result of :func:`convergence_study`.

    Attributes
    ----------
    levels : list
        The refinement levels that were evaluated.
    values : np.ndarray
        Computed values per level.
    reference : float or None
        Exact value, if supplied.
    errors : np.ndarray or None
        ``values - reference``.
    orders : np.ndarray
        Observed orders (from errors if a reference is known, otherwise
        from successive differences).
    extrapolated : np.ndarray
        Richardson-extrapolated values from consecutive levels.
    ratio, order : float
        Refinement ratio and assumed order used for extrapolation.
    """

    levels: list[Any]
    values: np.ndarray
    reference: Optional[float]
    errors: Optional[np.ndarray]
    orders: np.ndarray
    extrapolated: np.ndarray
    ratio: float
    order: float

    @property
    def best_estimate(self) -> float:
        """Richardson value from the two finest levels."""
        return float(self.extrapolated[-1])

    def __str__(self) -> str:
        lines = [f"{'level':>18} {'value':>18} {'error':>12} {'order':>7}"]
        for i, (lvl, v) in enumerate(zip(self.levels, self.values)):
            err = "" if self.errors is None else f"{self.errors[i]:12.3e}"
            if self.errors is not None:
                order = f"{self.orders[i - 1]:7.3f}" if i >= 1 else ""
            else:
                order = f"{self.orders[i - 2]:7.3f}" if i >= 2 else ""
            lines.append(f"{str(lvl):>18} {v:18.10f} {err:>12} {order:>7}")
        lines.append(f"Richardson (p={self.order:g}): {self.best_estimate:.10f}")
        return "\n".join(lines)


def convergence_study(
    solver: Callable[[Any], float],
    levels: Sequence[Any],
    reference: Optional[float] = None,
    ratio: float = 2.0,
    order: float = 2.0,
) -> ConvergenceStudy:
    """Evaluate ``solver(level)`` on successive refinements and analyse convergence.

    Parameters
    ----------
    solver : callable
        Maps a refinement level (e.g. a number of nodes, or a tuple of
        space and time resolutions) to a scalar result.
    levels : sequence
        Refinement levels, each finer than the previous by ``ratio``.
    reference : float, optional
        Exact value. Without it, orders are estimated from differences.
    ratio : float
        Refinement ratio between consecutive levels.
    order : float
        Assumed order for Richardson extrapolation.
    """
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("need at least two levels")
    values = np.array([float(solver(lvl)) for lvl in levels])
    if reference is not None:
        errors = values - float(reference)
        orders = observed_order(errors, ratio)
    else:
        errors = None
        orders = observed_order_from_values(values, ratio) if values.size >= 3 else np.array([])
    extrap = np.atleast_1d(richardson_extrapolate(values[:-1], values[1:], ratio, order))
    return ConvergenceStudy(
        levels=levels,
        values=values,
        reference=reference,
        errors=errors,
        orders=orders,
        extrapolated=extrap,
        ratio=ratio,
        order=order,
    )
