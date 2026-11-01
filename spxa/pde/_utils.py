"""Internal helpers shared by the :mod:`spxa.pde` solvers.

Coefficients of the PDEs solved in this package may be given as

* a scalar (constant coefficient),
* an array with one value per grid node, or
* a callable ``f(x, tau)`` returning an array broadcastable to ``x``,
  where ``tau`` is the time to maturity (the PDE time variable).

The helpers below normalise these representations and centralise argument
validation so every solver reports errors the same way.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

Coefficient = Union[float, int, np.ndarray, Callable[[np.ndarray, float], np.ndarray]]
"""Type alias for a PDE coefficient (scalar, nodal array or ``f(x, tau)``)."""


def evaluate_coefficient(coef: Coefficient, x: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate a coefficient on the nodes ``x`` at PDE time ``tau``.

    Parameters
    ----------
    coef : Coefficient
        Scalar, array of the same length as ``x`` or callable ``f(x, tau)``.
    x : np.ndarray
        Grid nodes.
    tau : float
        Time to maturity.

    Returns
    -------
    np.ndarray
        Array of shape ``x.shape`` (always a fresh float array).
    """
    if callable(coef):
        val = np.asarray(coef(x, tau), dtype=float)
    else:
        val = np.asarray(coef, dtype=float)
    out = np.broadcast_to(val, x.shape).astype(float, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError("PDE coefficient evaluated to non-finite values")
    return out


def check_positive(name: str, value: float, allow_zero: bool = False) -> float:
    """Validate that ``value`` is a finite positive (or non-negative) scalar."""
    v = float(value)
    if not np.isfinite(v) or (v < 0.0 if allow_zero else v <= 0.0):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {bound} number, got {value!r}")
    return v


def check_option_type(option: str) -> str:
    """Normalise an option type string to ``'call'`` or ``'put'``."""
    opt = str(option).lower()
    if opt in ("c", "call"):
        return "call"
    if opt in ("p", "put"):
        return "put"
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def cubic_interpolate(nodes: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Local four-point (cubic Lagrange) interpolation on a monotone grid.

    The stencil is the two nodes on either side of each query point (clamped
    at the ends of the grid). For smooth data the error is ``O(h^4)``; unlike
    a global spline, the interpolant is local, so a kink in the data only
    affects the interval that contains it.

    Query points outside ``[nodes[0], nodes[-1]]`` are linearly extrapolated
    from the two outermost nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    xq = np.atleast_1d(np.asarray(x, dtype=float))
    n = nodes.size
    if n < 4:
        return np.interp(xq, nodes, values)
    idx = np.searchsorted(nodes, xq, side="right") - 1
    start = np.clip(idx - 1, 0, n - 4)
    out = np.zeros_like(xq)
    for k in range(4):
        xk = nodes[start + k]
        lk = np.ones_like(xq)
        for m in range(4):
            if m == k:
                continue
            xm = nodes[start + m]
            lk *= (xq - xm) / (xk - xm)
        out += lk * values[start + k]
    left = xq < nodes[0]
    right = xq > nodes[-1]
    if np.any(left):
        slope = (values[1] - values[0]) / (nodes[1] - nodes[0])
        out[left] = values[0] + slope * (xq[left] - nodes[0])
    if np.any(right):
        slope = (values[-1] - values[-2]) / (nodes[-1] - nodes[-2])
        out[right] = values[-1] + slope * (xq[right] - nodes[-1])
    return out
