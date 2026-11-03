r"""Calibration of SVI slices.

Quasi-explicit SVI (Zeliade; De Marco & Martini 2009)
    In the variable :math:`y = (k - m)/\sigma` raw SVI reads

    .. math:: w = a + d\,y + c\sqrt{y^2 + 1}, \qquad c = b\sigma,\; d = \rho b\sigma,

    which is *linear* in :math:`(a, c, d)` for fixed :math:`(m, \sigma)`.
    The admissible set :math:`0 \le |d| \le c`, :math:`c + |d| \le s\sigma`
    (wing slopes :math:`b(1 \pm \rho) \le s`) and :math:`0 \le a \le \max w`
    becomes a box in :math:`(a, u, v) = (a, c + d, c - d)`:
    :math:`u, v \in [0, s\sigma]`. The inner problem is therefore a
    bounded linear least-squares problem solved exactly, and only the two
    parameters :math:`(m, \sigma)` are searched numerically.

    ``max_slope`` :math:`s` defaults to 2, Lee's moment bound on the
    asymptotic slope of total variance; De Marco & Martini use the weaker
    Rogers–Tehranchi bound :math:`s = 4` (pass ``max_slope=4``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import lsq_linear, minimize

from spxa.vol.svi import SVIRaw

__all__ = ["SVIFit", "calibrate_svi", "svi_inner_fit"]


def _arrays(k: Any, w: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kk = np.asarray(k, float).ravel()
    ww = np.asarray(w, float).ravel()
    if kk.shape != ww.shape or kk.size < 5:
        raise ValueError("need at least 5 (k, w) pairs of equal length")
    if np.any(ww <= 0) or not np.all(np.isfinite(ww)):
        raise ValueError("total variances must be positive and finite")
    if weights is None:
        sw = np.ones_like(kk)
    else:
        wt = np.asarray(weights, float).ravel()
        if wt.shape != kk.shape or np.any(wt < 0):
            raise ValueError("weights must be non-negative, one per point")
        sw = np.sqrt(wt)
    return kk, ww, sw


def svi_inner_fit(
    k: Any,
    w: Any,
    m: float,
    sigma: float,
    weights: Any = None,
    max_slope: float = 2.0,
) -> tuple[SVIRaw, float]:
    """Best raw SVI for fixed ``(m, sigma)``; returns ``(slice, weighted SSE)``."""
    kk, ww, sw = _arrays(k, w, weights)
    return _inner(kk, ww, sw, m, sigma, max_slope)


def _inner(
    kk: np.ndarray, ww: np.ndarray, sw: np.ndarray, m: float, sigma: float, max_slope: float
) -> tuple[SVIRaw, float]:
    y = (kk - m) / sigma
    z = np.sqrt(y * y + 1.0)
    # w = a + u (z + y)/2 + v (z - y)/2 with u = c + d, v = c - d
    A = np.column_stack([np.ones_like(y), 0.5 * (z + y), 0.5 * (z - y)]) * sw[:, None]
    hi = max_slope * sigma
    res = lsq_linear(A, ww * sw, bounds=([0.0, 0.0, 0.0], [ww.max(), hi, hi]), method="bvls")
    a, u, v = (float(x) for x in res.x)
    c, d = 0.5 * (u + v), 0.5 * (u - v)
    sse = float(np.sum((A @ res.x - ww * sw) ** 2))
    if c <= 0.0:
        # flat smile: any rho; choose rho = 0 and a vanishing b
        return SVIRaw(a, 0.0, 0.0, m, sigma), sse
    rho = float(np.clip(d / c, -1.0 + 1e-15, 1.0 - 1e-15))
    return SVIRaw(a, c / sigma, rho, m, sigma), sse


@dataclass(frozen=True)
class SVIFit:
    """Result of :func:`calibrate_svi`."""

    params: SVIRaw
    rmse: float
    sse: float
    nfev: int
    success: bool


def calibrate_svi(
    k: Any,
    w: Any,
    weights: Any = None,
    max_slope: float = 2.0,
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
    m_bounds: Optional[tuple[float, float]] = None,
    n_starts: int = 4,
) -> SVIFit:
    """Fit raw SVI to total variances ``w`` at log-moneyness ``k``.

    The outer search over :math:`(m, \\sigma)` starts from the best points
    of a coarse grid and refines each with bounded Nelder–Mead; the inner
    linear problem is solved exactly (see the module docstring).
    """
    kk, ww, sw = _arrays(k, w, weights)
    if max_slope <= 0:
        raise ValueError("max_slope must be positive")
    span = float(kk.max() - kk.min())
    if m_bounds is None:
        m_bounds = (float(kk.min()) - span, float(kk.max()) + span)
    lo = np.array([m_bounds[0], np.log(sigma_bounds[0])])
    hi = np.array([m_bounds[1], np.log(sigma_bounds[1])])
    nfev = 0

    def obj(x: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        m, ls = np.clip(x, lo, hi)
        return _inner(kk, ww, sw, float(m), float(np.exp(ls)), max_slope)[1]

    ms = np.linspace(
        m_bounds[0] + 0.25 * (m_bounds[1] - m_bounds[0]),
        m_bounds[1] - 0.25 * (m_bounds[1] - m_bounds[0]),
        9,
    )
    ls = np.linspace(lo[1], hi[1], 9)
    grid = sorted(((obj(np.array([m, s])), m, s) for m in ms for s in ls))
    best_x, best_f = np.array(grid[0][1:]), grid[0][0]
    ok = False
    for _, m0, s0 in grid[:n_starts]:
        r = minimize(
            obj,
            np.array([m0, s0]),
            method="Nelder-Mead",
            bounds=list(zip(lo, hi)),
            options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 4000},
        )
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
        ok = ok or bool(r.success)
    m, s = np.clip(best_x, lo, hi)
    params, sse = _inner(kk, ww, sw, float(m), float(np.exp(s)), max_slope)
    wsum = float(np.sum(sw * sw))
    return SVIFit(params, float(np.sqrt(sse / wsum)) if wsum > 0 else 0.0, sse, nfev, ok)
