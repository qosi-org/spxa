r"""Andreasen–Huge one-step local-volatility interpolation.

Andreasen & Huge (*Volatility interpolation*, Risk, March 2011) work with
normalised calls :math:`c(x, t) = C/(D_t F_t)` in normalised strike
:math:`x = K/F_t`. Under deterministic rates and proportional dividends
they solve the driftless Dupire equation

.. math:: \partial_t c = \tfrac12\,\vartheta^2(x, t)\,x^2\,\partial_{xx} c,
          \qquad c(x, 0) = (1 - x)^+,

with **one fully implicit step per expiry interval**:

.. math:: \Bigl[I - \tfrac12 (t_i - t_{i-1})\,\vartheta_i^2(x)\,x^2 D_{xx}\Bigr] c_i = c_{i-1}.

:math:`\vartheta_i` is piecewise constant in :math:`x`, with one level per
calibration node. The matrix is an M-matrix (positive diagonal,
non-positive off-diagonals, diagonally dominant), so its inverse is
non-negative and the step

* preserves convexity in :math:`x` and the bounds :math:`(1 - x)^+ \le c \le 1`,
  so no butterfly or vertical-spread arbitrage appears, and
* gives :math:`c_i \ge c_{i-1}`, so prices increase with maturity: no calendar
  arbitrage.

Between expiries the same step with a partial time step interpolates the
surface, and it stays arbitrage-free in time. :math:`D_{xx}` is the standard
three-point second difference on a non-uniform grid in :math:`x`, with
Dirichlet values :math:`c = 1 - x` at the left end and :math:`c = 0` at the
right end of a wide grid.

Normalised calls are interpolated between grid nodes by a cubic spline in
:math:`y = \log x`, and converted to implied total variance for
:class:`~spxa.vol.surface.VolSurface`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from spxa.vol.conventions import total_variance_from_price

__all__ = [
    "AndreasenHugeFit",
    "AndreasenHugeSurface",
    "andreasen_huge_grid",
    "calibrate_andreasen_huge",
    "piecewise_constant_vol",
]


def andreasen_huge_grid(y_max: float = 4.0, n: int = 401, concentration: float = 0.5) -> np.ndarray:
    """Grid in :math:`x = e^y`, dense near the money (sinh-spaced in ``y``)."""
    if n < 5 or y_max <= 0:
        raise ValueError("need n >= 5 and y_max > 0")
    u = np.linspace(-1.0, 1.0, n)
    c = float(concentration)
    y = c * np.sinh(u * np.arcsinh(y_max / c))
    return np.exp(y)


def piecewise_constant_vol(nodes_y: Sequence[float], levels: Sequence[float], y: Any) -> np.ndarray:
    """Levels on the cells between midpoints of ``nodes_y``, flat beyond the ends."""
    nodes = np.asarray(nodes_y, float)
    lev = np.asarray(levels, float)
    if nodes.shape != lev.shape or nodes.size == 0:
        raise ValueError("need one level per node")
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("nodes must be strictly increasing")
    edges = 0.5 * (nodes[1:] + nodes[:-1])
    return lev[np.searchsorted(edges, np.asarray(y, float), side="right")]


def _implicit_step(x: np.ndarray, c_prev: np.ndarray, var_dt: np.ndarray) -> np.ndarray:
    """Solve ``[I - 1/2 var_dt x^2 D_xx] c = c_prev`` with Dirichlet ends."""
    n = x.size
    h = np.diff(x)
    hm, hp = h[:-1], h[1:]  # left and right spacings at interior nodes
    a = 0.5 * var_dt[1:-1] * x[1:-1] ** 2
    lo = -a * 2.0 / (hm * (hm + hp))
    up = -a * 2.0 / (hp * (hm + hp))
    diag = np.ones(n)
    diag[1:-1] = 1.0 - lo - up
    ab = np.zeros((3, n))
    ab[1] = diag
    ab[0, 2:] = up  # super-diagonal
    ab[2, :-2] = lo  # sub-diagonal
    rhs = c_prev.copy()
    rhs[0], rhs[-1] = 1.0 - x[0], 0.0
    return solve_banded((1, 1), ab, rhs)


class AndreasenHugeSurface:
    """Arbitrage-free surface from Andreasen–Huge steps with given local-vol levels.

    Parameters
    ----------
    expiries
        Increasing positive expiries :math:`t_i`.
    nodes
        Per expiry, increasing log-moneyness nodes :math:`y_{ij}` of the
        piecewise-constant :math:`\\vartheta_i`.
    levels
        Per expiry, the levels :math:`\\vartheta_{ij} > 0`.
    grid
        Normalised strikes :math:`x` (default :func:`andreasen_huge_grid`).
    """

    def __init__(
        self,
        expiries: Sequence[float],
        nodes: Sequence[Sequence[float]],
        levels: Sequence[Sequence[float]],
        grid: Optional[np.ndarray] = None,
    ) -> None:
        T = np.asarray(expiries, float)
        if T.ndim != 1 or T.size == 0 or T[0] <= 0 or np.any(np.diff(T) <= 0):
            raise ValueError("expiries must be positive and strictly increasing")
        if len(nodes) != T.size or len(levels) != T.size:
            raise ValueError("need nodes and levels for every expiry")
        self.expiries = T
        self.nodes = [np.asarray(n, float) for n in nodes]
        self.levels = [np.asarray(v, float) for v in levels]
        for lv in self.levels:
            if np.any(lv <= 0):
                raise ValueError("local-vol levels must be positive")
        self.x = andreasen_huge_grid() if grid is None else np.asarray(grid, float)
        if np.any(np.diff(self.x) <= 0) or self.x[0] <= 0:
            raise ValueError("grid must be positive and strictly increasing")
        self.y = np.log(self.x)
        self._var = [
            piecewise_constant_vol(n, v, self.y) ** 2 for n, v in zip(self.nodes, self.levels)
        ]
        self.calls = [np.maximum(1.0 - self.x, 0.0)]
        t_prev = 0.0
        for t, var in zip(T, self._var):
            self.calls.append(_implicit_step(self.x, self.calls[-1], var * (t - t_prev)))
            t_prev = t
        self._splines: dict[float, CubicSpline] = {}

    def grid_calls(self, t: float) -> np.ndarray:
        """Normalised calls on the grid at time ``t`` (exact steps at expiries, partial steps between)."""
        if t < 0:
            raise ValueError("negative time")
        if t == 0:
            return self.calls[0]
        i = int(np.searchsorted(self.expiries, t, side="left"))
        if i < self.expiries.size and abs(self.expiries[i] - t) <= 1e-14 * max(1.0, t):
            return self.calls[i + 1]
        j = min(i, self.expiries.size - 1)  # beyond the last expiry, keep its levels
        t0 = 0.0 if i == 0 else self.expiries[i - 1]
        return _implicit_step(self.x, self.calls[i], self._var[j] * (t - t0))

    def normalised_call(self, y: Any, t: float) -> Any:
        """:math:`c(e^y, t)`, cubic-spline interpolated in ``y`` between grid nodes."""
        key = float(t)
        if key not in self._splines:
            self._splines[key] = CubicSpline(self.y, self.grid_calls(key))
        out = self._splines[key](np.asarray(y, float))
        return float(out) if np.ndim(out) == 0 else out

    def total_variance(self, k: Any, T: float) -> Any:
        """Implied total variance at log-moneyness ``k`` (a :class:`TotalVarianceModel`)."""
        kk = np.asarray(k, float)
        if T <= 0:
            out = np.zeros(kk.shape)
        else:
            c = np.asarray(self.normalised_call(kk, T), float)
            out = np.asarray(total_variance_from_price(c, 1.0, np.exp(kk), T), float)
        return float(out) if out.ndim == 0 else out


class AndreasenHugeFit:
    """Result of :func:`calibrate_andreasen_huge`."""

    def __init__(self, surface: AndreasenHugeSurface, errors: list[np.ndarray], nfev: int) -> None:
        self.surface = surface
        self.errors = errors
        self.nfev = nfev

    @property
    def max_error(self) -> float:
        """Largest absolute implied-volatility error over all quotes."""
        return max(float(np.max(np.abs(e))) for e in self.errors)

    @property
    def rmse(self) -> float:
        e = np.concatenate(self.errors)
        return float(np.sqrt(np.mean(e * e)))


def _node_subset(k: np.ndarray, max_nodes: Optional[int]) -> np.ndarray:
    if max_nodes is None or k.size <= max_nodes:
        return k.copy()
    idx = np.unique(np.round(np.linspace(0, k.size - 1, max_nodes)).astype(int))
    return k[idx]


def calibrate_andreasen_huge(
    expiries: Sequence[float],
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    grid: Optional[np.ndarray] = None,
    max_nodes: Optional[int] = None,
    weights: Optional[Sequence[Sequence[float]]] = None,
) -> AndreasenHugeFit:
    """Fit the local-vol levels expiry by expiry to implied volatilities.

    ``strikes`` are log-forward moneyness :math:`k = \\log(K/F_t)` per
    expiry and ``vols`` the Black implied volatilities. The nodes of
    :math:`\\vartheta_i` are the quoted strikes (or ``max_nodes`` of them,
    evenly spread). Expiry by expiry, a Levenberg–Marquardt least-squares
    fit in log levels minimises the weighted implied-volatility errors,
    starting from the previous slice's step, as in Andreasen–Huge.
    """
    from scipy.optimize import least_squares

    T = np.asarray(expiries, float)
    x = andreasen_huge_grid() if grid is None else np.asarray(grid, float)
    y = np.log(x)
    c_prev = np.maximum(1.0 - x, 0.0)
    t_prev = 0.0
    all_nodes, all_levels, errors = [], [], []
    nfev = 0
    for i, t in enumerate(T):
        k = np.asarray(strikes[i], float)
        v = np.asarray(vols[i], float)
        if k.shape != v.shape or np.any(np.diff(k) <= 0):
            raise ValueError("strikes must be increasing and match the vols")
        wts = np.ones_like(v) if weights is None else np.asarray(weights[i], float)
        nodes = _node_subset(k, max_nodes)
        dt = t - t_prev
        c_start = c_prev

        def model_vols(log_lev: np.ndarray) -> np.ndarray:
            var = piecewise_constant_vol(nodes, np.exp(log_lev), y) ** 2
            c = _implicit_step(x, c_start, var * dt)
            ck = CubicSpline(y, c)(k)
            w = np.asarray(total_variance_from_price(ck, 1.0, np.exp(k), t), float)
            return np.sqrt(np.maximum(w, 0.0) / t)

        def resid(log_lev: np.ndarray) -> np.ndarray:
            mv = model_vols(log_lev)
            return np.where(np.isfinite(mv), wts * (mv - v), 10.0)

        guess = np.log(np.interp(nodes, k, v))
        sol = least_squares(resid, guess, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        nfev += int(sol.nfev)
        lev = np.exp(sol.x)
        c_prev = _implicit_step(x, c_start, piecewise_constant_vol(nodes, lev, y) ** 2 * dt)
        t_prev = t
        all_nodes.append(nodes)
        all_levels.append(lev)
        errors.append(model_vols(sol.x) - v)
    surface = AndreasenHugeSurface(T, all_nodes, all_levels, grid=x)
    return AndreasenHugeFit(surface, errors, nfev)
