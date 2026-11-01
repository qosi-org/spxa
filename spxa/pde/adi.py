r"""
Two-dimensional convection–diffusion–reaction problems and ADI time stepping.

Problems are written in time to maturity :math:`\tau` as

.. math::
    u_\tau = a_{xx} u_{xx} + a_{xy} u_{xy} + a_{yy} u_{yy}
             + b_x u_x + b_y u_y + c\,u + f,

on a tensor grid :math:`x_0 < \dots < x_{n_x-1}`,
:math:`y_0 < \dots < y_{n_y-1}`. Unknowns are stored in C order,
``k = i * ny + j``.

Spatial discretisation (second order on non-uniform grids):

* central three-point stencils for :math:`u_x, u_{xx}` (and in :math:`y`);
  where the cell Péclet number exceeds one, the first derivative switches to
  the second-order upwind three-point stencil (as in in 't Hout & Foulon,
  2010, for large variance);
* the mixed derivative is the product of the two central first-derivative
  stencils (nine points), applied at interior nodes. At the edges it is
  omitted, which is exact when the cross derivative vanishes there (Neumann
  data independent of the other variable, or :math:`a_{xy} = 0` on the edge,
  as for Heston and two-asset Black–Scholes);
* edge conditions: :class:`Dirichlet2D`, :class:`Neumann2D` (ghost-point
  second derivative), :class:`Natural2D` (the equation itself with
  one-sided first derivatives, for degenerate edges such as :math:`v = 0`)
  and :class:`Linear2D` (the equation with :math:`u_{nn} = 0`, for far
  edges).

The operator is split as :math:`A = A_0 + A_1 + A_2` with :math:`A_0` the
mixed term and :math:`A_1, A_2` the :math:`x` and :math:`y` parts, each
carrying half of the reaction term. :class:`ADISolver` implements the
Douglas, Craig–Sneyd, modified Craig–Sneyd and Hundsdorfer–Verwer schemes
(in 't Hout & Welfert, 2007/2009) with implicit damping half-steps at the
start, and the Ikonen–Toivanen splitting for early exercise (Haentjens &
in 't Hout, 2015).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import splu

from spxa.pde.grid import Grid1D
from spxa.pde.operators import (
    central_first_weights,
    central_second_weights,
    one_sided_first_weights,
)

__all__ = [
    "Grid2D",
    "Dirichlet2D",
    "Neumann2D",
    "Natural2D",
    "Linear2D",
    "LinearPDE2D",
    "DiscreteOperator2D",
    "ADISolver",
    "ADIResult",
    "PDESolution2D",
]

Coefficient2D = Union[float, np.ndarray, Callable[[np.ndarray, np.ndarray, float], Any]]
EdgeValue = Union[float, Callable[[np.ndarray, float], Any]]

_EDGES = ("x_lo", "x_hi", "y_lo", "y_hi")


# ---------------------------------------------------------------------- grids
@dataclass(frozen=True)
class Grid2D:
    """Tensor-product grid ``x`` by ``y``."""

    x: Grid1D
    y: Grid1D

    @property
    def shape(self) -> tuple[int, int]:
        return (self.x.n, self.y.n)

    @property
    def n(self) -> int:
        return self.x.n * self.y.n

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)`` arrays of shape :attr:`shape` (``indexing='ij'``)."""
        return np.meshgrid(self.x.nodes, self.y.nodes, indexing="ij")

    def index(self, i: Any, j: Any) -> Any:
        """Flat C-order index of node ``(i, j)``."""
        return np.asarray(i) * self.y.n + np.asarray(j)

    def __repr__(self) -> str:
        x, y = self.x.nodes, self.y.nodes
        return (
            f"Grid2D({self.x.n}x{self.y.n}, x in [{x[0]:g}, {x[-1]:g}], y in [{y[0]:g}, {y[-1]:g}])"
        )


# ---------------------------------------------------------------------- edges
@dataclass(frozen=True)
class _Edge:
    value: EdgeValue = 0.0

    def __call__(self, other: np.ndarray, tau: float) -> np.ndarray:
        v = self.value(other, tau) if callable(self.value) else self.value
        return np.broadcast_to(np.asarray(v, dtype=float), other.shape).copy()


@dataclass(frozen=True)
class Dirichlet2D(_Edge):
    """``u = g(s, tau)`` on the edge; ``s`` is the coordinate along the edge."""


@dataclass(frozen=True)
class Neumann2D(_Edge):
    """Prescribed derivative across the edge, ``u_x = g`` (or ``u_y = g``)."""


@dataclass(frozen=True)
class Natural2D:
    """The equation holds on the edge with one-sided first derivatives.

    Intended for degenerate edges where the diffusion across the edge
    vanishes (``v = 0`` in Heston, ``S = 0`` in Black–Scholes); any remaining
    second derivative across the edge uses a first-order one-sided stencil.
    """


@dataclass(frozen=True)
class Linear2D:
    """Linear far-field condition: the equation holds with ``u_xx = 0`` across the edge.

    The standard "linearity" boundary condition for far edges where the
    solution is asymptotically linear in the normal direction (Windcliff,
    Forsyth & Vetzal, 2004); first derivatives are one-sided.
    """


EdgeCondition = Union[Dirichlet2D, Neumann2D, Natural2D, Linear2D]


# ---------------------------------------------------------------------- PDE
def _coef(c: Coefficient2D, X: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    v = c(X, Y, tau) if callable(c) else c
    return np.broadcast_to(np.asarray(v, dtype=float), X.shape).copy()


@dataclass
class LinearPDE2D:
    r"""Coefficients of :math:`u_\tau = a_{xx}u_{xx} + a_{xy}u_{xy} + a_{yy}u_{yy} + b_x u_x + b_y u_y + c u + f`.

    Each coefficient is a scalar, an array of the grid shape or a callable
    ``f(X, Y, tau)``. ``time_homogeneous`` defaults to "no coefficient is
    callable"; pass ``True`` for callables that ignore ``tau``.
    """

    axx: Coefficient2D = 0.0
    axy: Coefficient2D = 0.0
    ayy: Coefficient2D = 0.0
    bx: Coefficient2D = 0.0
    by: Coefficient2D = 0.0
    c: Coefficient2D = 0.0
    f: Coefficient2D = 0.0
    time_homogeneous: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.time_homogeneous is None:
            self.time_homogeneous = not any(
                callable(v)
                for v in (self.axx, self.axy, self.ayy, self.bx, self.by, self.c, self.f)
            )

    def coefficients(self, X: np.ndarray, Y: np.ndarray, tau: float) -> dict[str, np.ndarray]:
        out = {
            k: _coef(getattr(self, k), X, Y, tau)
            for k in ("axx", "axy", "ayy", "bx", "by", "c", "f")
        }
        if np.any(out["axx"] < 0.0) or np.any(out["ayy"] < 0.0):
            raise ValueError("diffusion coefficients must be non-negative")
        disc = 4.0 * out["axx"] * out["ayy"] - out["axy"] ** 2
        if np.any(disc < -1e-12 * (1.0 + np.abs(out["axy"]) ** 2)):
            raise ValueError("the operator is not parabolic (a_xy^2 > 4 a_xx a_yy)")
        return out


# ---------------------------------------------------------------------- operator
class DiscreteOperator2D:
    """Finite-difference discretisation of a :class:`LinearPDE2D` with ADI splitting.

    Parameters
    ----------
    pde : LinearPDE2D
    grid : Grid2D
    edges : dict
        Conditions for ``'x_lo'``, ``'x_hi'``, ``'y_lo'``, ``'y_hi'``.
        Dirichlet edges own their corners.
    upwind : {'auto', 'none'}
        ``'auto'`` switches the first derivative to the second-order upwind
        stencil where the cell Péclet number :math:`|b| h / (2a)` exceeds one.
    """

    def __init__(
        self,
        pde: LinearPDE2D,
        grid: Grid2D,
        edges: dict[str, EdgeCondition],
        upwind: str = "auto",
    ) -> None:
        missing = set(_EDGES) - set(edges)
        if missing:
            raise ValueError(f"missing edge conditions: {sorted(missing)}")
        for k, e in edges.items():
            if k not in _EDGES:
                raise ValueError(f"unknown edge {k!r}")
            if not isinstance(e, (Dirichlet2D, Neumann2D, Natural2D, Linear2D)):
                raise TypeError(f"unsupported edge condition {e!r}")
        if upwind not in ("auto", "none"):
            raise ValueError("upwind must be 'auto' or 'none'")
        if min(grid.shape) < 4:
            raise ValueError("need at least 4 nodes in each direction")
        self.pde = pde
        self.grid = grid
        self.edges = dict(edges)
        self.upwind = upwind
        self._X, self._Y = grid.mesh()
        mask = np.zeros(grid.shape, dtype=bool)
        if isinstance(edges["x_lo"], Dirichlet2D):
            mask[0, :] = True
        if isinstance(edges["x_hi"], Dirichlet2D):
            mask[-1, :] = True
        if isinstance(edges["y_lo"], Dirichlet2D):
            mask[:, 0] = True
        if isinstance(edges["y_hi"], Dirichlet2D):
            mask[:, -1] = True
        self.dirichlet_mask = mask
        self._dir_flat = np.flatnonzero(mask.ravel())
        self._cache: Optional[tuple[Any, ...]] = None

    @property
    def time_homogeneous(self) -> bool:
        return bool(self.pde.time_homogeneous)

    @property
    def dirichlet_nodes(self) -> np.ndarray:
        """Flat indices of nodes with Dirichlet values."""
        return self._dir_flat

    # ------------------------------------------------------------ assembly
    def _direction(
        self, coef: dict[str, np.ndarray], axis: int, tau: float
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Matrix, source vector and upwind mask of one direction."""
        g = self.grid
        nx, ny = g.shape
        if axis == 0:
            coord, other = g.x.nodes, g.y.nodes
            a, b = coef["axx"], coef["bx"]
            lo, hi = self.edges["x_lo"], self.edges["x_hi"]
            dmask = self.dirichlet_mask
        else:
            coord, other = g.y.nodes, g.x.nodes
            a, b = coef["ayy"].T, coef["by"].T
            lo, hi = self.edges["y_lo"], self.edges["y_hi"]
            dmask = self.dirichlet_mask.T
        c_half = 0.5 * (coef["c"] if axis == 0 else coef["c"].T)
        N, Q = coord.size, other.size  # direction length, number of lines

        def flat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            return p * ny + q if axis == 0 else q * ny + p

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        src = np.zeros((N, Q))
        up_mask = np.zeros((N, Q), dtype=bool)

        def add(p: np.ndarray, q: np.ndarray, pc: np.ndarray, v: np.ndarray) -> None:
            keep = ~dmask[p, q]
            rows.append(flat(p[keep], q[keep]))
            cols.append(flat(pc[keep], q[keep]))
            vals.append(np.asarray(v, dtype=float)[keep])

        # ---- interior nodes
        P, Qi = np.meshgrid(np.arange(1, N - 1), np.arange(Q), indexing="ij")
        l2, d2, u2 = central_second_weights(coord)
        l1, d1, u1 = central_first_weights(coord)
        A = a[1:-1, :]
        B = b[1:-1, :]
        add(P, Qi, P - 1, A * l2[:, None])
        add(P, Qi, P, A * d2[:, None] + c_half[1:-1, :])
        add(P, Qi, P + 1, A * u2[:, None])
        if self.upwind == "auto":
            h = 0.5 * (coord[2:] - coord[:-2])
            upw = np.abs(B) * h[:, None] > 2.0 * A
            # the upwind stencil needs two nodes on the upwind side
            p_idx = np.arange(1, N - 1)[:, None]
            upw &= ~((B > 0) & (p_idx >= N - 2)) & ~((B < 0) & (p_idx <= 1))
        else:
            upw = np.zeros_like(B, dtype=bool)
        up_mask[1:-1, :] = upw
        cen = ~upw
        add(P[cen], Qi[cen], P[cen] - 1, (B * l1[:, None])[cen])
        add(P[cen], Qi[cen], P[cen], (B * d1[:, None])[cen])
        add(P[cen], Qi[cen], P[cen] + 1, (B * u1[:, None])[cen])
        for sign in (1, -1):
            sel = upw & ((B > 0) if sign > 0 else (B < 0))
            if not np.any(sel):
                continue
            pp, qq = P[sel], Qi[sel]
            x0, x1, x2 = coord[pp], coord[pp + sign], coord[pp + 2 * sign]
            w0, w1, w2 = one_sided_first_weights(x0, x1, x2)
            bb = B[sel]
            add(pp, qq, pp, bb * w0)
            add(pp, qq, pp + sign, bb * w1)
            add(pp, qq, pp + 2 * sign, bb * w2)

        # ---- edges of this direction
        for side, cond in (("lo", lo), ("hi", hi)):
            if isinstance(cond, Dirichlet2D):
                continue
            p0, p1, p2 = (0, 1, 2) if side == "lo" else (N - 1, N - 2, N - 3)
            q = np.arange(Q)
            p = np.full(Q, p0)
            av, bv = a[p0, :], b[p0, :]
            hh = abs(coord[p1] - coord[p0])
            if isinstance(cond, Neumann2D):
                gv = cond(other, tau)
                sign = -1.0 if side == "lo" else 1.0
                add(p, q, p, -2.0 * av / hh**2 + c_half[p0, :])
                add(p, q, np.full(Q, p1), 2.0 * av / hh**2)
                src[p0, :] += sign * 2.0 * av * gv / hh + bv * gv
            else:  # Natural2D / Linear2D
                w0, w1, w2 = one_sided_first_weights(coord[p0], coord[p1], coord[p2])
                h1 = coord[p1] - coord[p0]
                h2 = coord[p2] - coord[p0]
                if isinstance(cond, Linear2D):
                    s0 = s1 = s2 = 0.0
                else:
                    # first-order one-sided second derivative (exact for quadratics)
                    s0 = 2.0 / (h1 * h2)
                    s1 = 2.0 / (h1 * (h1 - h2))
                    s2 = 2.0 / (h2 * (h2 - h1))
                add(p, q, p, bv * w0 + av * s0 + c_half[p0, :])
                add(p, q, np.full(Q, p1), bv * w1 + av * s1)
                add(p, q, np.full(Q, p2), bv * w2 + av * s2)
        M = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(g.n, g.n),
        )
        src_flat = (src if axis == 0 else src.T).ravel()
        src_flat[self._dir_flat] = 0.0
        mask_out = up_mask if axis == 0 else up_mask.T
        return M, src_flat, mask_out

    def _mixed(self, coef: dict[str, np.ndarray]) -> sp.csr_matrix:
        g = self.grid
        nx, ny = g.shape
        axy = coef["axy"]
        if not np.any(axy[1:-1, 1:-1]):
            return sp.csr_matrix((g.n, g.n))
        lx = central_first_weights(g.x.nodes)
        ly = central_first_weights(g.y.nodes)
        ii, jj = np.meshgrid(np.arange(1, nx - 1), np.arange(1, ny - 1), indexing="ij")
        keep = ~self.dirichlet_mask[ii, jj]
        ii, jj = ii[keep], jj[keep]
        a = axy[ii, jj]
        rows, cols, vals = [], [], []
        for di, wx in zip((-1, 0, 1), lx):
            for dj, wy in zip((-1, 0, 1), ly):
                rows.append(g.index(ii, jj))
                cols.append(g.index(ii + di, jj + dj))
                vals.append(a * wx[ii - 1] * wy[jj - 1])
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(g.n, g.n),
        )

    def split(self, tau: float = 0.0) -> tuple[list[sp.csr_matrix], list[np.ndarray]]:
        """``([A0, A1, A2], [b0, b1, b2])`` with ``A u + b`` the semi-discrete right-hand side.

        Rows of Dirichlet nodes are zero in every part.
        """
        if self.time_homogeneous and self._cache is not None and not self._has_tau_edges():
            mats, srcs = self._cache
            return list(mats), [s.copy() for s in srcs]
        coef = self.pde.coefficients(self._X, self._Y, tau)
        A1, b1, up1 = self._direction(coef, 0, tau)
        A2, b2, up2 = self._direction(coef, 1, tau)
        A0 = self._mixed(coef)
        b0 = coef["f"].ravel().copy()
        b0[self._dir_flat] = 0.0
        self.upwind_mask = (up1, up2)
        mats, srcs = [A0, A1, A2], [b0, b1, b2]
        if self.time_homogeneous:
            self._cache = (tuple(mats), tuple(srcs))
        return mats, srcs

    def _has_tau_edges(self) -> bool:
        return any(isinstance(e, Neumann2D) and callable(e.value) for e in self.edges.values())

    def matrices(self, tau: float = 0.0) -> list[sp.csr_matrix]:
        """``[A0, A1, A2]`` (cached for time-homogeneous coefficients)."""
        if self.time_homogeneous and self._cache is not None:
            return list(self._cache[0])
        return self.split(tau)[0]

    def sources(self, tau: float) -> list[np.ndarray]:
        """``[b0, b1, b2]`` at ``tau`` (Neumann data and the source term)."""
        if self.time_homogeneous and self._cache is not None and not self._has_tau_edges():
            return [s.copy() for s in self._cache[1]]
        if self.time_homogeneous and self._cache is not None:
            # only the Neumann data depends on tau: rebuild the direction sources
            coef = self.pde.coefficients(self._X, self._Y, tau)
            _, b1, _ = self._direction(coef, 0, tau)
            _, b2, _ = self._direction(coef, 1, tau)
            return [self._cache[1][0].copy(), b1, b2]
        return self.split(tau)[1]

    def dirichlet_values(self, tau: float) -> np.ndarray:
        """Flat vector holding the Dirichlet data at Dirichlet nodes (zero elsewhere)."""
        g = self.grid
        out = np.zeros(g.shape)
        e = self.edges
        # y edges first, x edges last so x-Dirichlet corners use the x data
        if isinstance(e["y_lo"], Dirichlet2D):
            out[:, 0] = e["y_lo"](g.x.nodes, tau)
        if isinstance(e["y_hi"], Dirichlet2D):
            out[:, -1] = e["y_hi"](g.x.nodes, tau)
        if isinstance(e["x_lo"], Dirichlet2D):
            out[0, :] = e["x_lo"](g.y.nodes, tau)
        if isinstance(e["x_hi"], Dirichlet2D):
            out[-1, :] = e["x_hi"](g.y.nodes, tau)
        flat = out.ravel()
        res = np.zeros(g.n)
        res[self._dir_flat] = flat[self._dir_flat]
        return res

    def apply(self, u: np.ndarray, tau: float) -> np.ndarray:
        """Semi-discrete right-hand side ``A u + b`` (zero at Dirichlet nodes)."""
        mats, srcs = self.split(tau)
        v = np.asarray(u, dtype=float).ravel()
        out = sum(M @ v for M in mats) + sum(srcs)
        return np.asarray(out).reshape(self.grid.shape)

    def __repr__(self) -> str:
        kinds = {k: type(v).__name__ for k, v in self.edges.items()}
        return f"DiscreteOperator2D({self.grid!r}, edges={kinds}, upwind={self.upwind!r})"


# ---------------------------------------------------------------------- solution
@dataclass
class PDESolution2D:
    """Solution of a two-dimensional pricing PDE at ``tau = T``.

    Values are interpolated with a bicubic spline through the nodal values.
    """

    grid: Grid2D
    values: np.ndarray
    previous_values: Optional[np.ndarray] = None
    dt_last: Optional[float] = None
    info: dict[str, Any] = field(default_factory=dict)
    label: str = "2D PDE solution"
    variables: tuple[str, str] = ("x", "y")

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).reshape(self.grid.shape)
        self._spline: Optional[RectBivariateSpline] = None

    def _spl(self) -> RectBivariateSpline:
        if self._spline is None:
            self._spline = RectBivariateSpline(
                self.grid.x.nodes, self.grid.y.nodes, self.values, kx=3, ky=3, s=0
            )
        return self._spline

    def _check(self, x: np.ndarray, y: np.ndarray) -> None:
        gx, gy = self.grid.x.nodes, self.grid.y.nodes
        if np.any((x < gx[0]) | (x > gx[-1]) | (y < gy[0]) | (y > gy[-1])):
            raise ValueError("point outside the grid")

    def derivative(self, x: Any, y: Any, dx: int = 0, dy: int = 0) -> Any:
        """Spline derivative ``d^{dx+dy} u / dx^{dx} dy^{dy}`` at ``(x, y)``."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self._check(xa, ya)
        out = self._spl().ev(xa.ravel(), ya.ravel(), dx=dx, dy=dy).reshape(xa.shape)
        return float(out) if out.ndim == 0 else out

    def value(self, x: Any, y: Any) -> Any:
        return self.derivative(x, y)

    def theta(self, x: Any, y: Any) -> Any:
        """Calendar-time theta ``-du/dtau`` from the last time step."""
        if self.previous_values is None or not self.dt_last:
            raise ValueError("theta needs the previous time level")
        prev = RectBivariateSpline(
            self.grid.x.nodes, self.grid.y.nodes, self.previous_values, kx=3, ky=3, s=0
        )
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        du = self._spl().ev(xa.ravel(), ya.ravel()) - prev.ev(xa.ravel(), ya.ravel())
        out = (-du / self.dt_last).reshape(xa.shape)
        return float(out) if out.ndim == 0 else out

    def __story__(self) -> str:
        """Plain-language account of how the solution was obtained."""
        g = self.grid
        vx, vy = self.variables
        lines = [
            self.label,
            f"  grid: {g.x.n} x {g.y.n} nodes, {vx} in [{g.x.lo:.6g}, {g.x.hi:.6g}], "
            f"{vy} in [{g.y.lo:.6g}, {g.y.hi:.6g}]",
        ]
        info = self.info
        if "scheme" in info:
            lines.append(
                f"  time stepping: {info['scheme']} ADI (theta={info.get('theta', float('nan')):.4g}), "
                f"{info.get('n_steps', '?')} steps, {info.get('damping_steps', 0)} damping steps"
            )
        if info.get("american"):
            lines.append("  early exercise: Ikonen-Toivanen splitting")
        if "edges" in info:
            lines.append(f"  edges: {info['edges']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PDESolution2D({self.label}, grid={self.grid!r})"


@dataclass
class ADIResult:
    """Raw output of :meth:`ADISolver.solve`."""

    values: np.ndarray
    previous_values: np.ndarray
    times: np.ndarray
    info: dict[str, Any]
    multiplier: Optional[np.ndarray] = None
    history: Optional[list[np.ndarray]] = None


# ---------------------------------------------------------------------- ADI
_SCHEMES = {
    "douglas": 0.5,
    "craig_sneyd": 0.5,
    "modified_craig_sneyd": 1.0 / 3.0,
    "hundsdorfer_verwer": 0.5 + math.sqrt(3.0) / 6.0,
}
_ALIASES = {
    "do": "douglas",
    "cs": "craig_sneyd",
    "mcs": "modified_craig_sneyd",
    "hv": "hundsdorfer_verwer",
}


class ADISolver:
    r"""Alternating-direction implicit time stepping for :class:`DiscreteOperator2D`.

    With :math:`F_j(t, u) = A_j u + b_j(t)` and :math:`F = F_0 + F_1 + F_2`,
    every scheme starts with the Douglas stages

    .. math::
        Y_0 = U + \Delta\tau F(t_n, U), \qquad
        Y_j = Y_{j-1} + \theta\Delta\tau\,(F_j(t_{n+1}, Y_j) - F_j(t_n, U)),

    followed by a correction (Craig–Sneyd, modified Craig–Sneyd,
    Hundsdorfer–Verwer). All four are unconditionally stable for the
    convection–diffusion equations with mixed derivatives considered here
    for the default :math:`\theta`, and second order except Douglas with a
    mixed term.

    Parameters
    ----------
    operator : DiscreteOperator2D
    scheme : {'douglas', 'craig_sneyd', 'modified_craig_sneyd', 'hundsdorfer_verwer'}
        Short names ``'do'``, ``'cs'``, ``'mcs'``, ``'hv'`` are accepted.
    theta : float, optional
        Defaults: 1/2 (Do, CS), 1/3 (MCS), 1/2 + sqrt(3)/6 (HV).
    damping_steps : int
        Number of initial steps replaced by two implicit (Douglas,
        ``theta = 1``) half-steps, to damp the non-smooth payoff.
    """

    def __init__(
        self,
        operator: DiscreteOperator2D,
        scheme: str = "hundsdorfer_verwer",
        theta: Optional[float] = None,
        damping_steps: int = 1,
    ) -> None:
        name = _ALIASES.get(scheme.lower(), scheme.lower())
        if name not in _SCHEMES:
            raise ValueError(f"unknown ADI scheme {scheme!r}; choose from {sorted(_SCHEMES)}")
        if damping_steps < 0:
            raise ValueError("damping_steps must be non-negative")
        self.operator = operator
        self.scheme = name
        self.theta = float(_SCHEMES[name] if theta is None else theta)
        self.damping_steps = int(damping_steps)
        self._lu: dict[tuple[int, float, float], Any] = {}
        self._n_factorisations = 0

    # ------------------------------------------------------------ helpers
    def _solver(self, j: int, M: sp.csr_matrix, coeff: float, tau: float) -> Any:
        op = self.operator
        key = (j, round(coeff, 15), tau if not op.time_homogeneous else 0.0)
        lu = self._lu.get(key)
        if lu is None:
            n = op.grid.n
            S = (sp.identity(n, format="csr") - coeff * M).tolil()
            d = op.dirichlet_nodes
            if d.size:
                S[d, :] = 0.0
                S[d, d] = 1.0
            lu = splu(S.tocsc())
            self._n_factorisations += 1
            if op.time_homogeneous:
                self._lu[key] = lu
        return lu

    def _implicit(
        self, j: int, M: sp.csr_matrix, coeff: float, rhs: np.ndarray, g_new: np.ndarray, t1: float
    ) -> np.ndarray:
        d = self.operator.dirichlet_nodes
        r = rhs.copy()
        r[d] = g_new[d]
        return np.asarray(self._solver(j, M, coeff, t1).solve(r))

    def _step(
        self,
        U: np.ndarray,
        t0: float,
        t1: float,
        scheme: str,
        theta: float,
        lam: Optional[np.ndarray],
    ) -> np.ndarray:
        op = self.operator
        dt = t1 - t0
        A0, A1, A2 = op.matrices(t1)
        mats = (A0, A1, A2)
        if not op.time_homogeneous:
            mats_old = tuple(op.matrices(t0))
        else:
            mats_old = mats
        b_old = op.sources(t0)
        b_new = op.sources(t1)
        g_new = op.dirichlet_values(t1)
        F_old = [mats_old[j] @ U + b_old[j] for j in range(3)]

        def F_new(j: int, v: np.ndarray) -> np.ndarray:
            return mats[j] @ v + b_new[j]

        def stages(Y0: np.ndarray, ref: list[np.ndarray]) -> np.ndarray:
            # (I - theta dt A_j) Y_j = Y_{j-1} + theta dt (b_j(t_{n+1}) - ref_j)
            Y = Y0
            for j in (1, 2):
                rhs = Y + theta * dt * (b_new[j] - ref[j])
                Y = self._implicit(j, mats[j], theta * dt, rhs, g_new, t1)
            return Y

        Y0 = U + dt * sum(F_old)
        if lam is not None:
            Y0 = Y0 + dt * lam
        Y2 = stages(Y0, F_old)
        if scheme == "douglas":
            return Y2
        Fo = sum(F_old)
        Fn = [F_new(j, Y2) for j in range(3)]
        if scheme == "craig_sneyd":
            Yt0 = Y0 + 0.5 * dt * (Fn[0] - F_old[0])
            return stages(Yt0, F_old)
        if scheme == "modified_craig_sneyd":
            Yh0 = Y0 + theta * dt * (Fn[0] - F_old[0])
            Yt0 = Yh0 + (0.5 - theta) * dt * (sum(Fn) - Fo)
            return stages(Yt0, F_old)
        # Hundsdorfer-Verwer: the second sweep is relative to Y2 at t_{n+1}
        Yt0 = Y0 + 0.5 * dt * (sum(Fn) - Fo)
        return stages(Yt0, Fn)

    # ------------------------------------------------------------ public
    def solve(
        self,
        initial: np.ndarray,
        T: float,
        n_time: int = 100,
        obstacle: Optional[
            Union[np.ndarray, Callable[[np.ndarray, np.ndarray, float], Any]]
        ] = None,
        store_history: bool = False,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> ADIResult:
        """March from ``tau = 0`` to ``T`` in ``n_time`` uniform steps.

        With an ``obstacle`` the Ikonen–Toivanen splitting enforces
        ``u >= obstacle``: each step solves with the multiplier added
        explicitly, then ``u = max(u~ - dt*lam, g)`` and
        ``lam = max(0, lam + (g - u~)/dt)``.
        """
        if T <= 0.0 or n_time < 1:
            raise ValueError("need T > 0 and n_time >= 1")
        op = self.operator
        g = op.grid
        U = np.asarray(initial, dtype=float).reshape(g.n).copy()
        d = op.dirichlet_nodes
        U[d] = op.dirichlet_values(0.0)[d]
        X, Y = g.mesh()
        free = np.ones(g.n, dtype=bool)
        free[d] = False

        def obst(tau: float) -> Optional[np.ndarray]:
            if obstacle is None:
                return None
            v = obstacle(X, Y, tau) if callable(obstacle) else obstacle
            return np.broadcast_to(np.asarray(v, dtype=float), g.shape).ravel()

        lam = np.zeros(g.n) if obstacle is not None else None
        times = np.linspace(0.0, T, n_time + 1)
        prev = U.copy()
        history = [U.reshape(g.shape).copy()] if store_history else None
        if callback is not None:
            callback(0, 0.0, U.reshape(g.shape))

        def advance(U: np.ndarray, t0: float, t1: float, scheme: str, theta: float) -> np.ndarray:
            nonlocal lam
            Ut = self._step(U, t0, t1, scheme, theta, lam)
            if lam is None:
                return Ut
            gv = obst(t1)
            assert gv is not None
            dt = t1 - t0
            out = Ut.copy()
            out[free] = np.maximum(Ut[free] - dt * lam[free], gv[free])
            new = np.zeros_like(lam)
            new[free] = np.maximum(0.0, lam[free] + (gv[free] - Ut[free]) / dt)
            lam = new
            return out

        for k in range(1, times.size):
            t0, t1 = float(times[k - 1]), float(times[k])
            prev = U
            if k <= self.damping_steps:
                mid = 0.5 * (t0 + t1)
                U = advance(U, t0, mid, "douglas", 1.0)
                U = advance(U, mid, t1, "douglas", 1.0)
            else:
                U = advance(U, t0, t1, self.scheme, self.theta)
            if history is not None:
                history.append(U.reshape(g.shape).copy())
            if callback is not None:
                callback(k, t1, U.reshape(g.shape))
        info = {
            "scheme": self.scheme,
            "theta": self.theta,
            "n_steps": n_time,
            "damping_steps": self.damping_steps,
            "n_factorisations": self._n_factorisations,
            "american": obstacle is not None,
            "edges": {k: type(v).__name__ for k, v in self.operator.edges.items()},
        }
        return ADIResult(
            values=U.reshape(g.shape),
            previous_values=prev.reshape(g.shape),
            times=times,
            info=info,
            multiplier=None if lam is None else lam.reshape(g.shape),
            history=history,
        )
