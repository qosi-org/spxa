r"""Finite-difference operators on non-uniform one-dimensional grids.

Stencils
--------
At an interior node :math:`x_i` with :math:`h_- = x_i - x_{i-1}` and
:math:`h_+ = x_{i+1} - x_i`, the three-point central approximations are

.. math::
    u'(x_i) &\approx -\frac{h_+}{h_-(h_- + h_+)} u_{i-1}
              + \frac{h_+ - h_-}{h_- h_+} u_i
              + \frac{h_-}{h_+(h_- + h_+)} u_{i+1}, \\
    u''(x_i) &\approx \frac{2}{h_-(h_- + h_+)} u_{i-1}
              - \frac{2}{h_- h_+} u_i
              + \frac{2}{h_+(h_- + h_+)} u_{i+1}.

The first-derivative stencil is exact for quadratics, so it is second-order
on any grid. The second-derivative stencil is exact for quadratics and has a
leading error of :math:`\tfrac13 (h_+ - h_-) u'''`, which is
:math:`O(h^2)` on smoothly stretched grids such as the sinh grids of
:mod:`spxa.pde.grid`.

Operator
--------
:class:`LinearPDE1D` stores the coefficients of the backward equation

.. math::
    u_\tau = a(x, \tau) u_{xx} + b(x, \tau) u_x + c(x, \tau) u + f(x, \tau),

and :class:`DiscreteOperator1D` assembles its affine discretisation
:math:`L_h u + s` together with the boundary closures of
:mod:`spxa.pde.boundary`.

Convection may be discretised by central differences (second order) or by
first-order upwinding. ``'auto'`` upwinds only at nodes where the central
scheme would produce a negative off-diagonal coefficient, which is where it
would lose the discrete maximum principle. Such nodes are those with a cell
Péclet number above one (:math:`|b| h > 2a`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from spxa.pde._utils import Coefficient, evaluate_coefficient
from spxa.pde.boundary import BoundaryCondition, Dirichlet, Linear, Natural, Neumann
from spxa.pde.grid import Grid1D


def central_first_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central first-derivative weights ``(lower, diag, upper)`` at interior nodes.

    Returns arrays of length ``len(x) - 2``; entry ``k`` belongs to node ``k + 1``.
    """
    x = np.asarray(x, dtype=float)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    lower = -hp / (hm * (hm + hp))
    diag = (hp - hm) / (hm * hp)
    upper = hm / (hp * (hm + hp))
    return lower, diag, upper


def central_second_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central second-derivative weights ``(lower, diag, upper)`` at interior nodes."""
    x = np.asarray(x, dtype=float)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    lower = 2.0 / (hm * (hm + hp))
    diag = -2.0 / (hm * hp)
    upper = 2.0 / (hp * (hm + hp))
    return lower, diag, upper


def one_sided_first_weights(x0: float, x1: float, x2: float) -> tuple[float, float, float]:
    """Second-order one-sided first-derivative weights at ``x0`` using ``x1``, ``x2``.

    Works for either side (``x1``, ``x2`` may lie to the left of ``x0``); the
    stencil is exact for quadratics.
    """
    d1 = x1 - x0
    d2 = x2 - x0
    w1 = d2 / (d1 * (d2 - d1))
    w2 = -d1 / (d2 * (d2 - d1))
    return -(w1 + w2), w1, w2


def derivative_matrices(x: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Sparse first- and second-derivative matrices with one-sided boundary rows.

    Interior rows use the central stencils. Boundary rows use second-order
    one-sided stencils for the first derivative; the second derivative is
    extrapolated from the adjacent interior row (first-order). These
    matrices are meant for evaluating derivatives of computed solutions
    (Greeks), not for imposing boundary conditions.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 nodes")
    l1, d1, u1 = central_first_weights(x)
    l2, d2, u2 = central_second_weights(x)
    idx = np.arange(1, n - 1)
    rows = np.concatenate([idx, idx, idx])
    cols = np.concatenate([idx - 1, idx, idx + 1])
    D1 = sp.lil_matrix(sp.csr_matrix((np.concatenate([l1, d1, u1]), (rows, cols)), shape=(n, n)))
    D2 = sp.lil_matrix(sp.csr_matrix((np.concatenate([l2, d2, u2]), (rows, cols)), shape=(n, n)))
    w = one_sided_first_weights(x[0], x[1], x[2])
    D1[0, 0:3] = np.array(w)
    w = one_sided_first_weights(x[-1], x[-2], x[-3])
    D1[n - 1, n - 1] = w[0]
    D1[n - 1, n - 2] = w[1]
    D1[n - 1, n - 3] = w[2]
    D2[0, 0:3] = np.array([l2[0], d2[0], u2[0]])
    D2[n - 1, n - 3 :] = np.array([l2[-1], d2[-1], u2[-1]])
    return D1.tocsr(), D2.tocsr()


@dataclass
class LinearPDE1D:
    r"""Coefficients of :math:`u_\tau = a u_{xx} + b u_x + c u + f`.

    Each coefficient may be a scalar, a nodal array or a callable
    ``f(x, tau)``.

    Parameters
    ----------
    diffusion : Coefficient
        :math:`a(x, \tau) \ge 0` (for a diffusion :math:`dX = \mu dt + \sigma dW`,
        :math:`a = \tfrac12 \sigma^2`).
    convection : Coefficient
        :math:`b(x, \tau)` (the drift :math:`\mu`).
    reaction : Coefficient
        :math:`c(x, \tau)` (minus the discount rate).
    source : Coefficient
        :math:`f(x, \tau)` (running payoff, e.g. a continuous coupon).
    time_homogeneous : bool, optional
        Whether the coefficients are independent of ``tau``. Defaults to
        ``True`` when no coefficient is callable. Solvers cache matrix
        factorisations for time-homogeneous problems.
    """

    diffusion: Coefficient = 0.0
    convection: Coefficient = 0.0
    reaction: Coefficient = 0.0
    source: Coefficient = 0.0
    time_homogeneous: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.time_homogeneous is None:
            self.time_homogeneous = not any(
                callable(c) for c in (self.diffusion, self.convection, self.reaction, self.source)
            )

    def coefficients(
        self, x: np.ndarray, tau: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``(a, b, c, f)`` on the nodes ``x`` at time ``tau``."""
        a = evaluate_coefficient(self.diffusion, x, tau)
        if np.any(a < -1e-14):
            raise ValueError("diffusion coefficient must be non-negative")
        a = np.maximum(a, 0.0)
        b = evaluate_coefficient(self.convection, x, tau)
        c = evaluate_coefficient(self.reaction, x, tau)
        f = evaluate_coefficient(self.source, x, tau)
        return a, b, c, f


_SCHEMES = ("central", "upwind", "auto")


class DiscreteOperator1D:
    r"""Affine finite-difference discretisation :math:`L_h u + s` of a :class:`LinearPDE1D`.

    Parameters
    ----------
    pde : LinearPDE1D
        Coefficients.
    grid : Grid1D
        Spatial grid.
    left, right : BoundaryCondition
        Boundary closures at ``grid.lo`` and ``grid.hi``.
    convection_scheme : {'central', 'upwind', 'auto'}
        Discretisation of the first-order term (see module docstring).
    """

    def __init__(
        self,
        pde: LinearPDE1D,
        grid: Grid1D,
        left: BoundaryCondition,
        right: BoundaryCondition,
        convection_scheme: str = "auto",
    ) -> None:
        if convection_scheme not in _SCHEMES:
            raise ValueError(f"convection_scheme must be one of {_SCHEMES}")
        for bc in (left, right):
            if not isinstance(bc, BoundaryCondition):
                raise TypeError("boundary conditions must be BoundaryCondition instances")
        self.pde = pde
        self.grid = grid
        self.left = left
        self.right = right
        self.convection_scheme = convection_scheme
        x = grid.nodes
        self._x = x
        self._c1 = central_first_weights(x)
        self._c2 = central_second_weights(x)
        self._hm = x[1:-1] - x[:-2]
        self._hp = x[2:] - x[1:-1]

    @property
    def n(self) -> int:
        """Number of grid nodes."""
        return self.grid.n

    @property
    def time_homogeneous(self) -> bool:
        """True when neither coefficients nor boundary data depend on ``tau``."""
        bcs_static = all(
            not callable(getattr(bc, "value", None))
            and not callable(getattr(bc, "derivative", None))
            for bc in (self.left, self.right)
        )
        return bool(self.pde.time_homogeneous) and bcs_static

    @property
    def matrix_time_homogeneous(self) -> bool:
        """True when the matrix :math:`L_h` (not necessarily ``s``) is ``tau``-independent."""
        return bool(self.pde.time_homogeneous)

    @property
    def constraint_rows(self) -> np.ndarray:
        """Indices of rows that are algebraic boundary constraints."""
        rows = []
        if self.left.is_constraint:
            rows.append(0)
        if self.right.is_constraint:
            rows.append(self.n - 1)
        return np.array(rows, dtype=int)

    # ------------------------------------------------------------------ assembly
    def _interior_weights(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ai, bi, ci = a[1:-1], b[1:-1], c[1:-1]
        l2, d2, u2 = self._c2
        l1, d1, u1 = self._c1
        lower = ai * l2 + bi * l1
        diag = ai * d2 + bi * d1 + ci
        upper = ai * u2 + bi * u1
        if self.convection_scheme == "central":
            return lower, diag, upper
        # first-order upwind convection
        bp = np.maximum(bi, 0.0)
        bm = np.minimum(bi, 0.0)
        lower_up = ai * l2 - bm / self._hm
        diag_up = ai * d2 - bp / self._hp + bm / self._hm + ci
        upper_up = ai * u2 + bp / self._hp
        if self.convection_scheme == "upwind":
            return lower_up, diag_up, upper_up
        bad = (lower < 0.0) | (upper < 0.0)
        return (
            np.where(bad, lower_up, lower),
            np.where(bad, diag_up, diag),
            np.where(bad, upper_up, upper),
        )

    def upwinded_nodes(self, tau: float = 0.0) -> np.ndarray:
        """Boolean mask of interior nodes where ``'auto'`` switches to upwinding."""
        a, b, c, _ = self.pde.coefficients(self._x, tau)
        l2, _, u2 = self._c2
        l1, _, u1 = self._c1
        lower = a[1:-1] * l2 + b[1:-1] * l1
        upper = a[1:-1] * u2 + b[1:-1] * u1
        mask = np.zeros(self.n, dtype=bool)
        if self.convection_scheme == "upwind":
            mask[1:-1] = True
        elif self.convection_scheme == "auto":
            mask[1:-1] = (lower < 0.0) | (upper < 0.0)
        return mask

    def assemble(self, tau: float) -> tuple[sp.csr_matrix, np.ndarray]:
        r"""Return ``(L, s)`` with ``L u + s`` approximating the PDE right-hand side.

        Rows belonging to constraint boundaries are identically zero; use
        :meth:`constraints` for those rows.
        """
        x = self._x
        n = self.n
        a, b, c, f = self.pde.coefficients(x, tau)
        lower, diag, upper = self._interior_weights(a, b, c)
        idx = np.arange(1, n - 1)
        rows = [idx, idx, idx]
        cols = [idx - 1, idx, idx + 1]
        vals = [lower, diag, upper]
        s = f.copy()
        # ---- boundary rows
        for side in ("left", "right"):
            bc = self.left if side == "left" else self.right
            i, j1, j2 = (0, 1, 2) if side == "left" else (n - 1, n - 2, n - 3)
            if bc.is_constraint:
                s[i] = 0.0
                continue
            h = abs(x[j1] - x[i])
            if isinstance(bc, Neumann):
                g = bc(tau)
                # ghost node mirrored across the boundary, u_ghost = u_j1 -/+ 2 h g
                sign = -1.0 if side == "left" else 1.0
                rows.append(np.array([i, i]))
                cols.append(np.array([i, j1]))
                vals.append(np.array([-2.0 * a[i] / h**2 + c[i], 2.0 * a[i] / h**2]))
                s[i] += sign * 2.0 * a[i] * g / h + b[i] * g
            elif isinstance(bc, Natural):
                w0, w1, w2 = one_sided_first_weights(x[i], x[j1], x[j2])
                rows.append(np.array([i, i, i]))
                cols.append(np.array([i, j1, j2]))
                vals.append(np.array([b[i] * w0 + c[i], b[i] * w1, b[i] * w2]))
            else:  # pragma: no cover - guarded in __init__ via type hierarchy
                raise TypeError(f"unsupported boundary condition {bc!r}")
        L = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        return L, s

    def constraints(self, tau: float) -> tuple[sp.csr_matrix, np.ndarray]:
        """Constraint rows ``C u = g`` for algebraic boundaries.

        Returns an ``n x n`` sparse matrix that is non-zero only on the
        constraint rows and the right-hand side vector ``g`` (zero elsewhere).
        """
        x = self._x
        n = self.n
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        g = np.zeros(n)
        for side in ("left", "right"):
            bc = self.left if side == "left" else self.right
            i, j1, j2 = (0, 1, 2) if side == "left" else (n - 1, n - 2, n - 3)
            if isinstance(bc, Dirichlet):
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                g[i] = bc(tau)
            elif isinstance(bc, Linear):
                w1, w2 = Linear.weights(x[i], x[j1], x[j2])
                rows += [i, i, i]
                cols += [i, j1, j2]
                vals += [1.0, -w1, -w2]
        C = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return C, g

    def apply(self, u: np.ndarray, tau: float) -> np.ndarray:
        """Evaluate ``L u + s`` (constraint rows return zero)."""
        L, s = self.assemble(tau)
        return L @ np.asarray(u, dtype=float) + s

    def enforce_constraints(self, u: np.ndarray, tau: float) -> np.ndarray:
        """Return a copy of ``u`` whose constraint rows satisfy ``C u = g``."""
        out = np.array(u, dtype=float, copy=True)
        n = self.n
        x = self._x
        for side in ("left", "right"):
            bc = self.left if side == "left" else self.right
            i, j1, j2 = (0, 1, 2) if side == "left" else (n - 1, n - 2, n - 3)
            if isinstance(bc, Dirichlet):
                out[i] = bc(tau)
            elif isinstance(bc, Linear):
                w1, w2 = Linear.weights(x[i], x[j1], x[j2])
                out[i] = w1 * out[j1] + w2 * out[j2]
        return out

    def __repr__(self) -> str:
        return (
            f"DiscreteOperator1D(n={self.n}, left={self.left.describe()}, "
            f"right={self.right.describe()}, convection={self.convection_scheme!r})"
        )
