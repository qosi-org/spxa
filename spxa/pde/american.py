r"""Solvers for the linear complementarity problems of American-style options.

With early exercise, each implicit time step becomes the linear
complementarity problem (LCP)

.. math::
    M u \ge b, \qquad u \ge g, \qquad (M u - b)^\top (u - g) = 0,

where :math:`M = I - \theta\Delta\tau L_h` and :math:`g` is the exercise
value. All solvers implement the
:class:`~spxa.pde.theta.LCPStepper` protocol and can be passed to
:class:`~spxa.pde.theta.ThetaSolver1D` (and the pricers built on it).

* :class:`PenaltyLCP`: the penalty iteration of Forsyth & Vetzal (2002).
  It works for any matrix structure, converges finitely in practice and is
  the default.
* :class:`PSORLCP`: projected successive over-relaxation (Cryer, 1971), the
  classical but slow iterative method.
* :class:`BrennanSchwartzLCP`: the direct method of Brennan & Schwartz
  (1977). It is exact in one sweep for tridiagonal systems whose exercise
  region is an interval at one end of the grid (puts, or calls with
  dividends).
* :class:`IkonenToivanenLCP`: the operator splitting of Ikonen & Toivanen
  (2004). One linear solve per step plus an explicit update of a Lagrange
  multiplier; it extends verbatim to ADI schemes in two dimensions.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def complementarity_residual(
    M: sp.spmatrix, u: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray, free: np.ndarray
) -> float:
    """Largest violation of the LCP conditions on the ``free`` rows (scaled).

    Returns ``max(max(-w), max(g - u), max |w * (u - g)|)`` with
    ``w = M u - rhs``, normalised by ``max(1, max|u|)``.
    """
    w = M @ u - rhs
    gap = u - obstacle
    scale = max(1.0, float(np.max(np.abs(u))))
    viol = max(
        float(np.max(-w[free], initial=0.0)),
        float(np.max(-gap[free], initial=0.0)),
        float(np.max(np.abs(w[free] * gap[free]), initial=0.0)) / scale,
    )
    return viol / scale


class PenaltyLCP:
    r"""Penalty iteration of Forsyth & Vetzal (2002).

    Iterates

    .. math::
        (M + P^k) u^{k+1} = b + P^k g, \qquad
        P^k_{ii} = \begin{cases} 1/\varepsilon & u^k_i < g_i \\ 0 & \text{else}\end{cases}

    until the active set stops changing or the relative update falls below
    ``tol``. The constraint is satisfied to :math:`O(\varepsilon)`.

    Parameters
    ----------
    penalty : float
        Penalty factor :math:`1/\varepsilon`.
    tol : float
        Relative convergence tolerance.
    max_iter : int
        Iteration cap per time step.
    """

    def __init__(self, penalty: float = 1e8, tol: float = 1e-10, max_iter: int = 100) -> None:
        if penalty <= 0 or tol <= 0 or max_iter < 1:
            raise ValueError("penalty, tol and max_iter must be positive")
        self.penalty = float(penalty)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.iterations: list[int] = []

    def reset(self, n: int) -> None:
        self.iterations = []

    def step(
        self,
        M: sp.csr_matrix,
        solve: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        obstacle: np.ndarray,
        guess: np.ndarray,
        dt: float,
        free: np.ndarray,
    ) -> np.ndarray:
        u = np.maximum(guess, np.where(free, obstacle, -np.inf))
        active = free & (u <= obstacle)
        n = u.size
        for k in range(1, self.max_iter + 1):
            if not np.any(active):
                u_new = solve(rhs)
            else:
                p = np.where(active, self.penalty, 0.0)
                A = (M + sp.diags(p, 0, shape=(n, n))).tocsc()
                u_new = splu(A).solve(rhs + p * obstacle)
            new_active = free & (u_new < obstacle)
            change = np.max(np.abs(u_new - u)) / max(1.0, float(np.max(np.abs(u_new))))
            u = u_new
            if np.array_equal(new_active, active) or change < self.tol:
                self.iterations.append(k)
                return u
            active = new_active
        self.iterations.append(self.max_iter)
        return u

    def describe(self) -> str:
        return f"penalty iteration (1/eps={self.penalty:.0e})"


class PSORLCP:
    r"""Projected successive over-relaxation (Cryer, 1971).

    Gauss–Seidel sweeps with relaxation :math:`\omega \in (0, 2)`, each
    update projected onto :math:`u \ge g`. Converges for symmetric positive
    definite and for strictly diagonally dominant M-matrices, but needs
    :math:`O(n)` sweeps per step on fine grids.
    """

    def __init__(self, omega: float = 1.5, tol: float = 1e-10, max_iter: int = 20000) -> None:
        if not 0.0 < omega < 2.0:
            raise ValueError("omega must lie in (0, 2)")
        self.omega = float(omega)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.iterations: list[int] = []

    def reset(self, n: int) -> None:
        self.iterations = []

    def step(
        self,
        M: sp.csr_matrix,
        solve: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        obstacle: np.ndarray,
        guess: np.ndarray,
        dt: float,
        free: np.ndarray,
    ) -> np.ndarray:
        M = sp.csr_matrix(M)
        indptr, indices, data = M.indptr, M.indices, M.data
        n = rhs.size
        diag = M.diagonal()
        if np.any(diag == 0.0):
            raise ValueError("PSOR needs a non-zero diagonal")
        u = np.where(free, np.maximum(guess, obstacle), guess).astype(float)
        w = self.omega
        for k in range(1, self.max_iter + 1):
            delta = 0.0
            for i in range(n):
                lo, hi = indptr[i], indptr[i + 1]
                s = float(np.dot(data[lo:hi], u[indices[lo:hi]]))
                gs = u[i] + (rhs[i] - s) / diag[i]
                new = u[i] + w * (gs - u[i]) if free[i] else gs
                if free[i] and new < obstacle[i]:
                    new = obstacle[i]
                delta = max(delta, abs(new - u[i]))
                u[i] = new
            if delta <= self.tol * max(1.0, float(np.max(np.abs(u)))):
                self.iterations.append(k)
                return u
        self.iterations.append(self.max_iter)
        return u

    def describe(self) -> str:
        return f"PSOR (omega={self.omega})"


class BrennanSchwartzLCP:
    r"""Direct LCP solver of Brennan & Schwartz (1977) for tridiagonal systems.

    For an exercise region :math:`\{i < i^*\}` (``side='left'``, e.g. a put)
    the tridiagonal system is reduced to lower-bidiagonal form by eliminating
    from the last row upwards, and the solution is then recovered by forward
    substitution with projection,
    :math:`u_i = \max\big((d'_i - a_i u_{i-1})/b'_i,\; g_i\big)`.
    Jaillet, Lamberton & Lapeyre (1990) show that this yields the exact LCP
    solution whenever the exercise region is a single interval at that end
    of the grid. ``side='right'`` handles exercise regions at the upper end
    (calls with dividends) by reversing the ordering.

    Boundary rows whose stencil reaches two nodes inward (zero-curvature and
    natural boundaries) are eliminated algebraically: the boundary unknown
    is substituted into the adjacent row and recovered after the sweep. This
    treats the boundary row as an equation and is exact when that node lies
    in the continuation region (or is a constraint row).
    """

    def __init__(self, side: str = "left") -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side

    def reset(self, n: int) -> None:
        pass

    @staticmethod
    def _sweep(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, g: np.ndarray, fr: np.ndarray
    ) -> np.ndarray:
        n = d.size
        bp = b.copy()
        dp = d.copy()
        for i in range(n - 2, -1, -1):
            m = c[i] / bp[i + 1]
            bp[i] -= m * a[i + 1]
            dp[i] -= m * dp[i + 1]
        u = np.empty(n)
        u[0] = dp[0] / bp[0]
        if fr[0]:
            u[0] = max(u[0], g[0])
        for i in range(1, n):
            u[i] = (dp[i] - a[i] * u[i - 1]) / bp[i]
            if fr[i]:
                u[i] = max(u[i], g[i])
        return u

    def step(
        self,
        M: sp.csr_matrix,
        solve: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        obstacle: np.ndarray,
        guess: np.ndarray,
        dt: float,
        free: np.ndarray,
    ) -> np.ndarray:
        A = sp.csr_matrix(M, copy=True)
        A.eliminate_zeros()
        A = A.tolil()
        d = np.asarray(rhs, dtype=float).copy()
        n = d.size
        i0, i1 = 0, n - 1
        first = last = None
        # eliminate boundary unknowns whose rows reach two nodes inward
        if n > 3 and A[0, 2] != 0.0:
            c0, c1, c2 = A[0, 0], A[0, 1], A[0, 2]
            a1 = A[1, 0]
            A[1, 1] = A[1, 1] - a1 * c1 / c0
            A[1, 2] = A[1, 2] - a1 * c2 / c0
            d[1] -= a1 * d[0] / c0
            A[1, 0] = 0.0
            first = (c0, c1, c2, d[0])
            i0 = 1
        if n > 3 and A[n - 1, n - 3] != 0.0:
            c0, c1, c2 = A[n - 1, n - 1], A[n - 1, n - 2], A[n - 1, n - 3]
            a1 = A[n - 2, n - 1]
            A[n - 2, n - 2] = A[n - 2, n - 2] - a1 * c1 / c0
            A[n - 2, n - 3] = A[n - 2, n - 3] - a1 * c2 / c0
            d[n - 2] -= a1 * d[n - 1] / c0
            A[n - 2, n - 1] = 0.0
            last = (c0, c1, c2, d[n - 1])
            i1 = n - 2
        sub = A.tocsr()[i0 : i1 + 1, i0 : i1 + 1]
        sub.eliminate_zeros()
        coo = sub.tocoo()
        if np.any(np.abs(coo.row - coo.col) > 1):
            raise ValueError("Brennan-Schwartz requires a tridiagonal system")
        m = i1 - i0 + 1
        a = np.zeros(m)
        c = np.zeros(m)
        a[1:] = sub.diagonal(-1)
        c[:-1] = sub.diagonal(1)
        b = sub.diagonal().astype(float)
        dd = d[i0 : i1 + 1]
        g = obstacle[i0 : i1 + 1]
        fr = free[i0 : i1 + 1]
        if self.side == "right":
            core = self._sweep(c[::-1], b[::-1], a[::-1], dd[::-1], g[::-1], fr[::-1])[::-1]
        else:
            core = self._sweep(a, b, c, dd, g, fr)
        u = np.empty(n)
        u[i0 : i1 + 1] = core
        if first is not None:
            c0, c1, c2, d0 = first
            u[0] = (d0 - c1 * u[1] - c2 * u[2]) / c0
            if free[0]:
                u[0] = max(u[0], obstacle[0])
        if last is not None:
            c0, c1, c2, dn = last
            u[n - 1] = (dn - c1 * u[n - 2] - c2 * u[n - 3]) / c0
            if free[n - 1]:
                u[n - 1] = max(u[n - 1], obstacle[n - 1])
        return u

    def describe(self) -> str:
        return f"Brennan-Schwartz direct method (exercise on the {self.side})"


class IkonenToivanenLCP:
    r"""Operator splitting of Ikonen & Toivanen (2004).

    With a Lagrange multiplier :math:`\lambda \ge 0` carried between steps,

    .. math::
        M \tilde u &= b + \Delta\tau\, \lambda^n, \\
        u^{n+1} &= \max(\tilde u - \Delta\tau \lambda^n,\; g), \qquad
        \lambda^{n+1} = \max\!\big(0,\; \lambda^n + (g - \tilde u)/\Delta\tau\big).

    Only one linear solve (with the cached factorisation) per step; the
    splitting error is :math:`O(\Delta\tau)` locally but the scheme retains
    second-order accuracy with Crank–Nicolson in practice.
    """

    def __init__(self) -> None:
        self._lam: Optional[np.ndarray] = None

    def reset(self, n: int) -> None:
        self._lam = np.zeros(n)

    @property
    def multiplier(self) -> Optional[np.ndarray]:
        """Current Lagrange multiplier (the early-exercise 'premium rate')."""
        return self._lam

    def step(
        self,
        M: sp.csr_matrix,
        solve: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        obstacle: np.ndarray,
        guess: np.ndarray,
        dt: float,
        free: np.ndarray,
    ) -> np.ndarray:
        if self._lam is None or self._lam.size != rhs.size:
            self._lam = np.zeros(rhs.size)
        lam = np.where(free, self._lam, 0.0)
        u_t = solve(rhs + dt * lam)
        u = u_t.copy()
        u[free] = np.maximum(u_t[free] - dt * lam[free], obstacle[free])
        new_lam = np.zeros_like(lam)
        new_lam[free] = np.maximum(0.0, lam[free] + (obstacle[free] - u_t[free]) / dt)
        self._lam = new_lam
        return u

    def describe(self) -> str:
        return "Ikonen-Toivanen operator splitting"


def make_lcp(method: str, side: str = "left") -> object:
    """Factory: ``'penalty'``, ``'psor'``, ``'brennan_schwartz'`` or ``'ikonen_toivanen'``."""
    m = method.lower().replace("-", "_")
    if m == "penalty":
        return PenaltyLCP()
    if m == "psor":
        return PSORLCP()
    if m in ("brennan_schwartz", "bs"):
        return BrennanSchwartzLCP(side=side)
    if m in ("ikonen_toivanen", "it", "splitting"):
        return IkonenToivanenLCP()
    raise ValueError("method must be 'penalty', 'psor', 'brennan_schwartz' or 'ikonen_toivanen'")
