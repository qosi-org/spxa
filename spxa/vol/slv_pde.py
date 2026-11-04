r"""
Backward pricing PDE of the SLV model on ADI.

In spot :math:`S` and variance :math:`v`, a claim :math:`u(\tau, S, v)` with
:math:`\tau = T - t` solves

.. math::
    u_\tau = \tfrac12 L^2 v S^2 u_{SS} + \rho\xi L v S\,u_{Sv} + \tfrac12\xi^2 v\,u_{vv}
        + (r - q)S\,u_S + \kappa(\theta - v)\,u_v - r u,

with the leverage evaluated at the forward log-moneyness
:math:`y = \log(S/F_t)`. The far spot edge uses the linearity condition
(exact for calls, puts and forwards, which are asymptotically linear in
:math:`S`; in log-spot the same condition would break put-call parity),
and a knock-out puts the edge of the grid at the barrier with a zero
Dirichlet condition. The equation is solved with :class:`spxa.pde.ADISolver`
(Hundsdorfer–Verwer); the variance edges are those of
:class:`spxa.pde.HestonPDE` (natural at :math:`v = 0`, zero slope at the top).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from spxa.pde.adi import (
    ADISolver,
    Dirichlet2D,
    DiscreteOperator2D,
    Grid2D,
    Linear2D,
    LinearPDE2D,
    Natural2D,
    Neumann2D,
    PDESolution2D,
)
from spxa.pde.grid import Grid1D
from spxa.vol.heston import Heston
from spxa.vol.slv import slv_grid

__all__ = ["SLVPDE"]


def _cell_average(
    f: Callable[[np.ndarray], np.ndarray], nodes: np.ndarray, n_gauss: int = 8
) -> np.ndarray:
    """Average of ``f`` over the control volume of each node (half cells at the ends)."""
    faces = np.r_[nodes[0], 0.5 * (nodes[1:] + nodes[:-1]), nodes[-1]]
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    a, b = faces[:-1], faces[1:]
    pts = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * xg[None, :]
    avg = (f(pts) * wg[None, :]).sum(axis=1) / 2.0
    width = b - a
    return np.where(width > 0, avg, f(nodes))


class SLVPDE:
    """SLV pricer for a :class:`Heston` variance and leverage ``L(y, t)`` with rates ``r``, ``q``."""

    def __init__(
        self,
        model: Heston,
        leverage: Optional[Callable[[np.ndarray, float], Any]] = None,
        r: float = 0.0,
        q: float = 0.0,
        n_x: int = 161,
        n_v: int = 61,
        n_time: int = 100,
        n_sd: float = 5.0,
        vol: Optional[float] = None,
    ) -> None:
        self.model = model
        self._constant = leverage is None
        self.leverage = leverage if leverage is not None else (lambda y, t: np.ones_like(y))
        self.r, self.q = float(r), float(q)
        self.n_x, self.n_v, self.n_time, self.n_sd = int(n_x), int(n_v), int(n_time), float(n_sd)
        self.vol = vol

    def _solve(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        T: float,
        spot: float,
        lower: Optional[float],
        upper: Optional[float],
        align: tuple[float, ...],
    ) -> float:
        m = self.model
        base = slv_grid(m, T, self.n_x, self.n_v, self.n_sd, self.vol)
        half = base.y.hi
        drift = (self.r - self.q) * T
        lo = float(lower) if lower is not None else 0.0
        hi = float(upper) if upper is not None else spot * math.exp(max(half, drift + half))
        if not lo < spot < hi:
            raise ValueError("the spot must lie strictly between the barriers")
        pts = [k for k in align if lo < k < hi]
        x = Grid1D.sinh(lo, hi, self.n_x, center=spot, alpha=0.1 * spot, align=pts)
        grid = Grid2D(x, base.v)
        r, q, rho, xi = self.r, self.q, m.rho, m.xi
        lev = self.leverage
        F0 = spot

        def L(S: np.ndarray, tau: float) -> np.ndarray:
            t = T - tau
            with np.errstate(divide="ignore"):
                y = np.log(np.maximum(S[:, 0], 1e-300) / (F0 * math.exp((r - q) * t)))
            return np.asarray(lev(y, max(t - 1e-12, 0.0)), float)[:, None] * np.ones_like(S)

        pde = LinearPDE2D(
            axx=lambda S, V, tau: 0.5 * (L(S, tau) * S) ** 2 * V,
            axy=lambda S, V, tau: rho * xi * L(S, tau) * S * V,
            ayy=lambda S, V, tau: 0.5 * xi**2 * V,
            bx=lambda S, V, tau: (r - q) * S,
            by=lambda S, V, tau: m.kappa * (m.theta - V),
            c=-r,
            time_homogeneous=self._constant,
        )
        edges = {
            "x_lo": Dirichlet2D(0.0) if lower is not None else Natural2D(),
            "x_hi": Dirichlet2D(0.0) if upper is not None else Linear2D(),
            "y_lo": Natural2D(),
            "y_hi": Neumann2D(0.0),
        }
        op = DiscreteOperator2D(pde, grid, edges)
        u0 = _cell_average(payoff, x.nodes)
        initial = np.repeat(u0[:, None], base.v.n, axis=1)
        res = ADISolver(op, "hundsdorfer_verwer", damping_steps=2).solve(initial, T, self.n_time)
        sol = PDESolution2D(
            grid=grid,
            values=res.values,
            previous_values=res.previous_values,
            dt_last=float(res.times[-1] - res.times[-2]),
            info=dict(res.info),
            label="SLV",
            variables=("S", "v"),
        )
        return float(sol.value(spot, m.v0))

    def european(self, K: float, T: float, option: str = "call", spot: float = 1.0) -> float:
        sign = 1.0 if option == "call" else -1.0
        return self._solve(lambda S: np.maximum(sign * (S - K), 0.0), T, spot, None, None, (K,))

    def barrier(
        self,
        K: float,
        T: float,
        option: str = "call",
        spot: float = 1.0,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> float:
        """Continuously monitored knock-out (``lower`` and/or ``upper`` spot barriers), no rebate."""
        if lower is None and upper is None:
            raise ValueError("give a lower and/or an upper barrier")
        sign = 1.0 if option == "call" else -1.0
        return self._solve(lambda S: np.maximum(sign * (S - K), 0.0), T, spot, lower, upper, (K,))

    def price(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        T: float,
        spot: float = 1.0,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> float:
        """A general payoff of ``S_T`` with optional knock-out barriers."""
        return self._solve(payoff, T, spot, lower, upper, ())
