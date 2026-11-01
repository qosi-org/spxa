r"""
European and American options under the Heston model by ADI finite differences.

In time to maturity :math:`\tau` the price :math:`u(S, v, \tau)` solves

.. math::
    u_\tau = \tfrac12 S^2 v\,u_{SS} + \rho\sigma S v\,u_{Sv} + \tfrac12\sigma^2 v\,u_{vv}
             + (r - q) S\,u_S + \kappa(\theta - v)\,u_v - r u

on :math:`[0, S_{\max}] \times [0, v_{\max}]` with (in 't Hout & Foulon, 2010)

* :math:`S = 0`: Dirichlet (``0`` for calls, :math:`K e^{-r\tau}` for
  European puts, :math:`K` for American puts);
* :math:`S = S_{\max}`: Neumann, :math:`u_S = e^{-q\tau}` for calls (``1``
  for American calls) and ``0`` for puts;
* :math:`v = 0`: the equation itself, which degenerates to a first-order
  equation there (no boundary condition is needed when the Feller condition
  fails or holds), discretised with a one-sided :math:`u_v`;
* :math:`v = v_{\max}`: :math:`u_v = 0`.

The mixed derivative vanishes on the Neumann edges (the data do not depend
on the other variable) and its coefficient vanishes at :math:`S = 0` and
:math:`v = 0`, so omitting it at edge nodes is exact. The grids are
sinh-stretched around the strike (:math:`c = K/5`) and around
:math:`v = 0` (:math:`d = v_{\max}/500`), the strike is a node, and the
first derivative in :math:`v` is upwinded (second order) where convection
dominates, i.e. at large variance. Time stepping is any of the ADI schemes
of :class:`~spxa.pde.adi.ADISolver` (Hundsdorfer–Verwer by default) with
two implicit damping half-steps; American options use the Ikonen–Toivanen
splitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.adi import (
    ADISolver,
    Dirichlet2D,
    DiscreteOperator2D,
    Grid2D,
    LinearPDE2D,
    Natural2D,
    Neumann2D,
    PDESolution2D,
)
from spxa.pde.grid import Grid1D
from spxa.pde.payoffs import project_payoff, vanilla
from spxa.pde.solution import ExerciseBoundary

__all__ = ["HestonPDE", "HestonPDEResult"]


@dataclass
class HestonPDEResult:
    """Price and grid sensitivities of a Heston PDE valuation.

    Attributes
    ----------
    price, delta, gamma, theta : float
        Value, :math:`u_S`, :math:`u_{SS}` and calendar theta at ``(spot, v0)``.
    dv : float
        Sensitivity to the instantaneous variance, :math:`u_v`.
    vega : float
        Sensitivity to the instantaneous volatility
        :math:`\\sqrt{v}`: :math:`2\\sqrt{v_0}\\,u_v`.
    spot, v0 : float
        Evaluation point.
    solution : PDESolution2D
        Full grid solution in ``(S, v)``.
    exercise_boundary : ExerciseBoundary, optional
        For American options, the boundary in ``S`` along the variance node
        nearest ``v0``.
    method : str
    """

    price: float
    delta: float
    gamma: float
    theta: float
    dv: float
    vega: float
    spot: float
    v0: float
    solution: PDESolution2D
    exercise_boundary: Optional[ExerciseBoundary] = None
    method: str = ""

    def __story__(self) -> str:
        """Plain-language summary of the valuation."""
        lines = [
            f"{self.method} at S = {self.spot:g}, v = {self.v0:g}",
            f"  price = {self.price:.8f}, delta = {self.delta:.6f}, gamma = {self.gamma:.6f}, "
            f"theta = {self.theta:.6f}, u_v = {self.dv:.6f}, vega = {self.vega:.6f}",
        ]
        if self.exercise_boundary is not None and self.exercise_boundary.taus.size:
            lines.append(f"  exercise boundary today: {self.exercise_boundary.locations[-1]:.6g}")
        lines.append(self.solution.__story__())
        return "\n".join(lines)

    def __float__(self) -> float:
        return float(self.price)


class HestonPDE:
    r"""Heston model :math:`dS = (r - q)S\,dt + \sqrt{v}S\,dW_1`,
    :math:`dv = \kappa(\theta - v)\,dt + \sigma\sqrt{v}\,dW_2`,
    :math:`d\langle W_1, W_2\rangle = \rho\,dt`.
    """

    def __init__(
        self,
        r: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        q: float = 0.0,
    ) -> None:
        self.r = float(r)
        self.q = float(q)
        self.kappa = check_positive("kappa", kappa)
        self.theta = check_positive("theta", theta)
        self.sigma = check_positive("sigma", sigma)
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        self.rho = float(rho)

    def __repr__(self) -> str:
        return (
            f"HestonPDE(r={self.r:g}, q={self.q:g}, kappa={self.kappa:g}, "
            f"theta={self.theta:g}, sigma={self.sigma:g}, rho={self.rho:g})"
        )

    @property
    def feller(self) -> bool:
        r""":math:`2\kappa\theta \ge \sigma^2` (zero variance is unattainable)."""
        return 2.0 * self.kappa * self.theta >= self.sigma**2

    @classmethod
    def from_process(cls, process: Any, r: float, q: float = 0.0) -> "HestonPDE":
        """Build from a zoo ``HestonProcess`` (its drift is replaced by ``r - q``)."""
        return cls(r, process.kappa, process.theta, process.sigma_v, process.rho, q=q)

    # ------------------------------------------------------------ set-up
    def grid(
        self,
        K: float,
        n_s: int,
        n_v: int,
        s_max: Optional[float] = None,
        v_max: Optional[float] = None,
        spot: Optional[float] = None,
    ) -> Grid2D:
        """Sinh grids clustered at the strike and at zero variance."""
        s_hi = 8.0 * K if s_max is None else float(s_max)
        v_hi = 5.0 if v_max is None else float(v_max)
        align = () if spot is None or abs(spot - K) < 1e-12 * K else (float(spot),)
        gs = Grid1D.sinh(0.0, s_hi, n_s, center=K, alpha=K / 5.0, align=align)
        gv = Grid1D.sinh(0.0, v_hi, n_v, center=0.0, alpha=v_hi / 500.0)
        return Grid2D(gs, gv)

    def operator(self, grid: Grid2D, K: float, option: str, american: bool) -> DiscreteOperator2D:
        S, V = grid.mesh()
        pde = LinearPDE2D(
            axx=0.5 * S**2 * V,
            axy=self.rho * self.sigma * S * V,
            ayy=0.5 * self.sigma**2 * V,
            bx=(self.r - self.q) * S,
            by=self.kappa * (self.theta - V),
            c=-self.r,
        )
        r, q = self.r, self.q
        if option == "call":
            left = Dirichlet2D(0.0)
            if american:
                right = Neumann2D(1.0)
            else:
                right = Neumann2D(lambda v, tau: np.full_like(v, math.exp(-q * tau)))
        else:
            if american:
                left = Dirichlet2D(K)
            else:
                left = Dirichlet2D(lambda v, tau: np.full_like(v, K * math.exp(-r * tau)))
            right = Neumann2D(0.0)
        edges = {"x_lo": left, "x_hi": right, "y_lo": Natural2D(), "y_hi": Neumann2D(0.0)}
        return DiscreteOperator2D(pde, grid, edges, upwind="auto")

    # ------------------------------------------------------------ pricing
    def _price(
        self,
        K: float,
        T: float,
        spot: float,
        v0: float,
        option: str,
        american: bool,
        n_s: int,
        n_v: int,
        n_time: int,
        scheme: str,
        s_max: Optional[float],
        v_max: Optional[float],
        damping_steps: int,
    ) -> HestonPDEResult:
        opt = check_option_type(option)
        K = check_positive("K", K)
        T = check_positive("T", T)
        spot = check_positive("spot", spot)
        if v0 < 0.0:
            raise ValueError("v0 must be non-negative")
        grid = self.grid(K, n_s, n_v, s_max, v_max, spot)
        if not (spot < grid.x.nodes[-1] and v0 <= grid.y.nodes[-1]):
            raise ValueError("(spot, v0) lies outside the computational domain")
        op = self.operator(grid, K, opt, american)
        payoff = vanilla(K, opt)
        u_s = project_payoff(payoff, grid.x, method="kink")
        initial = np.repeat(u_s[:, None], grid.y.n, axis=1)
        solver = ADISolver(op, scheme=scheme, damping_steps=damping_steps)
        obstacle = None
        record = None
        taus: list[float] = []
        locs: list[float] = []
        if american:
            g_s = payoff(grid.x.nodes)
            obstacle = np.repeat(g_s[:, None], grid.y.n, axis=1)
            jv = int(np.argmin(np.abs(grid.y.nodes - v0)))
            side = "left" if opt == "put" else "right"

            def record(k: int, tau: float, u: np.ndarray) -> None:
                col = u[:, jv]
                ex = (col - g_s <= 1e-7 * K) & (g_s > 0.0)
                taus.append(tau)
                if not np.any(ex):
                    locs.append(float("nan"))
                else:
                    s_ex = grid.x.nodes[ex]
                    locs.append(float(np.max(s_ex) if side == "left" else np.min(s_ex)))

        res = solver.solve(initial, T, n_time, obstacle=obstacle, callback=record)
        sol = PDESolution2D(
            grid=grid,
            values=res.values,
            previous_values=res.previous_values,
            dt_last=float(res.times[-1] - res.times[-2]),
            info=dict(res.info, model=repr(self), feller=self.feller),
            label=f"Heston {'American' if american else 'European'} {opt}",
            variables=("S", "v"),
        )
        eb = None
        if american:
            eb = ExerciseBoundary(np.array(taus), np.array(locs), side)
            sol.info["multiplier"] = res.multiplier
        price = sol.value(spot, v0)
        dv = sol.derivative(spot, v0, dy=1)
        return HestonPDEResult(
            price=float(price),
            delta=float(sol.derivative(spot, v0, dx=1)),
            gamma=float(sol.derivative(spot, v0, dx=2)),
            theta=float(sol.theta(spot, v0)),
            dv=float(dv),
            vega=float(2.0 * math.sqrt(v0) * dv),
            spot=spot,
            v0=float(v0),
            solution=sol,
            exercise_boundary=eb,
            method=f"Heston ADI ({solver.scheme}) {'American' if american else 'European'} {opt}",
        )

    def european(
        self,
        K: float,
        T: float,
        spot: float,
        v0: float,
        option: str = "call",
        n_s: int = 101,
        n_v: int = 51,
        n_time: int = 100,
        scheme: str = "hundsdorfer_verwer",
        s_max: Optional[float] = None,
        v_max: Optional[float] = None,
        damping_steps: int = 1,
    ) -> HestonPDEResult:
        """European call or put at ``(spot, v0)``."""
        return self._price(
            K, T, spot, v0, option, False, n_s, n_v, n_time, scheme, s_max, v_max, damping_steps
        )

    def american(
        self,
        K: float,
        T: float,
        spot: float,
        v0: float,
        option: str = "put",
        n_s: int = 101,
        n_v: int = 51,
        n_time: int = 100,
        scheme: str = "hundsdorfer_verwer",
        s_max: Optional[float] = None,
        v_max: Optional[float] = None,
        damping_steps: int = 1,
    ) -> HestonPDEResult:
        """American call or put: ADI with the Ikonen–Toivanen splitting."""
        return self._price(
            K, T, spot, v0, option, True, n_s, n_v, n_time, scheme, s_max, v_max, damping_steps
        )
