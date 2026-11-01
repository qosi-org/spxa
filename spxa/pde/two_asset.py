r"""
Options on two correlated Black–Scholes assets by ADI finite differences.

In time to maturity :math:`\tau`,

.. math::
    u_\tau = \tfrac12\sigma_1^2 S_1^2 u_{11} + \rho\sigma_1\sigma_2 S_1 S_2 u_{12}
             + \tfrac12\sigma_2^2 S_2^2 u_{22}
             + (r - q_1) S_1 u_1 + (r - q_2) S_2 u_2 - r u.

The domain is :math:`[0, S_1^{\max}] \times [0, S_2^{\max}]` on sinh grids
clustered at the spots. On the axes the equation degenerates to a
one-asset Black–Scholes equation (the mixed term vanishes), so the edges
:math:`S_i = 0` use the equation itself (:class:`~spxa.pde.adi.Natural2D`)
unless the contract supplies Dirichlet data; the far edges use the linear
condition :math:`u_{S_i S_i} = 0`.

Contracts: generic payoffs :math:`g(S_1, S_2)`, exchange options (Margrabe),
spread calls and puts, and best-of / worst-of calls. European and American
exercise (Ikonen–Toivanen) are supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from spxa.pde._utils import check_positive
from spxa.pde.adi import (
    ADISolver,
    DiscreteOperator2D,
    EdgeCondition,
    Grid2D,
    Linear2D,
    LinearPDE2D,
    Natural2D,
    PDESolution2D,
)
from spxa.pde.grid import Grid1D

__all__ = ["TwoAssetBlackScholesPDE", "TwoAssetPDEResult"]

Payoff2D = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class TwoAssetPDEResult:
    """Price and grid Greeks at ``(s1, s2)``.

    ``delta = (u_1, u_2)``; ``gamma = (u_11, u_22)``; ``cross_gamma = u_12``.
    """

    price: float
    delta: tuple[float, float]
    gamma: tuple[float, float]
    cross_gamma: float
    theta: float
    spots: tuple[float, float]
    solution: PDESolution2D
    method: str = ""

    def __float__(self) -> float:
        return float(self.price)

    def __story__(self) -> str:
        """Plain-language summary of the valuation."""
        return "\n".join(
            [
                f"{self.method} at S1 = {self.spots[0]:g}, S2 = {self.spots[1]:g}",
                f"  price = {self.price:.8f}, delta = ({self.delta[0]:.6f}, {self.delta[1]:.6f}), "
                f"gamma = ({self.gamma[0]:.6f}, {self.gamma[1]:.6f}), "
                f"cross gamma = {self.cross_gamma:.6f}",
                self.solution.__story__(),
            ]
        )


class TwoAssetBlackScholesPDE:
    """Two correlated geometric Brownian motions under the risk-neutral measure."""

    def __init__(
        self,
        r: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        q1: float = 0.0,
        q2: float = 0.0,
    ) -> None:
        self.r = float(r)
        self.sigma1 = check_positive("sigma1", sigma1)
        self.sigma2 = check_positive("sigma2", sigma2)
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        self.rho = float(rho)
        self.q1 = float(q1)
        self.q2 = float(q2)

    def __repr__(self) -> str:
        return (
            f"TwoAssetBlackScholesPDE(r={self.r:g}, sigma=({self.sigma1:g}, {self.sigma2:g}), "
            f"rho={self.rho:g}, q=({self.q1:g}, {self.q2:g}))"
        )

    # ------------------------------------------------------------ set-up
    def grid(
        self,
        s1: float,
        s2: float,
        T: float,
        n1: int,
        n2: int,
        s_max: Optional[tuple[float, float]] = None,
        align: tuple[tuple[float, ...], tuple[float, ...]] = ((), ()),
    ) -> Grid2D:
        """Sinh grids clustered at the spots, 6 standard deviations wide."""
        if s_max is None:
            m1 = s1 * math.exp(6.0 * self.sigma1 * math.sqrt(T) + abs(self.r - self.q1) * T)
            m2 = s2 * math.exp(6.0 * self.sigma2 * math.sqrt(T) + abs(self.r - self.q2) * T)
        else:
            m1, m2 = s_max
        a1 = tuple(p for p in align[0] if 0.0 < p < m1 and abs(p - s1) > 1e-12 * s1)
        a2 = tuple(p for p in align[1] if 0.0 < p < m2 and abs(p - s2) > 1e-12 * s2)
        g1 = Grid1D.sinh(0.0, m1, n1, center=s1, alpha=s1 / 2.0, align=a1)
        g2 = Grid1D.sinh(0.0, m2, n2, center=s2, alpha=s2 / 2.0, align=a2)
        return Grid2D(g1, g2)

    def operator(
        self, grid: Grid2D, edges: Optional[dict[str, EdgeCondition]] = None
    ) -> DiscreteOperator2D:
        S1, S2 = grid.mesh()
        pde = LinearPDE2D(
            axx=0.5 * self.sigma1**2 * S1**2,
            axy=self.rho * self.sigma1 * self.sigma2 * S1 * S2,
            ayy=0.5 * self.sigma2**2 * S2**2,
            bx=(self.r - self.q1) * S1,
            by=(self.r - self.q2) * S2,
            c=-self.r,
        )
        default: dict[str, EdgeCondition] = {
            "x_lo": Natural2D(),
            "x_hi": Linear2D(),
            "y_lo": Natural2D(),
            "y_hi": Linear2D(),
        }
        if edges:
            default.update(edges)
        return DiscreteOperator2D(pde, grid, default, upwind="auto")

    # ------------------------------------------------------------ pricing
    def price(
        self,
        payoff: Payoff2D,
        T: float,
        s1: float,
        s2: float,
        n1: int = 101,
        n2: int = 101,
        n_time: int = 100,
        american: bool = False,
        scheme: str = "hundsdorfer_verwer",
        damping_steps: int = 2,
        s_max: Optional[tuple[float, float]] = None,
        align: tuple[tuple[float, ...], tuple[float, ...]] = ((), ()),
        edges: Optional[dict[str, EdgeCondition]] = None,
        label: str = "two-asset payoff",
    ) -> TwoAssetPDEResult:
        """Value a payoff ``g(S1, S2)`` (vectorised) at ``(s1, s2)``.

        The payoff is sampled at the nodes; ``damping_steps`` implicit
        half-step pairs smooth kinks that are not aligned with the grid
        (e.g. along :math:`S_1 = S_2`).
        """
        T = check_positive("T", T)
        s1 = check_positive("s1", s1)
        s2 = check_positive("s2", s2)
        grid = self.grid(s1, s2, T, n1, n2, s_max, align)
        op = self.operator(grid, edges)
        S1, S2 = grid.mesh()
        g = np.asarray(payoff(S1, S2), dtype=float)
        solver = ADISolver(op, scheme=scheme, damping_steps=damping_steps)
        res = solver.solve(g, T, n_time, obstacle=g if american else None)
        sol = PDESolution2D(
            grid=grid,
            values=res.values,
            previous_values=res.previous_values,
            dt_last=float(res.times[-1] - res.times[-2]),
            info=dict(res.info, model=repr(self)),
            label=label,
            variables=("S1", "S2"),
        )
        return TwoAssetPDEResult(
            price=float(sol.value(s1, s2)),
            delta=(float(sol.derivative(s1, s2, dx=1)), float(sol.derivative(s1, s2, dy=1))),
            gamma=(float(sol.derivative(s1, s2, dx=2)), float(sol.derivative(s1, s2, dy=2))),
            cross_gamma=float(sol.derivative(s1, s2, dx=1, dy=1)),
            theta=float(sol.theta(s1, s2)),
            spots=(s1, s2),
            solution=sol,
            method=f"two-asset ADI ({solver.scheme}) {'American' if american else 'European'} {label}",
        )

    def exchange(self, T: float, s1: float, s2: float, **kwargs: Any) -> TwoAssetPDEResult:
        r"""Option to exchange asset 2 for asset 1, :math:`(S_1 - S_2)^+`."""
        return self.price(
            lambda a, b: np.maximum(a - b, 0.0), T, s1, s2, label="exchange option", **kwargs
        )

    def spread(
        self, K: float, T: float, s1: float, s2: float, option: str = "call", **kwargs: Any
    ) -> TwoAssetPDEResult:
        r"""Spread option on :math:`S_1 - S_2 - K`."""
        sign = 1.0 if option == "call" else -1.0
        if option not in ("call", "put"):
            raise ValueError("option must be 'call' or 'put'")
        return self.price(
            lambda a, b: np.maximum(sign * (a - b - K), 0.0),
            T,
            s1,
            s2,
            label=f"spread {option} K={K:g}",
            **kwargs,
        )

    def best_of(self, K: float, T: float, s1: float, s2: float, **kwargs: Any) -> TwoAssetPDEResult:
        r"""Call on the maximum, :math:`(\max(S_1, S_2) - K)^+`."""
        return self.price(
            lambda a, b: np.maximum(np.maximum(a, b) - K, 0.0),
            T,
            s1,
            s2,
            label=f"best-of call K={K:g}",
            **kwargs,
        )

    def worst_of(
        self, K: float, T: float, s1: float, s2: float, **kwargs: Any
    ) -> TwoAssetPDEResult:
        r"""Call on the minimum, :math:`(\min(S_1, S_2) - K)^+`."""
        return self.price(
            lambda a, b: np.maximum(np.minimum(a, b) - K, 0.0),
            T,
            s1,
            s2,
            label=f"worst-of call K={K:g}",
            **kwargs,
        )
