r"""
One-factor short-rate PDEs (Vasicek / Hull–White with constant parameters, CIR).

Zero-coupon bonds and claims on them solve

.. math::
    u_\tau = \tfrac12 \sigma^2 r^{2\beta} u_{rr} + \kappa(\theta - r) u_r - r u,

with :math:`\beta = 0` (Vasicek) or :math:`\beta = \tfrac12` (CIR). The
Vasicek domain spans :math:`\pm 8` conditional standard deviations around
:math:`r_0` and :math:`\theta` with zero-curvature conditions at both ends
(the bond price is log-linear in :math:`r`, so the truncation error is
second order in the far-field curvature). For CIR the domain starts at
:math:`r = 0`, where the diffusion vanishes and the drift
:math:`\kappa\theta \ge 0` points into the domain, so the equation itself
(with a one-sided derivative) is the correct boundary condition whether or
not the Feller condition holds.

Bond options are solved in two stages on a common grid: the bond
:math:`P(T, S; r)` on :math:`[T, S]`, then the option with payoff
:math:`(P(T, S; r) - K)^\pm` on :math:`[0, T]`. The kink of that payoff at
:math:`r^*` with :math:`P(T, S; r^*) = K` is located from a first pass and
placed exactly on a node of the final grid. American bond options use the
bond price at each exercise time as the obstacle, read from the stored
first-stage history on a matching time grid.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.american import make_lcp
from spxa.pde.boundary import BoundaryCondition, Linear, Natural
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.solution import PDEPricingResult, PDESolution1D
from spxa.pde.theta import ThetaSolver1D

__all__ = ["ShortRatePDE"]

_MODELS = ("vasicek", "cir")


class ShortRatePDE:
    r"""Vasicek or CIR short-rate model for PDE pricing.

    Parameters
    ----------
    kappa, theta, sigma : float
        Mean-reversion speed, long-run level and volatility.
    model : {'vasicek', 'cir'}
    """

    def __init__(self, kappa: float, theta: float, sigma: float, model: str = "vasicek") -> None:
        self.kappa = check_positive("kappa", kappa)
        self.theta = float(theta)
        self.sigma = check_positive("sigma", sigma)
        self.model = str(model).lower()
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}")
        if self.model == "cir" and self.theta < 0.0:
            raise ValueError("CIR requires theta >= 0")

    def __repr__(self) -> str:
        return (
            f"ShortRatePDE(kappa={self.kappa:g}, theta={self.theta:g}, "
            f"sigma={self.sigma:g}, model='{self.model}')"
        )

    @property
    def feller(self) -> bool:
        """Whether :math:`2\\kappa\\theta \\ge \\sigma^2` (CIR only)."""
        return self.model == "cir" and 2 * self.kappa * self.theta >= self.sigma**2

    def pde(self) -> LinearPDE1D:
        """Backward operator in ``r`` with discounting at the short rate."""
        k, th, s2 = self.kappa, self.theta, 0.5 * self.sigma**2
        if self.model == "vasicek":
            diffusion: Any = s2
        else:

            def diffusion(r: np.ndarray, tau: float) -> np.ndarray:
                return s2 * np.maximum(r, 0.0)

        return LinearPDE1D(
            diffusion=diffusion,
            convection=lambda r, tau: k * (th - r),
            reaction=lambda r, tau: -r,
            time_homogeneous=True,
        )

    def default_domain(self, r0: float, T: float) -> tuple[float, float]:
        """Truncated rate domain for horizon ``T``."""
        k = self.kappa
        var_factor = (1.0 - math.exp(-2.0 * k * T)) / (2.0 * k)
        if self.model == "vasicek":
            sd = self.sigma * math.sqrt(var_factor)
            return min(r0, self.theta) - 8.0 * sd, max(r0, self.theta) + 8.0 * sd
        level = max(r0, self.theta, self.sigma**2 / (2.0 * k))
        sd = self.sigma * math.sqrt(level * var_factor)
        return 0.0, max(r0, self.theta) + 10.0 * sd

    def _boundaries(self) -> tuple[BoundaryCondition, BoundaryCondition]:
        left: BoundaryCondition = Natural() if self.model == "cir" else Linear()
        return left, Linear()

    def _grid(
        self, lo: float, hi: float, n: int, center: float, align: Sequence[float] = ()
    ) -> Grid1D:
        # mild clustering: rates claims are smooth except at an option kink
        return Grid1D.sinh(lo, hi, n, center=center, alpha=0.5 * (hi - lo), align=align)

    def _march(
        self,
        grid: Grid1D,
        initial: np.ndarray,
        time_grid: TimeGrid,
        events: Optional[dict[float, Any]] = None,
        rannacher: int = 2,
        lcp: Any = None,
        obstacle: Any = None,
        store_history: bool = False,
    ) -> Any:
        left, right = self._boundaries()
        op = DiscreteOperator1D(self.pde(), grid, left, right, convection_scheme="auto")
        solver = ThetaSolver1D(op, theta=0.5, rannacher_steps=rannacher)
        return solver.solve(
            initial,
            time_grid,
            events=events,
            lcp=lcp,
            obstacle=obstacle,
            store_history=store_history,
        )

    def _result(self, sol: PDESolution1D, r0: float, method: str) -> PDEPricingResult:
        g = sol.greeks(r0)
        return PDEPricingResult(
            price=g["value"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g.get("theta", float("nan")),
            spot=float(r0),
            solution=sol,
            method=method,
        )

    # ------------------------------------------------------------------ products
    def zero_coupon_bond(
        self, T: float, r0: float, n_space: int = 401, n_time: int = 200
    ) -> PDEPricingResult:
        r"""Zero-coupon bond :math:`P(0, T)`; ``delta``/``gamma`` are
        :math:`\partial P/\partial r` and :math:`\partial^2 P/\partial r^2`."""
        return self.coupon_bond([(T, 1.0)], r0, n_space=n_space, n_time=n_time)

    def coupon_bond(
        self,
        cashflows: Sequence[tuple[float, float]],
        r0: float,
        n_space: int = 401,
        n_time: int = 200,
    ) -> PDEPricingResult:
        """Bond paying ``amount`` at each ``(time, amount)``; the last date is maturity."""
        flows = sorted((float(t), float(c)) for t, c in cashflows)
        if not flows or flows[0][0] <= 0.0:
            raise ValueError("cash-flow times must be positive")
        T = flows[-1][0]
        lo, hi = self.default_domain(r0, T)
        grid = self._grid(lo, hi, n_space, center=r0)
        events: dict[float, Any] = {}
        terminal = 0.0
        for t, c in flows:
            if abs(t - T) < 1e-14:
                terminal += c
            else:
                events[T - t] = lambda u, x, tau, c=c: u + c
        tg = TimeGrid.uniform(T, n_time, events.keys())
        res = self._march(grid, np.full(grid.n, terminal), tg, events=events, rannacher=0)
        sol = PDESolution1D.from_theta_result(
            res,
            variable="S",
            label=f"{self.model} bond ({len(flows)} cash flows)",
            info={"variable": "short rate r", "model": self.model},
        )
        return self._result(sol, r0, f"{self.model} short-rate PDE bond")

    def bond_option(
        self,
        K: float,
        T: float,
        S: float,
        r0: float,
        option: str = "call",
        american: bool = False,
        method: str = "penalty",
        n_space: int = 401,
        n_time: int = 200,
    ) -> PDEPricingResult:
        """European or American option expiring at ``T`` on the zero-coupon bond maturing at ``S``."""
        opt = check_option_type(option)
        T = check_positive("T", T)
        if not S > T:
            raise ValueError("bond maturity S must exceed option expiry T")
        K = check_positive("K", K)
        lo, hi = self.default_domain(r0, S)
        n_bond = int(n_time)  # the bond stage is smooth; same step count suffices

        # pass 1: locate the kink r* where P(T, S; r*) = K
        g0 = self._grid(lo, hi, n_space, center=r0)
        p0 = self._march(g0, np.ones(g0.n), TimeGrid.uniform(S - T, n_bond), rannacher=0).values
        f = p0 - K
        if not (f[0] > 0.0 > f[-1]):
            raise ValueError("strike is outside the range of attainable bond prices")
        j = int(np.argmax(f < 0.0))
        r_star = brentq(
            lambda x: float(g0.interpolate(f, np.array([x]))[0]), g0.nodes[j - 1], g0.nodes[j]
        )

        # pass 2: grid with r0 at the centre and r* on a node
        grid = self._grid(lo, hi, n_space, center=r0, align=[r_star])
        opt_grid = TimeGrid.uniform(T, n_time)
        if american:
            # bond history on [0, S - T] followed by the option's time levels
            taus = np.concatenate(
                [np.linspace(0.0, S - T, n_bond + 1), (S - T) + opt_grid.times[1:]]
            )
            bond = self._march(
                grid, np.ones(grid.n), TimeGrid(taus), rannacher=0, store_history=True
            )
            hist = bond.history[n_bond:]
            bond_at_T = hist[0]
            hist_taus = opt_grid.times
            sign = 1.0 if opt == "call" else -1.0

            def obstacle(x: np.ndarray, tau: float) -> np.ndarray:
                k = int(np.clip(np.searchsorted(hist_taus, tau) - 1, 0, len(hist_taus) - 2))
                w = (tau - hist_taus[k]) / (hist_taus[k + 1] - hist_taus[k])
                p = (1.0 - w) * hist[k] + w * hist[k + 1]
                return np.maximum(sign * (p - K), 0.0)

            side = "left" if opt == "call" else "right"  # calls are exercised at low rates
            lcp: Any = make_lcp(method, side=side)
        else:
            bond_at_T = self._march(
                grid, np.ones(grid.n), TimeGrid.uniform(S - T, n_bond), rannacher=0
            ).values
            obstacle = lcp = None
        payoff = np.maximum((bond_at_T - K) if opt == "call" else (K - bond_at_T), 0.0)
        res = self._march(grid, payoff, opt_grid, lcp=lcp, obstacle=obstacle)
        kind = "American" if american else "European"
        sol = PDESolution1D.from_theta_result(
            res,
            variable="S",
            label=f"{kind} {opt} on P(., {S:g}), K={K:g}, T={T:g}",
            info={"variable": "short rate r", "model": self.model, "r_star": r_star},
        )
        return self._result(sol, r0, f"{self.model} short-rate PDE {kind} bond {opt}")
