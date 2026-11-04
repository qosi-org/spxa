r"""Hull–White pricing by finite differences on the state :math:`x`.

A claim :math:`V(t, x)` on the model of :mod:`spxa.rates.hull_white`
satisfies

.. math:: \partial_t V + \bigl(y(t) - a x\bigr)\partial_x V + \tfrac12\sigma(t)^2\partial_{xx} V - \bigl(f(0, t) + x\bigr)V = 0.

It is solved backwards in :math:`\tau = T - t` with the Crank–Nicolson
solver of :mod:`spxa.pde` (Rannacher start, linear boundary conditions)
on a uniform grid of :math:`\pm` ``n_sd`` standard deviations of
:math:`x_T`, with a node at :math:`x = 0`. Bond prices at intermediate
dates are taken from the closed form, so the only discretisation is that
of the claim itself. American and Bermudan exercise use the
complementarity solvers of :mod:`spxa.pde`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.pde.american import make_lcp
from spxa.pde.boundary import Linear
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.theta import ThetaSolver1D
from spxa.rates.hull_white import HullWhite

__all__ = ["HullWhitePDE"]


class HullWhitePDE:
    """Finite-difference pricer for :class:`~spxa.rates.hull_white.HullWhite`."""

    def __init__(
        self, model: HullWhite, n_space: int = 201, n_time: int = 200, n_sd: float = 7.0
    ) -> None:
        self.model = model
        self.n_space = int(n_space) | 1  # odd, so x = 0 is a node
        self.n_time = int(n_time)
        self.n_sd = float(n_sd)

    def _sigma(self, t: float) -> float:
        m = self.model
        return float(m.sigma[np.searchsorted(m.sigma_times, t, side="right")])

    def _pde(self, T: float) -> LinearPDE1D:
        m = self.model

        def diffusion(x: np.ndarray, tau: float) -> np.ndarray:
            return np.full_like(x, 0.5 * self._sigma(T - tau) ** 2)

        def convection(x: np.ndarray, tau: float) -> np.ndarray:
            return m.y(T - tau) - m.a * x

        def reaction(x: np.ndarray, tau: float) -> np.ndarray:
            return -(m.phi(max(T - tau, 0.0)) + x)

        return LinearPDE1D(diffusion=diffusion, convection=convection, reaction=reaction)

    def grid(self, T: float) -> Grid1D:
        half = self.n_sd * np.sqrt(max(self.model.y(T), self.model.y(1.0) * 1e-2))
        return Grid1D.uniform(-half, half, self.n_space)

    def solve(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        T: float,
        exercise: Optional[dict[float, Callable[[np.ndarray], np.ndarray]]] = None,
        american: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        lcp: str = "brennan_schwartz",
        side: str = "left",
    ) -> tuple[Grid1D, np.ndarray]:
        """Value at :math:`t = 0` on the grid of a claim paying ``payoff(x)`` at ``T``.

        ``exercise`` maps Bermudan dates :math:`t` to exercise values
        :math:`g(x)`. ``american(x, t)`` is a continuous exercise value.
        """
        grid = self.grid(T)
        events: dict[float, Any] = {}
        for t, g in (exercise or {}).items():
            if 0 < t < T:
                events[T - t] = lambda u, x, tau, g=g: np.maximum(u, g(x))
        # volatility knots on the time grid keep the scheme second order
        for k in self.model.sigma_times[self.model.sigma_times < T]:
            events.setdefault(T - float(k), lambda u, x, tau: u)
        tg = TimeGrid.uniform(T, self.n_time, events.keys())
        op = DiscreteOperator1D(self._pde(T), grid, Linear(), Linear(), convection_scheme="auto")
        solver = ThetaSolver1D(op, theta=0.5, rannacher_steps=2)
        obstacle = None if american is None else (lambda x, tau: american(x, T - tau))
        res = solver.solve(
            np.asarray(payoff(grid.nodes), float),
            tg,
            events=events,
            lcp=make_lcp(lcp, side=side) if american is not None else None,
            obstacle=obstacle,
        )
        return grid, np.asarray(res.values, float)

    def value(self, payoff: Callable[[np.ndarray], np.ndarray], T: float, **kw: Any) -> float:
        grid, v = self.solve(payoff, T, **kw)
        return float(v[self.n_space // 2])

    def zero_bond(self, T: float) -> float:
        return self.value(lambda x: np.ones_like(x), T)

    def zcb_option(
        self, T: float, S: float, K: float, option: str = "call", american: bool = False
    ) -> float:
        eta = 1.0 if option == "call" else -1.0
        m = self.model

        def pay(x: np.ndarray) -> np.ndarray:
            return np.maximum(eta * (m.discount_bond(T, S, x) - K), 0.0)

        def ex(x: np.ndarray, t: float) -> np.ndarray:
            return np.maximum(eta * (m.discount_bond(t, S, x) - K), 0.0)

        # bond prices fall with x: puts are exercised at high x
        return self.value(
            pay, T, american=ex if american else None, side="right" if eta < 0 else "left"
        )

    def swaption(self, swaption: Any, projection: Optional[YieldCurve] = None) -> float:
        """European physically settled swaption."""
        from spxa.rates.model_cashflows import swap_cashflows

        m = self.model
        T = float(m.curve.time(swaption.expiry))
        cf = swap_cashflows(swaption.swap, m.curve, projection)
        return self.value(lambda x: np.maximum(cf.value(m, T, x), 0.0), T)
