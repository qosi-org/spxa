r"""
Forward Kolmogorov (Fokker–Planck) equation for transition densities.

For :math:`dX = \mu(X, t)\,dt + \sigma(X, t)\,dW` the transition density
:math:`p(x, t)` solves

.. math::
    p_t = \tfrac12 \bigl(\sigma^2 p\bigr)_{xx} - \bigl(\mu p\bigr)_x,
    \qquad p(\cdot, 0) = \delta_{x_0}.

Rather than discretising this equation directly, the solver uses the
*discrete adjoint* of the backward generator. If :math:`A` is the
finite-difference generator acting on nodal values (the matrix used by the
backward pricing solvers with zero discounting) and :math:`V` the diagonal
matrix of control-volume widths, densities evolve with

.. math::
    P = V^{-1} A^{\mathsf T} V.

Because every row of :math:`A` sums to zero (no discounting; degenerate
"natural" boundary rows), the discrete mass :math:`\sum_i V_i p_i` is
conserved exactly, and forward and backward solutions are dual:
:math:`\sum_i V_i p_i(T) g_i` equals the backward solution with payoff
``g`` at :math:`x_0` to rounding error when both use the same time steps.
The initial delta is the nodal function :math:`e_j / V_j` at the node
:math:`x_j = x_0`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from spxa.pde.boundary import Natural
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D

__all__ = ["FokkerPlanck1D", "DensitySolution"]

Coefficient = Union[float, np.ndarray, Callable[[np.ndarray, float], np.ndarray]]


@dataclass
class DensitySolution:
    """Nodal transition density on a grid, with quadrature weights."""

    grid: Grid1D
    density: np.ndarray
    time: float
    history: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    info: dict[str, Any] = field(default_factory=dict)

    def __story__(self) -> str:
        """Plain-language summary of the density."""
        lines = [
            f"Transition density at t = {self.time:g} on {self.grid.n} nodes "
            f"[{self.grid.lo:.6g}, {self.grid.hi:.6g}]",
            f"  mass = {self.mass():.12f}, mean = {self.mean():.8f}, variance = {self.variance():.8f}",
        ]
        for key in ("scheme", "n_steps", "boundary"):
            if key in self.info:
                lines.append(f"  {key}: {self.info[key]}")
        return "\n".join(lines)

    @property
    def weights(self) -> np.ndarray:
        """Control-volume widths used as quadrature weights."""
        return self.grid.cell_widths

    def mass(self) -> float:
        """Total probability :math:`\\sum_i V_i p_i`."""
        return float(np.dot(self.weights, self.density))

    def expectation(self, g: Callable[[np.ndarray], np.ndarray]) -> float:
        """:math:`\\mathbb{E}[g(X_t)]` by the same quadrature."""
        return float(np.dot(self.weights * self.density, g(self.grid.nodes)))

    def mean(self) -> float:
        """Mean of :math:`X_t`."""
        return self.expectation(lambda x: x)

    def variance(self) -> float:
        """Variance of :math:`X_t`."""
        m = self.mean()
        return self.expectation(lambda x: (x - m) ** 2)

    def pdf(self, x: Any) -> Any:
        """Density interpolated at ``x``."""
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        out = self.grid.interpolate(self.density, xs)
        return float(out[0]) if np.ndim(x) == 0 else out

    def cdf(self, x: Any) -> Any:
        """Distribution function by cumulative quadrature over control volumes."""
        faces = self.grid.faces
        cum = np.concatenate([[0.0], np.cumsum(self.weights * self.density)])
        xs = np.asarray(x, dtype=float)
        out = np.interp(xs, faces, cum)
        return float(out) if np.ndim(x) == 0 else out


class FokkerPlanck1D:
    r"""Forward equation for :math:`dX = \mu\,dt + \sigma\,dW` on a fixed grid.

    Parameters
    ----------
    drift, volatility : coefficient
        :math:`\mu(x, t)` and :math:`\sigma(x, t)` (scalars, nodal arrays or
        callables).
    grid : Grid1D
        Spatial grid; it should comfortably contain the support of the
        density over the horizon.
    convection_scheme : {'central', 'upwind', 'auto'}
        Discretisation of the drift in the underlying generator; ``'auto'``
        upwinds only where the cell Péclet number exceeds one, which keeps
        the density non-negative.
    time_homogeneous : bool, optional
        Declare callable coefficients independent of ``t`` so that matrices
        and factorisations are reused (inferred as ``False`` for callables).
    """

    def __init__(
        self,
        drift: Coefficient,
        volatility: Coefficient,
        grid: Grid1D,
        convection_scheme: str = "auto",
        time_homogeneous: Optional[bool] = None,
    ) -> None:
        self.grid = grid
        self.drift = drift
        self.volatility = volatility

        def diffusion(x: np.ndarray, t: float) -> np.ndarray:
            v = volatility(x, t) if callable(volatility) else volatility
            return 0.5 * np.asarray(v, dtype=float) ** 2 * np.ones_like(x)

        homogeneous = (
            not (callable(drift) or callable(volatility))
            if time_homogeneous is None
            else bool(time_homogeneous)
        )
        pde = LinearPDE1D(
            diffusion=diffusion, convection=drift, reaction=0.0, time_homogeneous=homogeneous
        )
        # natural boundaries keep every generator row summing to zero
        self.operator = DiscreteOperator1D(pde, grid, Natural(), Natural(), convection_scheme)
        self._V = grid.cell_widths
        self._cache: dict[tuple[float, float], Any] = {}
        self._P0: Optional[sp.csr_matrix] = None

    def forward_matrix(self, t: float = 0.0) -> sp.csr_matrix:
        r"""Discrete adjoint :math:`P = V^{-1} A^{\mathsf T} V` at time ``t``."""
        if self.operator.time_homogeneous and self._P0 is not None:
            return self._P0
        A, _ = self.operator.assemble(t)
        Vinv = sp.diags(1.0 / self._V)
        P = (Vinv @ A.T.tocsr() @ sp.diags(self._V)).tocsr()
        if self.operator.time_homogeneous:
            self._P0 = P
        return P

    def delta(self, x0: float) -> np.ndarray:
        """Discrete delta at the node nearest ``x0`` (unit discrete mass)."""
        j = int(np.argmin(np.abs(self.grid.nodes - x0)))
        p = np.zeros(self.grid.n)
        p[j] = 1.0 / self._V[j]
        return p

    def solve(
        self,
        initial: Union[float, np.ndarray],
        T: float,
        n_time: int = 200,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        store_history: bool = False,
    ) -> DensitySolution:
        r"""March the density from ``t = 0`` to ``T``.

        Parameters
        ----------
        initial : float or array
            Starting point :math:`x_0` (a discrete delta is used) or an
            initial nodal density.
        rannacher_steps : int
            Leading steps replaced by two backward-Euler half steps, which
            damps the oscillations Crank–Nicolson would produce from the
            delta initial condition.
        """
        if np.ndim(initial) == 0:
            p = self.delta(float(initial))
            x0: Optional[float] = float(self.grid.nodes[np.argmax(p)])
        else:
            p = np.asarray(initial, dtype=float).copy()
            if p.shape != self.grid.nodes.shape:
                raise ValueError("initial density must have one value per node")
            x0 = None
        tg = TimeGrid.uniform(T, n_time)
        times = tg.times
        n = self.grid.n
        eye = sp.identity(n, format="csr")
        hist = [p.copy()] if store_history else None
        left = int(rannacher_steps) if theta < 1.0 else 0
        n_fact = 0

        def step(p: np.ndarray, t0: float, t1: float, th: float) -> np.ndarray:
            nonlocal n_fact
            dt = t1 - t0
            key = (round(dt, 15), th)
            P1 = self.forward_matrix(t1)
            if self.operator.time_homogeneous and key in self._cache:
                lu = self._cache[key]
            else:
                lu = splu((eye - th * dt * P1).tocsc())
                n_fact += 1
                if self.operator.time_homogeneous:
                    self._cache[key] = lu
            rhs = p.copy()
            if th < 1.0:
                rhs += (1.0 - th) * dt * (self.forward_matrix(t0) @ p)
            return lu.solve(rhs)

        for k in range(1, times.size):
            t0, t1 = float(times[k - 1]), float(times[k])
            if left > 0:
                mid = 0.5 * (t0 + t1)
                p = step(step(p, t0, mid, 1.0), mid, t1, 1.0)
                left -= 1
            else:
                p = step(p, t0, t1, theta)
            if hist is not None:
                hist.append(p.copy())
        return DensitySolution(
            grid=self.grid,
            density=p,
            time=float(T),
            history=None if hist is None else np.array(hist),
            times=np.array(times) if hist is not None else None,
            info={"x0": x0, "n_steps": times.size - 1, "n_factorisations": n_fact},
        )
