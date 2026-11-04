r"""
Forward (Fokker–Planck) equation of the SLV model on ADI.

The joint density :math:`p(t, y, v)` of the SLV model in
:mod:`spxa.vol.slv` solves

.. math::
    p_t = \tfrac12\partial_{yy}(L^2 v p) + \partial_{yv}(\rho\xi L v p)
        + \tfrac12\partial_{vv}(\xi^2 v p)
        + \tfrac12\partial_y(L^2 v p) - \partial_v(\kappa(\theta - v)p),
    \qquad p(0) = \delta_{(0, v_0)}.

As in :class:`spxa.pde.FokkerPlanck1D`, the equation is discretised as the
*discrete adjoint* of the backward generator: :class:`spxa.pde.DiscreteOperator2D`
supplies the splitting :math:`A = A_0 + A_1 + A_2` into mixed, ``y`` and ``v``
parts, and with :math:`W` the control-volume areas the density evolves with
:math:`P_j = W^{-1}A_j^{\mathsf T}W`. Every generator row sums to zero
(zero discounting), so the discrete mass :math:`\sum W p` is conserved to
rounding. The degenerate edge :math:`v = 0` is "natural" (the equation holds
there with one-sided differences); the far edges in ``y`` and the top of the
variance grid are absorbing (their generator rows vanish), which keeps the
splitting stable and parks the negligible mass that reaches them.

Time stepping is Hundsdorfer–Verwer ADI applied to :math:`P_0 + P_1 + P_2`,
with the first steps replaced by implicit Douglas half-steps to absorb the
initial delta. The leverage is frozen over each step at its value at the
start of the step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from spxa.pde.adi import Dirichlet2D, DiscreteOperator2D, Grid2D, LinearPDE2D, Natural2D
from spxa.vol.heston import Heston
from spxa.vol.slv import SLVGrid

__all__ = ["SLVDensity", "SLVForward", "slv_forward"]

Leverage = Callable[[np.ndarray, float], Any]
_HV_THETA = 0.5 + math.sqrt(3.0) / 6.0
_ABSORB = Dirichlet2D(0.0)


@dataclass
class SLVDensity:
    """Joint density of ``(y, v)`` at time ``t`` on an :class:`SLVGrid`."""

    grid: SLVGrid
    t: float
    p: np.ndarray

    @property
    def mass(self) -> float:
        return float(np.sum(self.p * self.grid.weights))

    def marginal(self) -> np.ndarray:
        """Density of ``y`` at the ``y`` nodes."""
        return np.asarray(self.p @ self.grid.v.cell_widths)

    def expectation(self, g: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
        Y, V = np.meshgrid(self.grid.y.nodes, self.grid.v.nodes, indexing="ij")
        return float(np.sum(g(Y, V) * self.p * self.grid.weights))

    def forward(self) -> float:
        r""":math:`E[e^{y_t}]`, one for an exact martingale."""
        return self.expectation(lambda Y, V: np.exp(Y))

    def conditional_variance(self, floor: float = 1e-8) -> np.ndarray:
        r""":math:`E[v_t \mid y_t]` at the ``y`` nodes.

        Nodes whose marginal density is below ``floor`` times its maximum
        carry no information and take the value of the nearest valid node.
        """
        q = np.maximum(self.p, 0.0) * self.grid.v.cell_widths
        m0, m1 = q.sum(axis=1), q @ self.grid.v.nodes
        ok = m0 > floor * m0.max()
        out = np.empty_like(m0)
        out[ok] = m1[ok] / m0[ok]
        idx = np.flatnonzero(ok)
        bad = np.flatnonzero(~ok)
        if bad.size:
            near = idx[np.abs(bad[:, None] - idx[None, :]).argmin(axis=1)]
            out[bad] = out[near]
        return out

    def call(self, k: Any) -> Any:
        r"""Normalised undiscounted calls :math:`E[(e^{y_t} - e^k)^+]`."""
        m = self.marginal() * self.grid.y.cell_widths
        ey = np.exp(self.grid.y.nodes)
        kk = np.atleast_1d(np.asarray(k, float))
        out = np.array([float(np.sum(np.maximum(ey - math.exp(x), 0.0) * m)) for x in kk])
        return float(out[0]) if np.ndim(k) == 0 else out.reshape(np.shape(k))

    def implied_vol(self, k: Any) -> Any:
        """Black implied vols at log-moneyness ``k`` (OTM side, forward one)."""
        from spxa.market.implied import implied_black_volatility

        kk = np.asarray(k, float)
        c = np.asarray(self.call(kk), float)
        K = np.exp(kk)
        otm = np.where(kk >= 0, c, c - 1.0 + K)
        kind = np.where(kk >= 0, "call", "put")
        vol = np.asarray(implied_black_volatility(otm, 1.0, K, self.t, 1.0, kind), float)
        return float(vol) if vol.ndim == 0 else vol


@dataclass
class SLVForward:
    """Densities at the recorded times (always including the horizon)."""

    grid: SLVGrid
    times: np.ndarray
    densities: dict[float, SLVDensity] = field(default_factory=dict)
    n_factorisations: int = 0

    def __getitem__(self, t: float) -> SLVDensity:
        key = min(self.densities, key=lambda s: abs(s - t))
        if abs(key - t) > 1e-9:
            raise KeyError(f"no density recorded at t = {t}")
        return self.densities[key]

    @property
    def final(self) -> SLVDensity:
        return self.densities[max(self.densities)]


class _ForwardStepper:
    """Adjoint ADI matrices for a given leverage profile ``L(y)``."""

    def __init__(self, model: Heston, grid: SLVGrid) -> None:
        self.model, self.grid = model, grid
        self.g2 = Grid2D(grid.y, grid.v)
        Y, V = self.g2.mesh()
        self.Y, self.V = Y, V
        w = grid.weights.ravel()
        self.W, self.Winv = sp.diags(w), sp.diags(1.0 / w)
        self.eye = sp.identity(w.size, format="csc")
        self.n_fact = 0

    def matrices(self, lev: np.ndarray) -> list[sp.csr_matrix]:
        m, V = self.model, self.V
        L = np.asarray(lev, float)[:, None] * np.ones_like(V)
        ayy = 0.5 * L * L * V
        pde = LinearPDE2D(
            axx=ayy,
            axy=m.rho * m.xi * L * V,
            ayy=0.5 * m.xi**2 * V,
            bx=-ayy,
            by=m.kappa * (m.theta - V),
            time_homogeneous=True,
        )
        edges = {"x_lo": _ABSORB, "x_hi": _ABSORB, "y_lo": Natural2D(), "y_hi": _ABSORB}
        op = DiscreteOperator2D(pde, self.g2, edges)
        return [(self.Winv @ A.T.tocsr() @ self.W).tocsr() for A in op.matrices()]

    def step(self, p: np.ndarray, P: list[sp.csr_matrix], dt: float, damped: bool) -> np.ndarray:
        th = 1.0 if damped else _HV_THETA
        lu = [splu((self.eye - th * dt * P[j]).tocsc()) for j in (1, 2)]
        self.n_fact += 2

        def F(u: np.ndarray) -> np.ndarray:
            return np.asarray(P[0] @ u + P[1] @ u + P[2] @ u)

        Y0 = p + dt * F(p)
        Y1 = lu[0].solve(Y0 - th * dt * (P[1] @ p))
        Y2 = lu[1].solve(Y1 - th * dt * (P[2] @ p))
        if damped:
            return np.asarray(Y2)
        Z0 = Y0 + 0.5 * dt * (F(Y2) - F(p))
        Z1 = lu[0].solve(Z0 - th * dt * (P[1] @ Y2))
        return np.asarray(lu[1].solve(Z1 - th * dt * (P[2] @ Y2)))


def _time_grid(T: float, n_time: int, record: Sequence[float]) -> np.ndarray:
    ts = np.linspace(0.0, T, int(n_time) + 1)
    extra = [float(t) for t in record if 0 < t < T]
    return np.unique(np.round(np.concatenate([ts, extra]), 14))


def slv_forward(
    model: Heston,
    grid: SLVGrid,
    T: float,
    n_time: int = 100,
    leverage: Optional[Leverage] = None,
    record: Sequence[float] = (),
    damping_steps: int = 2,
) -> SLVForward:
    r"""Propagate the SLV density from :math:`(0, v_0)` to ``T`` for a given leverage.

    Parameters
    ----------
    leverage : callable ``L(y, t)``, optional
        Defaults to :math:`L \equiv 1` (pure Heston).
    record : times
        Extra times at which to keep the density; they are added to the
        time grid.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    lev = leverage if leverage is not None else (lambda y, t: np.ones_like(y))
    st = _ForwardStepper(model, grid)
    ts = _time_grid(T, n_time, record)
    keep = {float(t) for t in record} | {float(ts[-1])}
    i0, j0 = grid.node(0.0, model.v0)
    p = np.zeros(grid.shape)
    p[i0, j0] = 1.0 / grid.weights[i0, j0]
    u = p.ravel()
    out = SLVForward(grid, ts)
    for n in range(ts.size - 1):
        t0, t1 = float(ts[n]), float(ts[n + 1])
        P = st.matrices(np.broadcast_to(lev(grid.y.nodes, t0), grid.y.nodes.shape))
        if n < damping_steps:
            h = 0.5 * (t1 - t0)
            u = st.step(st.step(u, P, h, True), P, h, True)
        else:
            u = st.step(u, P, t1 - t0, False)
        if any(abs(t1 - k) < 1e-12 for k in keep):
            out.densities[t1] = SLVDensity(grid, t1, u.reshape(grid.shape).copy())
    out.n_factorisations = st.n_fact
    return out
