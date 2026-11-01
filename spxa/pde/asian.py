r"""
Arithmetic Asian options via Večeř's one-dimensional PDE.

For a fixed-strike Asian call on :math:`A = \frac1T\int_0^T S_t\,dt` (or the
discrete average :math:`\frac1n\sum_i S_{t_i}`), Večeř (2001, 2002)
replicates :math:`A - K` with a self-financing portfolio holding

.. math::
    q_t = \frac{e^{-\delta (T-t)} - e^{-r(T-t)}}{(r - \delta) T}
    \qquad\text{(continuous)}, \qquad
    q_t = \frac{e^{-r(T-t)}}{n} \sum_{t_i > t} e^{(r-\delta)(t_i - t)}
    \qquad\text{(discrete)}

shares of the underlying. Measured in units of the dividend-reinvested
asset :math:`Y_t = S_t e^{\delta t}`, the portfolio value
:math:`z_t = X_t / Y_t` is a driftless diffusion
:math:`dz = \sigma(\varphi_t - z)\,dW^Y` with :math:`\varphi_t = q_t e^{-\delta t}`,
so the price :math:`S_0\,u(0, z_0)` solves

.. math::
    u_t + \tfrac12 \sigma^2 (\varphi_t - z)^2 u_{zz} = 0,
    \qquad u(T, z) = z^+, \qquad z_0 = q_0 - e^{-rT} \frac{K - A_{\text{past}}}{S_0}.

The equation has no convection or reaction term, the coefficients are
bounded on any finite :math:`z` interval and only the diffusion is
degenerate (at :math:`z = \varphi_t`), which makes it very well suited to
central differences. Far from the kink, :math:`u \to 0` for
:math:`z \to -\infty` and :math:`u \to z` for :math:`z \to \infty` (``z`` is
a martingale); these are imposed as Dirichlet conditions. For discrete
averaging :math:`q_t` jumps at every fixing, so the equation is solved
segment by segment between fixings with continuous ``u`` across them.

References
----------
J. Večeř, "A new PDE approach for pricing arithmetic average Asian
options", J. Comput. Finance 4(4), 2001; "Unified pricing of Asian
options", Risk, 2002.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.boundary import Dirichlet
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.payoffs import Payoff, project_payoff
from spxa.pde.solution import PDEPricingResult, PDESolution1D
from spxa.pde.theta import ThetaSolver1D

__all__ = ["AsianPDE", "vecer_shares"]


def vecer_shares(
    t: float,
    T: float,
    r: float,
    q: float = 0.0,
    fixings: Optional[Sequence[float]] = None,
) -> float:
    r"""Shares :math:`q_t` of the Večeř replicating portfolio at time ``t``.

    With ``fixings=None`` the average is continuous over ``[0, T]``;
    otherwise it is the equally weighted average over the given fixing
    times (fixings at or before ``t`` are already past).
    """
    if fixings is None:
        s = T - t
        if s <= 0.0:
            return 0.0
        k = r - q
        if abs(k) * s < 1e-8:
            # series of (e^{-q s} - e^{-r s}) / (r - q) in powers of k s
            return math.exp(-q * s) * s * (1.0 - 0.5 * k * s) / T
        return (math.exp(-q * s) - math.exp(-r * s)) / (k * T)
    f = np.asarray(fixings, dtype=float)
    fut = f[f > t]
    return float(np.exp(-r * (T - t)) / f.size * np.sum(np.exp((r - q) * (fut - t))))


class AsianPDE:
    r"""Fixed-strike arithmetic Asian options under Black–Scholes dynamics.

    Parameters
    ----------
    r, sigma, q : float
        Risk-free rate, volatility and continuous dividend yield.
    """

    def __init__(self, r: float, sigma: float, q: float = 0.0) -> None:
        self.r = float(r)
        self.sigma = check_positive("sigma", sigma)
        self.q = float(q)

    def __repr__(self) -> str:
        return f"AsianPDE(r={self.r:g}, sigma={self.sigma:g}, q={self.q:g})"

    def phi(self, t: float, T: float, fixings: Optional[Sequence[float]] = None) -> float:
        r""":math:`\varphi_t = q_t e^{-\delta t}`."""
        return vecer_shares(t, T, self.r, self.q, fixings) * math.exp(-self.q * t)

    def price(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        fixings: Optional[Sequence[float]] = None,
        past_average: float = 0.0,
        n_space: int = 401,
        n_time: int = 200,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        concentration: float = 1.0,
        store_history: bool = False,
    ) -> PDEPricingResult:
        r"""Price a fixed-strike arithmetic Asian option at ``t = 0``.

        Parameters
        ----------
        K, T, spot : float
            Strike, maturity and current underlying.
        option : {'call', 'put'}
        fixings : sequence of float, optional
            Fixing times in ``(0, T]`` of a discretely sampled average.
            ``None`` means continuous averaging over ``[0, T]``.
        past_average : float
            Contribution of fixings already observed to the final average
            (e.g. :math:`\frac1n \sum_{t_i \le 0} S_{t_i}`); it lowers the
            effective strike to ``K - past_average``.
        n_space, n_time : int
            Grid sizes in ``z`` and in time (time steps are shared out
            between fixing intervals in proportion to their length).
        concentration : float
            Sinh-grid scale relative to :math:`\sigma\sqrt{T}` around the
            kink at ``z = 0``.

        Returns
        -------
        PDEPricingResult
            ``delta`` and ``gamma`` are with respect to ``spot`` (chain rule
            through :math:`z_0`); ``theta`` is not computed (``nan``).
        """
        opt = check_option_type(option)
        T = check_positive("T", T)
        spot = check_positive("spot", spot)
        if fixings is not None:
            fx = np.sort(np.asarray(fixings, dtype=float))
            if fx.size == 0 or fx[0] <= 0.0 or fx[-1] > T + 1e-14:
                raise ValueError("fixings must be non-empty and lie in (0, T]")
            fix_list: Optional[list[float]] = [float(v) for v in fx]
        else:
            fix_list = None
        q0 = vecer_shares(0.0, T, self.r, self.q, fix_list)
        z0 = q0 - math.exp(-self.r * T) * (K - past_average) / spot

        # |z - phi| is roughly lognormal with volatility sigma, so the far
        # field sits at a fixed number of standard deviations in log|z - phi|
        width = max(1.0, math.expm1(5.0 * self.sigma * math.sqrt(T)))
        lo = min(0.0, z0) - width
        hi = max(0.0, z0) + width
        alpha = concentration * max(self.sigma * math.sqrt(T), 0.02)
        grid = Grid1D.sinh(lo, hi, n_space, center=0.0, alpha=alpha)
        if opt == "call":
            payoff = Payoff(lambda z: np.maximum(z, 0.0), (0.0,), "z+")
            left, right = Dirichlet(0.0), Dirichlet(lambda tau: hi)
        else:
            payoff = Payoff(lambda z: np.maximum(-z, 0.0), (0.0,), "(-z)+")
            left, right = Dirichlet(lambda tau: -lo), Dirichlet(0.0)
        # the kink sits on a node, so pointwise sampling keeps second order and
        # makes put-call parity hold exactly (z is in the discrete null space)
        u = project_payoff(payoff, grid, method="none")

        # calendar segments on which q_t is smooth, traversed backwards from T
        breaks = sorted({0.0, T} | set(f for f in (fix_list or []) if 0.0 < f < T))
        segments = list(zip(breaks[:-1], breaks[1:]))[::-1]
        counts = [max(2, int(round(n_time * (b - a) / T))) for a, b in segments]
        sig2 = 0.5 * self.sigma**2
        history: list[np.ndarray] = []
        n_steps = 0
        previous = u
        dt_last = T
        for (t_a, t_b), m in zip(segments, counts):
            # within (t_a, t_b) the set of future fixings is fixed: those > t_a
            def a_coef(z: np.ndarray, tau: float, t_b: float = t_b, t_a: float = t_a) -> np.ndarray:
                t = min(max(t_b - tau, t_a), t_b)
                ph = self._phi_segment(t, T, fix_list, t_a)
                return sig2 * (ph - z) ** 2

            pde = LinearPDE1D(diffusion=a_coef, time_homogeneous=False)
            op = DiscreteOperator1D(pde, grid, left, right)
            rs = rannacher_steps if t_b == T else 0
            solver = ThetaSolver1D(op, theta=theta, rannacher_steps=rs)
            res = solver.solve(u, TimeGrid.uniform(t_b - t_a, m), store_history=store_history)
            previous = res.previous_values if res.previous_values is not None else u
            dt_last = (t_b - t_a) / m
            u = res.values
            n_steps += m
            if store_history and res.history is not None:
                history.append(res.history)

        sol = PDESolution1D(
            grid=grid,
            values=u,
            variable="S",
            previous_values=previous,
            dt_last=dt_last,
            info={
                "variable": "z = X / (S e^{q t}) (Vecer)",
                "z0": z0,
                "q0": q0,
                "time_steps": n_steps,
                "segments": len(segments),
                "averaging": "continuous" if fix_list is None else f"{len(fix_list)} fixings",
            },
            label=f"Vecer Asian {opt} K={K:g} T={T:g}",
        )
        if store_history and history:
            sol.info["segment_histories"] = history
        u0 = float(sol.value(z0))
        uz = float(sol.delta(z0))
        uzz = float(sol.gamma(z0))
        dz = q0 - z0  # = e^{-rT}(K - past)/S0 = S0 dz0/dS0
        return PDEPricingResult(
            price=spot * u0,
            delta=u0 + uz * dz,
            gamma=dz**2 * uzz / spot,
            theta=float("nan"),
            spot=spot,
            solution=sol,
            method=f"Vecer PDE ({sol.info['averaging']} averaging)",
        )

    def _phi_segment(
        self, t: float, T: float, fixings: Optional[Sequence[float]], t_a: float
    ) -> float:
        """phi_t using the future fixings of the segment that starts at ``t_a``."""
        if fixings is None:
            return self.phi(t, T)
        f = np.asarray(fixings, dtype=float)
        fut = f[f > t_a]
        qt = float(
            np.exp(-self.r * (T - t)) / f.size * np.sum(np.exp((self.r - self.q) * (fut - t)))
        )
        return qt * math.exp(-self.q * t)

    # convenience wrappers ---------------------------------------------------------
    def call(self, K: float, T: float, spot: float, **kwargs: Any) -> PDEPricingResult:
        """Asian call (see :meth:`price`)."""
        return self.price(K, T, spot, "call", **kwargs)

    def put(self, K: float, T: float, spot: float, **kwargs: Any) -> PDEPricingResult:
        """Asian put (see :meth:`price`)."""
        return self.price(K, T, spot, "put", **kwargs)
