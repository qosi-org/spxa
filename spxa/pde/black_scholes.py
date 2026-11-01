r"""Black–Scholes and local-volatility PDE pricers.

The value :math:`u(S, \tau)` of a claim on an asset with dynamics
:math:`dS = (r - q) S\,dt + \sigma(S, t) S\,dW` solves

.. math::
    u_\tau = \tfrac12 \sigma(S, T - \tau)^2 S^2 u_{SS} + (r - q) S u_S - r u,
    \qquad u(S, 0) = g(S).

Discretisation choices, all verified by the test-suite:

* **Grid.** Sinh-stretched in :math:`S` on :math:`[0, S_{\max}]`, clustered at
  the strike, with strikes and barriers aligned to nodes.
  :math:`S_{\max} = \max(S_0, K)\, e^{6 \sigma \sqrt{T}}` by default.
* **Boundaries.** At :math:`S = 0` the PDE degenerates to
  :math:`u_\tau = -r u` and is applied as is (Fichera: no condition may be
  imposed). At :math:`S_{\max}` the zero-gamma condition
  :math:`u_{SS} = 0` holds for every payoff that is asymptotically linear.
* **Payoff.** Cell averages of the payoff (Pooley–Vetzal–Forsyth smoothing).
* **Time stepping.** Crank–Nicolson with Rannacher start-up, restarted after
  every discrete dividend.

Together these give clean second-order convergence of prices and grid
Greeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from spxa.pde._utils import check_option_type, check_positive, cubic_interpolate
from spxa.pde.boundary import BoundaryCondition, Dirichlet, Linear, Natural
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.american import make_lcp
from spxa.pde.payoffs import Payoff, digital, project_payoff, vanilla
from spxa.pde.solution import ExerciseBoundary, PDEPricingResult, PDESolution1D
from spxa.pde.theta import LCPStepper, ThetaSolver1D

Volatility = Union[float, Callable[[np.ndarray, float], np.ndarray]]


@dataclass(frozen=True)
class DiscreteDividend:
    r"""A dividend paid at calendar time ``time``.

    On the ex-dividend date the spot drops by ``amount + proportional * S``
    (floored at zero), so that by no-arbitrage
    :math:`V(S, t_d^-) = V(\max(S(1 - \delta) - D, 0), t_d^+)`.
    """

    time: float
    amount: float = 0.0
    proportional: float = 0.0

    def __post_init__(self) -> None:
        if self.amount < 0.0 or not 0.0 <= self.proportional < 1.0:
            raise ValueError("need amount >= 0 and 0 <= proportional < 1")

    def apply(self, s: np.ndarray) -> np.ndarray:
        """Post-dividend spot as a function of the pre-dividend spot."""
        return np.maximum(s * (1.0 - self.proportional) - self.amount, 0.0)


class BlackScholesPDE:
    r"""Finite-difference pricer for the Black–Scholes and local-volatility models.

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float or callable
        Constant volatility or local volatility ``sigma(S, t)`` with ``t``
        calendar time (vectorised in ``S``).
    q : float
        Continuous dividend yield.

    Examples
    --------
    >>> from spxa.pde import BlackScholesPDE
    >>> res = BlackScholesPDE(r=0.05, sigma=0.2).european(K=100, T=1.0, spot=100)
    >>> round(res.price, 3)
    10.451
    """

    def __init__(self, r: float, sigma: Volatility, q: float = 0.0) -> None:
        self.r = float(r)
        self.q = float(q)
        if not callable(sigma):
            check_positive("sigma", sigma)
            sigma = float(sigma)
        self.sigma = sigma

    # ------------------------------------------------------------------ model pieces
    @property
    def is_local_vol(self) -> bool:
        """True when the volatility is a function of ``(S, t)``."""
        return callable(self.sigma)

    def local_vol(self, s: np.ndarray, t: float) -> np.ndarray:
        """Volatility on the nodes ``s`` at calendar time ``t``."""
        s = np.asarray(s, dtype=float)
        if callable(self.sigma):
            return np.broadcast_to(np.asarray(self.sigma(s, t), dtype=float), s.shape)
        return np.full(s.shape, self.sigma)

    def _sigma_ref(self, spot: float, T: float) -> float:
        if not callable(self.sigma):
            return float(self.sigma)
        pts = spot * np.array([0.5, 0.8, 1.0, 1.25, 2.0])
        vals = [np.max(self.local_vol(pts, t)) for t in (0.0, 0.5 * T, T)]
        return float(max(max(vals), 1e-3))

    def pde(self, T: float) -> LinearPDE1D:
        """The backward PDE in ``S`` for maturity ``T``."""
        r, q = self.r, self.q
        if callable(self.sigma):

            def diffusion(s: np.ndarray, tau: float) -> np.ndarray:
                return 0.5 * self.local_vol(s, T - tau) ** 2 * s**2

            return LinearPDE1D(
                diffusion=diffusion,
                convection=lambda s, tau: (r - q) * s,
                reaction=-r,
                time_homogeneous=False,
            )
        sig = float(self.sigma)
        return LinearPDE1D(
            diffusion=lambda s, tau: 0.5 * sig**2 * s**2,
            convection=lambda s, tau: (r - q) * s,
            reaction=-r,
            time_homogeneous=True,
        )

    def default_s_max(self, spot: float, strike: float, T: float, n_std: float = 6.0) -> float:
        """Upper truncation ``max(S0, K) exp(n_std * sigma * sqrt(T))``."""
        sig = self._sigma_ref(spot, T)
        return float(max(spot, strike) * math.exp(n_std * sig * math.sqrt(T)))

    # ------------------------------------------------------------------ engine
    def solve(
        self,
        payoff: Payoff,
        T: float,
        *,
        center: float,
        n_space: int = 401,
        n_time: int = 200,
        s_min: float = 0.0,
        s_max: Optional[float] = None,
        left: Optional[BoundaryCondition] = None,
        right: Optional[BoundaryCondition] = None,
        align: Sequence[float] = (),
        events: Optional[dict[float, Any]] = None,
        event_times: Sequence[float] = (),
        dividends: Sequence[DiscreteDividend] = (),
        lcp: Optional[LCPStepper] = None,
        obstacle: Any = None,
        store_history: bool = False,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        smoothing: str = "average",
        grid: str = "sinh",
        concentration: float = 1.0,
        convection_scheme: str = "auto",
        label: str = "Black-Scholes PDE",
    ) -> PDESolution1D:
        r"""Solve the pricing PDE for an arbitrary payoff.

        Parameters
        ----------
        payoff : Payoff
            Terminal payoff (its kinks are aligned to grid nodes).
        T : float
            Maturity.
        center : float
            Grid concentration point (usually the strike).
        n_space, n_time : int
            Number of spatial nodes and time steps.
        s_min, s_max : float
            Computational domain (``s_max`` defaults to
            :meth:`default_s_max`).
        left, right : BoundaryCondition, optional
            Boundary closures (default: natural at ``S = 0``, zero gamma at
            ``s_max``; a positive ``s_min`` requires an explicit ``left``).
        align : sequence of float
            Extra points (barriers) to place exactly on nodes.
        events : dict, optional
            Additional event hooks ``{tau: hook}``; their times must be in
            ``event_times`` (calendar time).
        event_times : sequence of float
            Calendar times to insert into the time grid.
        dividends : sequence of DiscreteDividend
            Discrete dividends (jump conditions at the ex-dates).
        lcp, obstacle :
            Free-boundary solver and obstacle for American-style problems.
        smoothing : {'average', 'none'}
            Payoff projection.
        grid : {'sinh', 'uniform'}
            Spatial grid type; ``concentration`` sets the sinh scale
            ``alpha = concentration * center``.
        """
        T = check_positive("T", T)
        s_max = self.default_s_max(center, center, T) if s_max is None else float(s_max)
        if not s_min < center < s_max:
            raise ValueError("center must lie strictly inside (s_min, s_max)")
        if s_min > 0.0 and left is None:
            raise ValueError("a positive s_min requires an explicit left boundary condition")
        if grid == "sinh":
            g = Grid1D.sinh(
                s_min,
                s_max,
                n_space,
                center=center,
                alpha=concentration * center,
                align=list(payoff.kinks) + [float(a) for a in align],
            )
        elif grid == "uniform":
            g = Grid1D.uniform(s_min, s_max, n_space)
        else:
            raise ValueError("grid must be 'sinh' or 'uniform'")
        g = g.with_points(list(payoff.kinks) + [float(a) for a in align])
        left = Natural() if left is None else left
        right = Linear() if right is None else right
        op = DiscreteOperator1D(self.pde(T), g, left, right, convection_scheme)
        u0 = project_payoff(payoff, g, method=smoothing)
        all_times = sorted(set(float(t) for t in event_times) | {d.time for d in dividends})
        tg = TimeGrid.from_calendar_events(T, n_time, all_times)
        hooks: dict[float, list[Any]] = {}
        for tau, hk in (events or {}).items():
            hooks.setdefault(float(tau), []).extend(hk if isinstance(hk, list) else [hk])
        for d in dividends:
            if not 0.0 <= d.time < T:
                raise ValueError("dividend dates must lie in [0, T)")
            hooks.setdefault(T - d.time, []).append(_dividend_hook(d))
        # match hook times to the exact grid times
        ev: dict[float, list[Any]] = {}
        for tau, hk in hooks.items():
            j = int(np.argmin(np.abs(tg.times - tau)))
            ev.setdefault(float(tg.times[j]), []).extend(hk)
        solver = ThetaSolver1D(op, theta=theta, rannacher_steps=rannacher_steps)
        res = solver.solve(
            u0,
            tg,
            events=ev or None,
            lcp=lcp,
            obstacle=obstacle,
            store_history=store_history,
            callback=callback,
        )
        info = {
            "pde": "u_tau = 1/2 sigma^2 S^2 u_SS + (r - q) S u_S - r u",
            "boundary": f"left {left.describe()}, right {right.describe()}",
            "smoothing": f"{smoothing} payoff projection",
        }
        return PDESolution1D.from_theta_result(res, variable="S", label=label, info=info)

    # ------------------------------------------------------------------ results
    def _result(
        self,
        sol: PDESolution1D,
        spot: float,
        method: str,
        bump: Optional[Callable[["BlackScholesPDE"], float]] = None,
        exercise_boundary: Any = None,
    ) -> PDEPricingResult:
        g = sol.greeks(spot)
        vega = rho = None
        if bump is not None:
            h_s, h_r = 1e-3, 1e-4
            vega = (bump(self._bumped(dsigma=h_s)) - bump(self._bumped(dsigma=-h_s))) / (2 * h_s)
            rho = (bump(self._bumped(dr=h_r)) - bump(self._bumped(dr=-h_r))) / (2 * h_r)
        return PDEPricingResult(
            price=g["value"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g.get("theta", float("nan")),
            spot=float(spot),
            solution=sol,
            vega=vega,
            rho=rho,
            exercise_boundary=exercise_boundary,
            method=method,
        )

    def _bumped(self, dsigma: float = 0.0, dr: float = 0.0) -> "BlackScholesPDE":
        if callable(self.sigma):
            base = self.sigma

            def sig(s: np.ndarray, t: float) -> np.ndarray:
                return np.asarray(base(s, t), dtype=float) + dsigma

            return BlackScholesPDE(self.r + dr, sig, self.q)
        return BlackScholesPDE(self.r + dr, float(self.sigma) + dsigma, self.q)

    # ------------------------------------------------------------------ products
    def european(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        bump_greeks: bool = False,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """European call or put.

        Extra keyword arguments are forwarded to :meth:`solve` (resolution,
        grid, dividends, ...). ``bump_greeks=True`` adds vega and rho by
        central bump-and-revalue (four extra solves).
        """
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"European {opt} K={K:g} T={T:g}")
        sol = self.solve(vanilla(K, opt), T, center=K, **kwargs)
        bump = None
        if bump_greeks:

            def bump(model: "BlackScholesPDE") -> float:
                return float(model.solve(vanilla(K, opt), T, center=K, **kwargs).value(spot))

        return self._result(sol, spot, f"Black-Scholes PDE European {opt}", bump)

    def digital(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        cash: float = 1.0,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """Cash-or-nothing digital option."""
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"Digital {opt} K={K:g} T={T:g}")
        sol = self.solve(digital(K, opt, cash), T, center=K, **kwargs)
        return self._result(sol, spot, f"Black-Scholes PDE digital {opt}")

    def american(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "put",
        method: str = "penalty",
        bump_greeks: bool = False,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """American call or put (free-boundary problem).

        Parameters
        ----------
        method : {'penalty', 'psor', 'brennan_schwartz', 'ikonen_toivanen'}
            LCP solver (see :mod:`spxa.pde.american`).

        The early-exercise boundary :math:`S^*(\tau)` is recorded at every
        time level and returned in ``result.exercise_boundary``.
        """
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"American {opt} K={K:g} T={T:g}")
        payoff = vanilla(K, opt)
        side = "left" if opt == "put" else "right"

        def run(model: "BlackScholesPDE", record: bool) -> tuple[PDESolution1D, Any]:
            lcp = make_lcp(method, side=side)
            nodes: list[np.ndarray] = []  # the grid is built inside solve()
            taus: list[float] = []
            locs: list[float] = []

            def obstacle(s: np.ndarray, tau: float) -> np.ndarray:
                if not nodes:
                    nodes.append(s)
                return payoff(s)

            def callback(k: int, tau: float, u: np.ndarray) -> None:
                if record and nodes:
                    taus.append(tau)
                    locs.append(_boundary_location(nodes[0], u, payoff, side, K))

            sol = model.solve(
                payoff,
                T,
                center=K,
                lcp=lcp,  # type: ignore[arg-type]
                obstacle=obstacle,
                callback=callback,
                **kwargs,
            )
            sol.info["lcp"] = lcp.describe()  # type: ignore[attr-defined]
            if record:
                sol.exercise_boundary = ExerciseBoundary(np.array(taus), np.array(locs), side)
            return sol, sol.exercise_boundary

        sol, eb = run(self, True)
        bump = None
        if bump_greeks:

            def bump(model: "BlackScholesPDE") -> float:
                return float(run(model, False)[0].value(spot))

        return self._result(sol, spot, f"Black-Scholes PDE American {opt} ({method})", bump, eb)

    def bermudan(
        self,
        K: float,
        T: float,
        spot: float,
        exercise_times: Sequence[float],
        option: str = "put",
        **kwargs: Any,
    ) -> PDEPricingResult:
        r"""Bermudan call or put exercisable at the given calendar times.

        Exercise is applied as the jump condition :math:`u \leftarrow \max(u, g)`
        at each date in ``[0, T)``; exercise at ``T`` is the payoff itself.
        """
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"Bermudan {opt} K={K:g} T={T:g}")
        payoff = vanilla(K, opt)
        times = sorted(set(float(t) for t in exercise_times if float(t) < T))
        if any(t < 0.0 for t in times):
            raise ValueError("exercise times must be non-negative")

        def exercise(u: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
            return np.maximum(u, payoff(s))

        events = {T - t: exercise for t in times}
        sol = self.solve(payoff, T, center=K, events=events, event_times=times, **kwargs)
        sol.info["exercise_dates"] = len(times)
        return self._result(sol, spot, f"Black-Scholes PDE Bermudan {opt} ({len(times)} dates)")

    # ------------------------------------------------------------------ barriers
    def barrier(
        self,
        K: float,
        T: float,
        spot: float,
        barrier: float,
        barrier_type: str = "down-and-out",
        option: str = "call",
        rebate: float = 0.0,
        rebate_timing: str = "hit",
        american: bool = False,
        method: str = "penalty",
        **kwargs: Any,
    ) -> PDEPricingResult:
        """Continuously monitored single-barrier option.

        Knock-outs are solved on the domain bounded by the barrier with the
        Dirichlet condition ``u = rebate`` (``rebate_timing='hit'``) or
        ``u = rebate * exp(-r tau)`` (``'expiry'``) at the barrier. Knock-ins
        use in/out parity on the same grid: the vanilla minus the knock-out,
        plus the rebate times the price of a claim paying one at expiry if
        the barrier is never hit. American knock-outs are free-boundary
        problems on the knock-out domain.
        """
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        H = check_positive("barrier", barrier)
        bt = str(barrier_type).lower()
        if bt not in ("down-and-out", "down-and-in", "up-and-out", "up-and-in"):
            raise ValueError("unknown barrier_type")
        if rebate_timing not in ("hit", "expiry"):
            raise ValueError("rebate_timing must be 'hit' or 'expiry'")
        down = bt.startswith("down")
        if (down and spot <= H) or (not down and spot >= H):
            raise ValueError("spot has already breached the barrier")
        payoff = vanilla(K, opt)
        if bt.endswith("out"):
            return self._knock_out(
                payoff, T, spot, H, down, rebate, rebate_timing, american, method, K, kwargs
            )
        if american:
            raise NotImplementedError("American knock-in options are not supported")
        base = dict(kwargs)
        out = self._knock_out(payoff, T, spot, H, down, 0.0, "hit", False, method, K, base)
        van = self.european(K, T, spot, opt, **dict(kwargs))
        price = van.price - out.price
        delta = van.delta - out.delta
        gamma = van.gamma - out.gamma
        theta = van.theta - out.theta
        if rebate:
            one = Payoff(lambda s: np.ones_like(s), (), "unit")
            nohit = self._knock_out(one, T, spot, H, down, 0.0, "hit", False, method, K, base)
            price += rebate * nohit.price
            delta += rebate * nohit.delta
            gamma += rebate * nohit.gamma
            theta += rebate * nohit.theta
        out.solution.info["method"] = "in/out parity on the knock-out grid"
        return PDEPricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            spot=spot,
            solution=out.solution,
            method=f"Black-Scholes PDE {bt} {opt} (parity)",
        )

    def _knock_out(
        self,
        payoff: Payoff,
        T: float,
        spot: float,
        H: float,
        down: bool,
        rebate: float,
        rebate_timing: str,
        american: bool,
        method: str,
        K: float,
        kwargs: dict[str, Any],
    ) -> PDEPricingResult:
        r = self.r
        R = float(rebate)
        bc = (
            Dirichlet(R)
            if rebate_timing == "hit"
            else Dirichlet(lambda tau: R * math.exp(-r * tau))
        )
        kw = dict(kwargs)
        s_max = kw.pop("s_max", None)  # the barrier bounds the domain of up-and-outs
        if down:
            lo = H
            hi = self.default_s_max(spot, max(K, H), T) if s_max is None else float(s_max)
            left, right = bc, Linear()
        else:
            lo, hi = 0.0, H
            left, right = Natural(), bc
        center = K if lo < K < hi else spot
        kw.setdefault("label", f"knock-out barrier H={H:g}")
        extra: dict[str, Any] = {}
        if american:
            lcp = make_lcp(method, side="left" if payoff(np.array([0.0]))[0] > 0 else "right")
            extra = dict(lcp=lcp, obstacle=lambda s, tau: payoff(s))
        sol = self.solve(
            payoff, T, center=center, s_min=lo, s_max=hi, left=left, right=right, **extra, **kw
        )
        sol.info["boundary"] = f"barrier Dirichlet u = rebate ({rebate_timing}) at H={H:g}"
        return self._result(sol, spot, "Black-Scholes PDE knock-out")

    def double_barrier(
        self,
        K: float,
        T: float,
        spot: float,
        lower: float,
        upper: float,
        option: str = "call",
        rebate: float = 0.0,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """Continuously monitored double knock-out (rebate paid at the hit)."""
        opt = check_option_type(option)
        if not 0.0 < lower < spot < upper:
            raise ValueError("need 0 < lower < spot < upper")
        payoff = vanilla(K, opt)
        center = K if lower < K < upper else spot
        kwargs.setdefault("label", f"double knock-out [{lower:g}, {upper:g}]")
        sol = self.solve(
            payoff,
            T,
            center=center,
            s_min=lower,
            s_max=upper,
            left=Dirichlet(rebate),
            right=Dirichlet(rebate),
            **kwargs,
        )
        return self._result(sol, spot, f"Black-Scholes PDE double knock-out {opt}")

    def discrete_barrier(
        self,
        K: float,
        T: float,
        spot: float,
        barrier: float,
        monitoring_times: Sequence[float],
        barrier_type: str = "down-and-out",
        option: str = "call",
        rebate: float = 0.0,
        **kwargs: Any,
    ) -> PDEPricingResult:
        r"""Discretely monitored knock-out option.

        At each monitoring date :math:`t_i < T` the jump condition
        :math:`u \leftarrow R` on the knocked-out region is applied (the
        rebate is paid at the monitoring date); a monitoring date at ``T``
        is part of the payoff. The node on the barrier receives the average
        of both one-sided values, the nodal analogue of the cell average of
        the discontinuity.
        """
        opt = check_option_type(option)
        H = check_positive("barrier", barrier)
        bt = str(barrier_type).lower()
        if bt not in ("down-and-out", "up-and-out"):
            raise ValueError("discrete monitoring is supported for knock-out barriers")
        down = bt == "down-and-out"
        R = float(rebate)

        def knocked(s: np.ndarray) -> np.ndarray:
            return s < H if down else s > H

        def monitor(u: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
            out = np.where(knocked(s), R, u)
            on = s == H
            out[on] = 0.5 * (u[on] + R)
            return out

        base = vanilla(K, opt)
        at_T = any(abs(float(t) - T) < 1e-12 for t in monitoring_times)

        def pay(s: np.ndarray) -> np.ndarray:
            v = base(s)
            if at_T:
                v = np.where(knocked(s), R, v)
            return v

        payoff = Payoff(pay, tuple(sorted({K, H})), f"discrete {bt} payoff")
        times = sorted(set(float(t) for t in monitoring_times if 0.0 <= float(t) < T))
        events = {T - t: monitor for t in times}
        kwargs.setdefault("s_max", self.default_s_max(spot, max(K, H), T))
        kwargs.setdefault("label", f"discrete {bt} H={H:g}, {len(times)} dates")
        sol = self.solve(payoff, T, center=K, align=[H], events=events, event_times=times, **kwargs)
        return self._result(sol, spot, f"Black-Scholes PDE discrete {bt} {opt}")


def _boundary_location(
    s: np.ndarray, u: np.ndarray, payoff: Payoff, side: str, strike: float
) -> float:
    """Edge of the region where ``u`` equals the exercise value (in-the-money only)."""
    g = payoff(s)
    tol = 1e-7 * max(1.0, strike)
    ex = (u - g <= tol) & (g > 0.0)
    if not np.any(ex):
        return float("nan")
    return float(np.max(s[ex]) if side == "left" else np.min(s[ex]))


def _dividend_hook(div: DiscreteDividend) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    def hook(u: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
        return cubic_interpolate(s, u, div.apply(s))

    return hook
