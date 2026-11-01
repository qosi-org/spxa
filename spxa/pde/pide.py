r"""
Partial integro-differential equations for exponential Lévy models.

Under :math:`S_t = S_0 e^{(r - q)t + X_t}` with :math:`X` a Lévy process with
Gaussian variance :math:`\sigma^2` and Lévy density :math:`\nu`, option
prices :math:`u(x, \tau)` in :math:`x = \log S` solve

.. math::
    u_\tau = \tfrac12\sigma^2 u_{xx} + \bigl(r - q - \tfrac12\sigma^2 - \omega\bigr)u_x
             - r u + \int \bigl(u(x + y) - u(x)\bigr)\,\nu(dy),
    \qquad \omega = \int (e^y - 1)\,\nu(dy).

Discretisation (uniform grid in :math:`x`, spacing :math:`h`):

* **Jumps** with :math:`|y| \ge h/2` are integrated against the hat functions
  of the grid, giving weights :math:`w_k = \int \nu(y)\varphi_k(y)\,dy` so
  that :math:`\int u(x_i + y)\nu(dy) \approx \sum_k w_k u_{i+k}` is exact for
  piecewise-linear :math:`u` (second order). Values outside the domain are
  taken from the payoff's asymptotics, so the convolution needs no
  truncation of the solution.
* **Small jumps** :math:`|y| < h/2` are replaced by a diffusion of variance
  :math:`\sigma_\varepsilon^2 = \int_{|y|<h/2} y^2\nu(dy)` (Asmussen–Rosiński),
  which matters for infinite-activity models (VG, CGMY, NIG).
* **Moment matching.** Linear interpolation over-weights :math:`y^2` on every
  cell by :math:`\epsilon_h = \sum_k w_k (kh)^2 - \int_{|y|\ge h/2} y^2\nu(dy)
  = O(h^{2-Y})`; for Blumenthal–Getoor index :math:`Y \ge 1` (NIG, CGMY) this
  would make the scheme first order. The excess is removed from the local
  diffusion, so the discrete operator reproduces the zeroth, first and second
  moments of :math:`\nu` and stays second order.
* With :math:`\tilde\lambda = \sum_k w_k` and
  :math:`\tilde\omega = \sum_k w_k(e^{kh} - 1)` the local part is
  :math:`a = \tfrac12(\sigma^2 + \sigma_\varepsilon^2 - \epsilon_h)`,
  :math:`b = r - q - a - \tilde\omega`, :math:`c = -(r + \tilde\lambda)`; the
  discrete jump operator then maps :math:`e^x` exactly as the continuous one
  (the forward is preserved). :math:`a` may be slightly negative for pure-jump
  models; the nearest-neighbour jump weights keep the stencil well posed.
* **Time stepping** is Crank–Nicolson (with Rannacher start-up). The local
  operator and the nearest-neighbour jumps are implicit; the remaining jump
  integral is iterated to a fixed point at every step (d'Halluin, Forsyth &
  Vetzal, 2005), which contracts with factor at most
  :math:`\theta\Delta\tau\tilde\lambda / (1 + \theta\Delta\tau(r + \tilde\lambda))`.
* **American** constraints are enforced inside the same iteration with any of
  the LCP solvers (the stateful Ikonen–Toivanen splitting wraps the whole
  fixed-point solve and is applied once per step).

References: Cont & Voltchkova (2005), *SIAM J. Numer. Anal.* 43(4);
d'Halluin, Forsyth & Vetzal (2005), *IMA J. Numer. Anal.* 25(1);
Asmussen & Rosiński (2001), *J. Appl. Probab.* 38(2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.american import IkonenToivanenLCP, make_lcp
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.payoffs import Payoff, project_payoff, vanilla
from spxa.pde.solution import ExerciseBoundary, PDEPricingResult, PDESolution1D

__all__ = ["LevyJumpOperator", "LevyPIDE"]

Density = Callable[[np.ndarray], np.ndarray]
FarField = Callable[[np.ndarray, float], np.ndarray]

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _gauss(f: Density, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weighted values of ``f`` on each ``[a_i, b_i]``."""
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    y = mid[:, None] + half[:, None] * _GL_X[None, :]
    fy = np.asarray(f(y.ravel()), dtype=float).reshape(y.shape)
    return y, fy * _GL_W[None, :] * half[:, None]


@dataclass
class LevyJumpOperator:
    r"""Discrete jump integral on a uniform grid of spacing ``h``.

    Parameters
    ----------
    density : callable
        Lévy density :math:`\nu(y)` (vectorised).
    h : float
        Grid spacing in :math:`x = \log S`.
    max_jump : float
        Jumps larger than this in absolute value are neglected; the
        neglected mass is reported in :attr:`tail_mass`.
    """

    density: Density
    h: float
    max_jump: float
    offsets: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    small_jump_variance: float = field(init=False)
    interpolation_variance: float = field(init=False)
    tail_mass: float = field(init=False)

    def __post_init__(self) -> None:
        h = check_positive("h", self.h)
        m = max(1, int(math.ceil(self.max_jump / h)))
        self.offsets = np.arange(-m, m + 1)
        w = np.zeros(2 * m + 1)
        # interval [j h, (j + 1) h] for j = -m .. m-1, cut at |y| < h/2
        j = np.arange(-m, m)
        a = j * h
        b = (j + 1) * h
        a = np.where(j == 0, 0.5 * h, a)
        b = np.where(j == -1, -0.5 * h, b)
        y, fw = _gauss(self.density, a, b)
        left_node = (j * h)[:, None]
        frac = (y - left_node) / h  # hat of node j+1 rises linearly on the interval
        w_lo = np.sum(fw * (1.0 - frac), axis=1)
        w_hi = np.sum(fw * frac, axis=1)
        np.add.at(w, j + m, w_lo)
        np.add.at(w, j + m + 1, w_hi)
        self.weights = w
        # linear interpolation over-weights y^2 on every cell; the excess is
        # removed from the local diffusion so the second moment is exact
        m2_cells = float(np.sum(fw * y * y))
        self.interpolation_variance = float(np.dot(w, (self.offsets * h) ** 2)) - m2_cells
        self.small_jump_variance = self._small_jump_variance(0.5 * h)
        self.tail_mass = self._tail_mass(m * h)

    def _integral(self, f: Callable[[float], float], a: float, b: float) -> float:
        val, _ = quad(f, a, b, limit=200, epsabs=1e-14, epsrel=1e-10)
        return float(val)

    def _point(self, y: float) -> float:
        return float(np.asarray(self.density(np.array([y])), dtype=float)[0])

    def _small_jump_variance(self, eps: float) -> float:
        def f(y: float) -> float:
            return y * y * self._point(y)

        return self._integral(f, -eps, 0.0) + self._integral(f, 0.0, eps)

    def _tail_mass(self, y_max: float) -> float:
        return self._integral(self._point, -np.inf, -y_max) + self._integral(
            self._point, y_max, np.inf
        )

    @property
    def local_variance(self) -> float:
        r"""Variance the local operator must carry for the jump part.

        :math:`\sigma_\varepsilon^2 - \bigl(\sum_k w_k (kh)^2 - \int_{|y| \ge h/2} y^2\nu(dy)\bigr)`:
        the small jumps minus the excess second moment created by the linear
        interpolation. With it the discrete operator reproduces the zeroth,
        first and second moments of :math:`\nu`, which keeps the scheme
        second-order accurate for infinite-variation measures.
        """
        return self.small_jump_variance - self.interpolation_variance

    @property
    def intensity(self) -> float:
        r""":math:`\tilde\lambda = \sum_k w_k` (jumps with :math:`|y| \ge h/2`)."""
        return float(np.sum(self.weights))

    @property
    def compensator(self) -> float:
        r""":math:`\tilde\omega = \sum_k w_k (e^{kh} - 1)`."""
        return float(np.dot(self.weights, np.expm1(self.offsets * self.h)))

    def apply(
        self,
        u: np.ndarray,
        x: np.ndarray,
        tau: float,
        far_left: FarField,
        far_right: FarField,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r""":math:`\sum_k w_k u(x_i + kh)` with far-field values outside the grid.

        ``weights`` overrides :attr:`weights` (same offsets), e.g. to exclude
        the near field that a solver treats implicitly.
        """
        m = int(self.offsets[-1])
        h = self.h
        xl = x[0] - h * np.arange(m, 0, -1)
        xr = x[-1] + h * np.arange(1, m + 1)
        ext = np.concatenate([far_left(xl, tau), u, far_right(xr, tau)])
        # sum_k w_k ext[i + m + k] is a correlation: convolve with reversed weights
        wts = self.weights if weights is None else weights
        return fftconvolve(ext, wts[::-1], mode="valid")


class LevyPIDE:
    r"""European and American options under exponential Lévy dynamics.

    Parameters
    ----------
    r, q : float
        Rate and dividend yield.
    sigma : float
        Gaussian volatility (may be ``0`` for pure-jump models).
    density : callable
        Lévy density :math:`\nu(y)` of the log-price jumps.
    name : str
        Label used in results.
    """

    def __init__(
        self, r: float, q: float, sigma: float, density: Density, name: str = "Levy"
    ) -> None:
        self.r = float(r)
        self.q = float(q)
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        self.sigma = float(sigma)
        self.density = density
        self.name = name

    def __repr__(self) -> str:
        return f"LevyPIDE({self.name}, r={self.r:g}, q={self.q:g}, sigma={self.sigma:g})"

    @classmethod
    def from_process(cls, process: Any, r: float, q: float = 0.0) -> "LevyPIDE":
        """Build from a zoo Lévy process (uses its Gaussian variance and Lévy density)."""
        trip = process.triplet
        nu = trip.nu
        return cls(
            r,
            q,
            math.sqrt(float(trip.sigma_sq)),
            lambda y: np.asarray(nu.density(np.asarray(y, dtype=float)), dtype=float),
            name=type(process).__name__,
        )

    # ------------------------------------------------------------------ helpers
    def _second_moment(self) -> float:
        def f(y: float) -> float:
            return y * y * float(np.asarray(self.density(np.array([y])))[0])

        tot = 0.0
        for a, b in ((-np.inf, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, np.inf)):
            tot += quad(f, a, b, limit=200)[0]
        return tot

    def default_half_width(self, T: float) -> float:
        """Half-width of the log-price domain: 6 standard deviations, at least 1.5."""
        var = (self.sigma**2 + self._second_moment()) * T
        return max(6.0 * math.sqrt(var), 1.5)

    def _grid(self, spot: float, K: float, T: float, n: int, half_width: Optional[float]) -> Grid1D:
        L = self.default_half_width(T) if half_width is None else float(half_width)
        x0, k = math.log(spot), math.log(K)
        h = 2.0 * L / (n - 1)
        gap = abs(k - x0)
        if gap > 1e-12:
            # shrink h slightly so that both log K and log S0 are nodes
            h = gap / max(1, math.ceil(gap / h - 1e-9))
        # shift the uniform grid so that log K is a node
        j = round((k - (x0 - L)) / h)
        lo = k - j * h
        return Grid1D(lo + h * np.arange(n))

    def far_field(self, K: float, option: str, american: bool) -> tuple[FarField, FarField]:
        """Asymptotic values used outside the domain and as Dirichlet data."""
        r, q = self.r, self.q
        if option == "call":

            def left(x: np.ndarray, tau: float) -> np.ndarray:
                return np.zeros_like(x)

            def right(x: np.ndarray, tau: float) -> np.ndarray:
                if american:
                    return np.maximum(np.exp(x) - K, np.exp(x - q * tau) - K * math.exp(-r * tau))
                return np.exp(x - q * tau) - K * math.exp(-r * tau)

        else:

            def left(x: np.ndarray, tau: float) -> np.ndarray:
                if american:
                    return K - np.exp(x)
                return K * math.exp(-r * tau) - np.exp(x - q * tau)

            def right(x: np.ndarray, tau: float) -> np.ndarray:
                return np.zeros_like(x)

        return left, right

    def solve(
        self,
        payoff: Payoff,
        T: float,
        grid: Grid1D,
        far_left: FarField,
        far_right: FarField,
        n_time: int = 200,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        lcp: Any = None,
        obstacle: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
        max_jump: Optional[float] = None,
        convection_scheme: str = "central",
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> PDESolution1D:
        """March the PIDE on a uniform log-price ``grid``.

        Parameters
        ----------
        payoff : Payoff
            Terminal payoff as a function of ``S`` (projected with its kinks).
        T : float
            Maturity.
        grid : Grid1D
            Uniform grid in ``x = log S``.
        far_left, far_right : callable
            ``(x, tau) -> values`` used as Dirichlet data at the ends and for
            the jump integral beyond them.
        n_time, theta, rannacher_steps :
            Time grid size, theta (0.5 is Crank-Nicolson) and the number of
            fully implicit half-step pairs at the start.
        lcp, obstacle :
            Optional complementarity solver and obstacle ``(x, tau) -> g``
            for early exercise.
        tol, max_iter :
            Stopping rule of the jump fixed-point iteration (relative to
            ``max(1, max|u|)``).
        max_jump : float, optional
            Largest jump retained (default: the domain width).
        convection_scheme : {'central', 'upwind', 'auto'}
            ``'auto'`` upwinds only when the full stencil (including the
            nearest-neighbour jump weights) would have a negative
            off-diagonal. Central differencing is the default: it keeps the
            scheme second order, which matters for pure-jump models whose
            local diffusion is tiny.
        callback : callable, optional
            Called as ``callback(k, tau_k, u_k)`` after every time step.
        """
        from scipy.sparse import csr_matrix, identity
        from scipy.sparse.linalg import splu

        x = grid.nodes
        h = float(x[1] - x[0])
        if not np.allclose(np.diff(x), h, rtol=1e-9, atol=0.0):
            raise ValueError("the PIDE grid must be uniform in log S")
        jump = LevyJumpOperator(
            self.density, h, max_jump if max_jump is not None else (x[-1] - x[0])
        )
        lam, om = jump.intensity, jump.compensator
        a = 0.5 * (self.sigma**2 + jump.local_variance)
        b = self.r - self.q - a - om
        c = -(self.r + lam)
        n = grid.n
        m = int(jump.offsets[-1])
        w = jump.weights
        # nearest-neighbour jumps (where infinite-activity mass concentrates) are
        # treated implicitly together with the local operator
        wm, w0, wp = float(w[m - 1]), float(w[m]), float(w[m + 1])
        w_far = w.copy()
        w_far[m - 1 : m + 2] = 0.0
        lower = a / h**2 - b / (2.0 * h) + wm
        upper = a / h**2 + b / (2.0 * h) + wp
        diag = -2.0 * a / h**2 + c + w0
        if convection_scheme not in ("central", "upwind", "auto"):
            raise ValueError("convection_scheme must be 'central', 'upwind' or 'auto'")
        scheme = convection_scheme
        if scheme == "auto":
            # central differencing whenever the full stencil has non-negative
            # off-diagonals (d'Halluin, Forsyth & Vetzal, 2005)
            scheme = "central" if min(lower, upper) >= 0.0 else "upwind"
        if scheme == "upwind":
            lower = a / h**2 + max(-b, 0.0) / h + wm
            upper = a / h**2 + max(b, 0.0) / h + wp
            diag = -2.0 * a / h**2 - abs(b) / h + c + w0
        interior = np.arange(1, n - 1)
        L = csr_matrix(
            (
                np.concatenate(
                    [np.full(n - 2, lower), np.full(n - 2, diag), np.full(n - 2, upper)]
                ),
                (
                    np.concatenate([interior, interior, interior]),
                    np.concatenate([interior - 1, interior, interior + 1]),
                ),
            ),
            shape=(n, n),
        )
        rows = np.array([0, n - 1])
        free = np.ones(n, dtype=bool)
        free[rows] = False
        E = csr_matrix((np.ones(2), (rows, rows)), shape=(n, n))
        cache: dict[tuple[float, float], Any] = {}

        def system(dt: float, th: float) -> tuple[Any, Any]:
            key = (round(dt, 15), th)
            if key not in cache:
                M = (identity(n, format="csr") - th * dt * L).tolil()
                M[0, :] = 0.0
                M[n - 1, :] = 0.0
                M = (M.tocsr() + E).tocsr()
                cache[key] = (M, splu(M.tocsc()))
            return cache[key]

        def J(u: np.ndarray, tau: float) -> np.ndarray:
            out = np.zeros(n)
            out[1:-1] = jump.apply(u, x, tau, far_left, far_right, weights=w_far)[1:-1]
            return out

        def boundary_values(tau: float) -> tuple[float, float]:
            return (
                float(far_left(np.array([x[0]]), tau)[0]),
                float(far_right(np.array([x[-1]]), tau)[0]),
            )

        n_iter_total = 0
        if lcp is not None:
            lcp.reset(n)

        def fixed_point(
            inner: Callable[[np.ndarray, np.ndarray], np.ndarray],
            base: np.ndarray,
            guess: np.ndarray,
        ) -> np.ndarray:
            nonlocal n_iter_total
            v = guess
            for _ in range(max_iter):
                rhs = base + state["th_dt"] * J(v, state["t1"])
                rhs[0], rhs[-1] = state["g"]
                v_new = inner(rhs, v)
                n_iter_total += 1
                err = float(np.max(np.abs(v_new - v)))
                v = v_new
                if err <= tol * max(1.0, float(np.max(np.abs(v)))):
                    return v
            raise RuntimeError("jump fixed-point iteration did not converge")

        def base_bc(base: np.ndarray) -> np.ndarray:
            out = base.copy()
            out[0], out[-1] = state["g"]
            return out

        state: dict[str, Any] = {}

        def step(u: np.ndarray, t0: float, t1: float, th: float) -> np.ndarray:
            dt = t1 - t0
            M, lu = system(dt, th)
            base = u.copy()
            if th < 1.0:
                base += (1.0 - th) * dt * (L @ u + J(u, t0))
            state.update(th_dt=th * dt, t1=t1, g=boundary_values(t1))
            g_obs = obstacle(x, t1) if obstacle is not None else None
            v = u.copy()
            if isinstance(lcp, IkonenToivanenLCP):
                # the splitting is stateful: apply it once per step around a
                # linear solve that itself iterates the jump term to convergence
                def solve_full(r: np.ndarray) -> np.ndarray:
                    return fixed_point(lambda rhs, guess: lu.solve(rhs), r, v)

                return lcp.step(M, solve_full, base_bc(base), g_obs, u, dt, free)

            def inner(rhs: np.ndarray, guess: np.ndarray) -> np.ndarray:
                if lcp is None:
                    return lu.solve(rhs)
                return lcp.step(M, lu.solve, rhs, g_obs, guess, dt, free)

            return fixed_point(inner, base, v)

        u = project_payoff(payoff, grid, transform=np.exp, inverse=np.log, method="kink")
        if obstacle is not None:
            u = np.maximum(u, obstacle(x, 0.0))
        tg = TimeGrid.uniform(T, n_time)
        times = tg.times
        prev = u
        if callback is not None:
            callback(0, 0.0, u)
        left = rannacher_steps if theta < 1.0 else 0
        for k in range(1, times.size):
            t0, t1 = float(times[k - 1]), float(times[k])
            prev = u
            if left > 0:
                mid = 0.5 * (t0 + t1)
                u = step(step(u, t0, mid, 1.0), mid, t1, 1.0)
                left -= 1
            else:
                u = step(u, t0, t1, theta)
            if callback is not None:
                callback(k, t1, u)
        return PDESolution1D(
            grid=grid,
            values=u,
            variable="log",
            previous_values=prev,
            dt_last=float(times[-1] - times[-2]),
            info={
                "model": self.name,
                "jump_intensity": lam,
                "jump_compensator": om,
                "small_jump_variance": jump.small_jump_variance,
                "neglected_tail_mass": jump.tail_mass,
                "fixed_point_iterations": n_iter_total,
                "n_steps": tg.n_steps,
                "convection_scheme": scheme,
                "local_diffusion": a,
            },
            label=f"{self.name} PIDE",
        )

    # ------------------------------------------------------------------ products
    def european(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        n_space: int = 801,
        n_time: int = 200,
        half_width: Optional[float] = None,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """European call or put; Greeks from the log-space solution."""
        opt = check_option_type(option)
        T = check_positive("T", T)
        spot = check_positive("spot", spot)
        grid = self._grid(spot, K, T, n_space, half_width)
        fl, fr = self.far_field(K, opt, american=False)
        sol = self.solve(vanilla(K, opt), T, grid, fl, fr, n_time=n_time, **kwargs)
        g = sol.greeks(spot)
        return PDEPricingResult(
            price=g["value"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g.get("theta", float("nan")),
            spot=spot,
            solution=sol,
            method=f"{self.name} PIDE European {opt}",
        )

    def american(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "put",
        method: str = "penalty",
        n_space: int = 801,
        n_time: int = 200,
        half_width: Optional[float] = None,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """American call or put: the LCP is solved inside the jump fixed point."""
        opt = check_option_type(option)
        T = check_positive("T", T)
        spot = check_positive("spot", spot)
        grid = self._grid(spot, K, T, n_space, half_width)
        payoff = vanilla(K, opt)
        fl, fr = self.far_field(K, opt, american=True)
        side = "left" if opt == "put" else "right"
        lcp = make_lcp(method, side=side)
        taus: list[float] = []
        locs: list[float] = []
        s_nodes = np.exp(grid.nodes)

        def record(k: int, tau: float, u: np.ndarray) -> None:
            g = payoff(s_nodes)
            ex = (u - g <= 1e-7 * max(1.0, K)) & (g > 0.0)
            taus.append(tau)
            if not np.any(ex):
                locs.append(float("nan"))
            else:
                locs.append(float(np.max(s_nodes[ex]) if side == "left" else np.min(s_nodes[ex])))

        sol = self.solve(
            payoff,
            T,
            grid,
            fl,
            fr,
            n_time=n_time,
            lcp=lcp,
            obstacle=lambda x, tau: payoff(np.exp(x)),
            callback=record,
            **kwargs,
        )
        eb = ExerciseBoundary(np.array(taus), np.array(locs), side)
        sol.exercise_boundary = eb
        sol.info["lcp"] = lcp.describe()
        g = sol.greeks(spot)
        return PDEPricingResult(
            price=g["value"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g.get("theta", float("nan")),
            spot=spot,
            solution=sol,
            exercise_boundary=eb,
            method=f"{self.name} PIDE American {opt} ({method})",
        )
