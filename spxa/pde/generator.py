r"""
Bridge from spxa processes to PDE solvers.

Every scalar diffusion :math:`dX = \mu(X, t)\,dt + \sigma(X, t)\,dW` has the
generator :math:`\mathcal{A} = \tfrac12\sigma^2\partial_{xx} + \mu\partial_x`.
:class:`DiffusionPDE` packages the coefficients together with a sensible
truncated domain and exposes

* :meth:`DiffusionPDE.expectation` — the Feynman–Kac backward problem
  :math:`u = \mathbb{E}[e^{-\int_0^T k(X_s, s)\,ds} g(X_T)]` with a constant,
  state-dependent or ``'state'`` (short-rate) discount :math:`k`;
* :meth:`DiffusionPDE.density` — the transition density via the
  Fokker–Planck discrete adjoint;
* :meth:`DiffusionPDE.generator` — the same generator as an
  :class:`spxa.calculus.generator.InfinitesimalGenerator`.

:func:`to_pde` builds a :class:`DiffusionPDE` from a zoo process
(Brownian motion, Ornstein–Uhlenbeck, CIR, CEV) or from explicit
coefficients. Processes whose generator is not a local second-order operator
(jumps, stochastic volatility) are handled by the dedicated solvers and
raise ``NotImplementedError`` here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from spxa.pde.boundary import BoundaryCondition, Linear, Natural
from spxa.pde.fokker_planck import DensitySolution, FokkerPlanck1D
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.payoffs import Payoff, project_payoff
from spxa.pde.solution import PDESolution1D
from spxa.pde.theta import ThetaSolver1D

__all__ = ["DiffusionPDE", "to_pde"]

Coefficient = Union[float, Callable[[np.ndarray, float], np.ndarray]]
DomainFn = Callable[[float], tuple[float, float]]


def _eval(c: Coefficient, x: np.ndarray, t: float) -> np.ndarray:
    v = c(x, t) if callable(c) else c
    return np.asarray(v, dtype=float) * np.ones_like(x, dtype=float)


@dataclass
class DiffusionPDE:
    r"""PDE view of the scalar diffusion :math:`dX = \mu\,dt + \sigma\,dW`.

    Parameters
    ----------
    drift, volatility : float or callable ``f(x, t)``
    x0 : float
        Current state.
    lower, upper : float, optional
        Bounds of the state space (e.g. ``0`` for CIR or CEV). A bound where
        the volatility vanishes is treated with the degenerate equation
        itself (no boundary data needed); otherwise zero curvature is used.
    domain : callable, optional
        ``domain(T) -> (lo, hi)`` overriding the default truncation.
    name : str
    time_homogeneous : bool, optional
        Whether callable coefficients are independent of ``t``.
    """

    drift: Coefficient
    volatility: Coefficient
    x0: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    domain: Optional[DomainFn] = None
    name: str = "diffusion"
    time_homogeneous: Optional[bool] = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_homogeneous is None:
            self.time_homogeneous = not (callable(self.drift) or callable(self.volatility))

    def __repr__(self) -> str:
        return f"DiffusionPDE({self.name}, x0={self.x0:g})"

    # ------------------------------------------------------------------ helpers
    def truncated_domain(self, T: float, n_std: float = 8.0) -> tuple[float, float]:
        """Domain used for horizon ``T``: user-supplied or a local-moment estimate."""
        if self.domain is not None:
            lo, hi = self.domain(T)
        else:
            x = np.array([self.x0])
            mu = abs(float(_eval(self.drift, x, 0.0)[0]))
            sig = abs(float(_eval(self.volatility, x, 0.0)[0]))
            width = mu * T + n_std * max(sig, 1e-8) * math.sqrt(T)
            lo, hi = self.x0 - width, self.x0 + width
        if self.lower is not None:
            lo = max(lo, self.lower)
        if self.upper is not None:
            hi = min(hi, self.upper)
        if not lo < self.x0 < hi and not (self.lower is not None and self.x0 == lo):
            raise ValueError("x0 must lie inside the truncated domain")
        return float(lo), float(hi)

    def _degenerate_at(self, x: float) -> bool:
        return abs(float(_eval(self.volatility, np.array([x]), 0.0)[0])) < 1e-12

    def _boundaries(self, lo: float, hi: float) -> tuple[BoundaryCondition, BoundaryCondition]:
        left: BoundaryCondition = (
            Natural()
            if self.lower is not None and lo == self.lower and self._degenerate_at(lo)
            else Linear()
        )
        right: BoundaryCondition = (
            Natural()
            if self.upper is not None and hi == self.upper and self._degenerate_at(hi)
            else Linear()
        )
        return left, right

    def _grid(self, T: float, n: int, kinks: tuple[float, ...] = ()) -> Grid1D:
        lo, hi = self.truncated_domain(T)
        center = self.x0 if lo < self.x0 < hi else 0.5 * (lo + hi)
        return Grid1D.sinh(lo, hi, n, center=center, alpha=0.5 * (hi - lo), align=kinks)

    def pde(self, discount: Union[float, str, Coefficient] = 0.0) -> LinearPDE1D:
        """Backward operator :math:`\\mathcal{A} - k` with discount ``k``."""
        vol = self.volatility

        def diffusion(x: np.ndarray, t: float) -> np.ndarray:
            return 0.5 * _eval(vol, x, t) ** 2

        if isinstance(discount, str):
            if discount != "state":
                raise ValueError("discount must be a number, a callable or 'state'")
            reaction: Any = lambda x, t: -x  # noqa: E731
        elif callable(discount):
            reaction = lambda x, t: -_eval(discount, x, t)  # noqa: E731
        else:
            reaction = -float(discount)
        homogeneous = bool(self.time_homogeneous) and not (
            callable(discount) and not isinstance(discount, str)
        )
        return LinearPDE1D(
            diffusion=diffusion,
            convection=self.drift
            if not callable(self.drift)
            else (lambda x, t: _eval(self.drift, x, t)),
            reaction=reaction,
            time_homogeneous=homogeneous,
        )

    # ------------------------------------------------------------------ products
    def expectation(
        self,
        payoff: Union[Payoff, Callable[[np.ndarray], np.ndarray]],
        T: float,
        discount: Union[float, str, Coefficient] = 0.0,
        n_space: int = 401,
        n_time: int = 200,
        kinks: tuple[float, ...] = (),
        rannacher_steps: int = 2,
    ) -> PDESolution1D:
        r"""Solve :math:`u_\tau = \mathcal{A}u - k u`, :math:`u(\cdot, 0) = g`.

        ``kinks`` (or ``payoff.kinks``) are aligned to grid nodes and the
        payoff is averaged over the control volumes that contain them.
        """
        k_all = tuple(kinks) + tuple(getattr(payoff, "kinks", ()))
        grid = self._grid(T, n_space, k_all)
        left, right = self._boundaries(grid.lo, grid.hi)
        op = DiscreteOperator1D(self.pde(discount), grid, left, right, "auto")
        p = payoff if isinstance(payoff, Payoff) else Payoff(payoff, tuple(kinks))
        u0 = project_payoff(p, grid, method="kink")
        res = ThetaSolver1D(op, theta=0.5, rannacher_steps=rannacher_steps).solve(
            u0, TimeGrid.uniform(T, n_time)
        )
        return PDESolution1D.from_theta_result(
            res,
            variable="S",
            label=f"E[g(X_T)] for {self.name}",
            info={"process": self.name, "discount": str(discount)},
        )

    def density(
        self, T: float, n_space: int = 401, n_time: int = 200, rannacher_steps: int = 2
    ) -> DensitySolution:
        """Transition density of :math:`X_T` given :math:`X_0 = x_0`."""
        lo, hi = self.truncated_domain(T)
        grid = Grid1D.sinh(lo, hi, n_space, center=self.x0, alpha=0.5 * (hi - lo))
        fp = FokkerPlanck1D(
            self.drift, self.volatility, grid, time_homogeneous=self.time_homogeneous
        )
        return fp.solve(self.x0, T, n_time=n_time, rannacher_steps=rannacher_steps)

    def generator(self) -> Any:
        """The generator as an :class:`~spxa.calculus.generator.InfinitesimalGenerator`."""
        from spxa.calculus.generator import InfinitesimalGenerator

        def b(x: Any) -> float:
            return float(_eval(self.drift, np.atleast_1d(np.asarray(x, dtype=float)), 0.0)[0])

        def s(x: Any) -> float:
            return float(_eval(self.volatility, np.atleast_1d(np.asarray(x, dtype=float)), 0.0)[0])

        return InfinitesimalGenerator(b, s)


# ---------------------------------------------------------------------- factory
def _ou_domain(theta: float, mu: float, sigma: float, x0: float) -> DomainFn:
    def dom(T: float) -> tuple[float, float]:
        m = mu + (x0 - mu) * math.exp(-theta * T)
        sd = sigma * math.sqrt((1 - math.exp(-2 * theta * T)) / (2 * theta))
        return min(x0, m) - 8 * sd, max(x0, m) + 8 * sd

    return dom


def to_pde(process: Any = None, **kwargs: Any) -> Any:
    r"""PDE representation of a process.

    Parameters
    ----------
    process : Process, optional
        A zoo process. Diffusions: ``BrownianMotion``,
        ``OrnsteinUhlenbeckProcess``, ``CIRProcess`` and ``CEVProcess``.
        Lévy processes with jumps (Merton, Kou, VG, CGMY, NIG, ...) map to
        the exponential-Lévy pricing PIDE when ``r`` is given.
    **kwargs
        Without ``process``: ``drift``, ``volatility``, ``x0`` and optional
        :class:`DiffusionPDE` fields. With a diffusion: overrides such as
        ``domain``. With a jump process: ``r`` and optionally ``q``.

    Returns
    -------
    DiffusionPDE or LevyPIDE

    Raises
    ------
    ValueError
        For a jump process without ``r``.
    NotImplementedError
        For processes with no registered PDE (e.g. stochastic volatility).
    """
    if process is None:
        try:
            return DiffusionPDE(**kwargs)
        except TypeError as exc:
            raise TypeError("to_pde() needs a process or drift, volatility and x0") from exc
    cls = type(process).__name__
    spec: dict[str, Any]
    if cls == "OrnsteinUhlenbeckProcess":
        th, mu, s, x0 = process.theta, process.mu, process.sigma, process.x0
        spec = dict(
            drift=lambda x, t: th * (mu - x),
            volatility=s,
            x0=x0,
            domain=_ou_domain(th, mu, s, x0),
            time_homogeneous=True,
        )
    elif cls == "CIRProcess":
        k, th, s, r0 = process.kappa, process.theta, process.sigma, process.r0

        def cir_domain(T: float) -> tuple[float, float]:
            level = max(r0, th, s**2 / (2 * k))
            sd = s * math.sqrt(level * (1 - math.exp(-2 * k * T)) / (2 * k))
            return 0.0, max(r0, th) + 10 * sd

        spec = dict(
            drift=lambda x, t: k * (th - x),
            volatility=lambda x, t: s * np.sqrt(np.maximum(x, 0.0)),
            x0=r0,
            lower=0.0,
            domain=cir_domain,
            time_homogeneous=True,
        )
    elif cls == "CEVProcess":
        mu, s, g, s0 = process.mu, process.sigma, process.gamma, process.s0

        def cev_domain(T: float) -> tuple[float, float]:
            loc = s * s0 ** (g - 1.0)
            return 0.0, s0 * math.exp(6 * loc * math.sqrt(T) + abs(mu) * T)

        spec = dict(
            drift=lambda x, t: mu * x,
            volatility=lambda x, t: s * np.maximum(x, 0.0) ** g,
            x0=s0,
            lower=0.0,
            domain=cev_domain,
            time_homogeneous=True,
        )
    elif cls == "BrownianMotion":
        mu, s = float(process.mu), float(process.sigma)
        spec = dict(drift=mu, volatility=s, x0=0.0, time_homogeneous=True)
    else:
        try:
            triplet = getattr(process, "triplet", None)
        except Exception:  # processes without an exact Levy triplet raise here
            triplet = None
        if triplet is not None and getattr(triplet, "nu", None) is not None:
            if "r" not in kwargs:
                raise ValueError(
                    f"{cls} has jumps; its generator is an integro-differential operator. "
                    "Pass r= (and optionally q=) to obtain the exponential-Levy pricing "
                    "PIDE (LevyPIDE) with S = S0 exp((r - q)t + X_t)."
                )
            from spxa.pde.pide import LevyPIDE

            extra = set(kwargs) - {"r", "q"}
            if extra:
                raise TypeError(f"unexpected arguments for a Levy process: {sorted(extra)}")
            return LevyPIDE.from_process(process, kwargs["r"], kwargs.get("q", 0.0))
        raise NotImplementedError(f"no PDE representation is registered for {cls}")
    spec["name"] = getattr(process, "name", cls) or cls
    spec.update(kwargs)
    return DiffusionPDE(**spec)
