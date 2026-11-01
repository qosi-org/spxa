r"""Containers for PDE solutions, with interpolation and grid Greeks.

A :class:`PDESolution1D` wraps the nodal values produced by the solvers and
exposes them as functions of the *underlying*, even when the PDE was solved
in a transformed variable. For a grid in :math:`x = \log S`,

.. math::
    \Delta = \frac{u_x}{S}, \qquad \Gamma = \frac{u_{xx} - u_x}{S^2}.

Nodal derivatives use the second-order stencils of
:mod:`spxa.pde.operators` and are then interpolated with the local cubic
interpolant. The time derivative (calendar :math:`\Theta = -u_\tau`) comes
from the last two time levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spxa.pde.grid import Grid1D
from spxa.pde.operators import derivative_matrices

_VARIABLES = ("S", "log")


@dataclass
class ExerciseBoundary:
    r"""Early-exercise boundary :math:`S^*(\tau)` of an American-style contract.

    Attributes
    ----------
    taus : np.ndarray
        Times to maturity at which the boundary was recorded.
    locations : np.ndarray
        Boundary location in the underlying (``nan`` where no exercise
        region exists on the grid).
    side : {'left', 'right'}
        ``'left'`` when exercise is optimal below the boundary (puts),
        ``'right'`` when above it (calls).
    """

    taus: np.ndarray
    locations: np.ndarray
    side: str = "left"

    def at(self, tau: float) -> float:
        """Boundary location at ``tau`` (linear interpolation in time)."""
        ok = np.isfinite(self.locations)
        if not np.any(ok):
            return float("nan")
        return float(np.interp(tau, self.taus[ok], self.locations[ok]))

    def calendar(self, maturity: float) -> tuple[np.ndarray, np.ndarray]:
        """The boundary as a function of calendar time ``t = T - tau`` (increasing)."""
        t = maturity - self.taus
        order = np.argsort(t)
        return t[order], self.locations[order]


@dataclass
class PDESolution1D:
    """Solution of a one-dimensional pricing PDE at ``tau = T``.

    Parameters
    ----------
    grid : Grid1D
        Grid in the PDE variable.
    values : np.ndarray
        Nodal values at ``tau = T``.
    variable : {'S', 'log'}
        Whether the grid variable is the underlying itself or its logarithm.
    previous_values : np.ndarray, optional
        Values one time step earlier (``tau = T - dt_last``).
    dt_last : float, optional
        Size of the last time step.
    history, times : np.ndarray, optional
        Full time history and the corresponding ``tau`` values.
    exercise_boundary : ExerciseBoundary, optional
        Free boundary for American-style problems.
    info : dict
        Solver diagnostics and problem description.
    label : str
        Short description of the problem.
    """

    grid: Grid1D
    values: np.ndarray
    variable: str = "S"
    previous_values: Optional[np.ndarray] = None
    dt_last: Optional[float] = None
    history: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    exercise_boundary: Optional[ExerciseBoundary] = None
    info: dict[str, Any] = field(default_factory=dict)
    label: str = "PDE solution"

    def __post_init__(self) -> None:
        if self.variable not in _VARIABLES:
            raise ValueError(f"variable must be one of {_VARIABLES}")
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != self.grid.nodes.shape:
            raise ValueError("values must have one entry per grid node")
        self._derivs: Optional[tuple[np.ndarray, np.ndarray]] = None

    @classmethod
    def from_theta_result(cls, result: Any, variable: str = "S", **kwargs: Any) -> "PDESolution1D":
        """Build a solution from a :class:`~spxa.pde.theta.ThetaResult`."""
        times = result.time_grid.times
        return cls(
            grid=result.grid,
            values=result.values,
            variable=variable,
            previous_values=result.previous_values,
            dt_last=float(times[-1] - times[-2]),
            history=result.history,
            times=np.array(times) if result.history is not None else None,
            info=dict(result.info, **kwargs.pop("info", {})),
            **kwargs,
        )

    # ------------------------------------------------------------------ helpers
    @property
    def underlying_nodes(self) -> np.ndarray:
        """Grid nodes expressed in the underlying."""
        x = self.grid.nodes
        return np.exp(x) if self.variable == "log" else x

    def _to_x(self, s: Any) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.variable == "log":
            if np.any(s <= 0.0):
                raise ValueError("underlying must be positive for a log-space solution")
            return np.log(s)
        return s

    def _check_inside(self, x: np.ndarray) -> None:
        if np.any(x < self.grid.lo - 1e-12) or np.any(x > self.grid.hi + 1e-12):
            raise ValueError("evaluation point lies outside the computational domain")

    def _nodal_derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        if self._derivs is None:
            D1, D2 = derivative_matrices(self.grid.nodes)
            self._derivs = (D1 @ self.values, D2 @ self.values)
        return self._derivs

    @staticmethod
    def _shape(out: np.ndarray, s: Any) -> Any:
        return float(out[0]) if np.ndim(s) == 0 else out

    # ------------------------------------------------------------------ evaluation
    def value(self, s: Any) -> Any:
        """Solution at underlying value(s) ``s``."""
        x = np.atleast_1d(self._to_x(s))
        self._check_inside(x)
        return self._shape(self.grid.interpolate(self.values, x), s)

    __call__ = value

    def delta(self, s: Any) -> Any:
        r"""First derivative with respect to the underlying."""
        x = np.atleast_1d(self._to_x(s))
        self._check_inside(x)
        d1, _ = self._nodal_derivatives()
        ux = self.grid.interpolate(d1, x)
        out = ux / np.exp(x) if self.variable == "log" else ux
        return self._shape(out, s)

    def gamma(self, s: Any) -> Any:
        r"""Second derivative with respect to the underlying."""
        x = np.atleast_1d(self._to_x(s))
        self._check_inside(x)
        d1, d2 = self._nodal_derivatives()
        ux = self.grid.interpolate(d1, x)
        uxx = self.grid.interpolate(d2, x)
        out = (uxx - ux) / np.exp(2.0 * x) if self.variable == "log" else uxx
        return self._shape(out, s)

    def theta(self, s: Any) -> Any:
        r"""Calendar-time derivative :math:`\partial_t V = -\partial_\tau u`."""
        if self.previous_values is None or not self.dt_last:
            raise ValueError("theta requires the previous time level")
        x = np.atleast_1d(self._to_x(s))
        self._check_inside(x)
        du = (self.values - self.previous_values) / self.dt_last
        return self._shape(-self.grid.interpolate(du, x), s)

    def greeks(self, s: float) -> dict[str, float]:
        """Value, delta, gamma and (if available) theta at a single point."""
        out = {
            "value": float(self.value(s)),
            "delta": float(self.delta(s)),
            "gamma": float(self.gamma(s)),
        }
        if self.previous_values is not None and self.dt_last:
            out["theta"] = float(self.theta(s))
        return out

    def value_at(self, s: Any, tau: float) -> Any:
        """Solution at an intermediate time to maturity (requires the history)."""
        if self.history is None or self.times is None:
            raise ValueError("value_at requires a solution computed with store_history=True")
        if not (0.0 <= tau <= self.times[-1] + 1e-14):
            raise ValueError("tau outside the solved time interval")
        x = np.atleast_1d(self._to_x(s))
        self._check_inside(x)
        k = int(np.clip(np.searchsorted(self.times, tau) - 1, 0, self.times.size - 2))
        w = (tau - self.times[k]) / (self.times[k + 1] - self.times[k])
        u = (1.0 - w) * self.history[k] + w * self.history[k + 1]
        return self._shape(self.grid.interpolate(u, x), s)

    # ------------------------------------------------------------------ reporting
    def __story__(self) -> str:
        """Plain-language account of how the solution was obtained."""
        lines = [f"{self.label}"]
        lines.append(
            f"  grid: {self.grid.n} nodes on [{self.grid.lo:.6g}, {self.grid.hi:.6g}] "
            f"({'uniform' if self.grid.is_uniform else 'non-uniform'}) in "
            f"{'log S' if self.variable == 'log' else 'S'}"
        )
        info = self.info
        if "n_steps" in info:
            lines.append(
                f"  time stepping: theta={info.get('theta', 0.5)}, {info['n_steps']} steps, "
                f"{info.get('n_rannacher_steps', 0)} Rannacher start-up steps"
            )
        for key in ("pde", "boundary", "smoothing", "lcp", "method"):
            if key in info:
                lines.append(f"  {key}: {info[key]}")
        if self.exercise_boundary is not None:
            lines.append(f"  early exercise region on the {self.exercise_boundary.side}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PDESolution1D({self.label!r}, n={self.grid.n}, variable={self.variable!r})"


@dataclass
class PDEPricingResult:
    """Price and sensitivities of a contract computed by a PDE pricer.

    Attributes
    ----------
    price, delta, gamma, theta : float
        Value and grid Greeks at the spot (theta per year, calendar time).
    spot : float
        Spot at which the result is reported.
    solution : PDESolution1D
        Full grid solution (evaluate at other spots, plot, ...).
    vega, rho : float, optional
        Bump-and-revalue sensitivities, when requested.
    exercise_boundary : ExerciseBoundary, optional
        Early-exercise boundary for American-style contracts.
    method : str
        Short description of the numerical method.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    spot: float
    solution: PDESolution1D
    vega: Optional[float] = None
    rho: Optional[float] = None
    exercise_boundary: Optional[ExerciseBoundary] = None
    method: str = ""

    def __float__(self) -> float:
        return float(self.price)

    def as_dict(self) -> dict[str, float]:
        """Scalar results as a dictionary (missing sensitivities omitted)."""
        out = {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
        }
        if self.vega is not None:
            out["vega"] = self.vega
        if self.rho is not None:
            out["rho"] = self.rho
        return out

    def __story__(self) -> str:
        """Plain-language account of the computation and its results."""
        lines = [f"{self.method or self.solution.label} at S = {self.spot:g}"]
        lines.append(
            "  price = {:.8f}, delta = {:.6f}, gamma = {:.6f}, theta = {:.6f}".format(
                self.price, self.delta, self.gamma, self.theta
            )
        )
        if self.vega is not None or self.rho is not None:
            lines.append(f"  vega = {self.vega}, rho = {self.rho} (bump and revalue)")
        if self.exercise_boundary is not None and self.exercise_boundary.taus.size:
            final = self.exercise_boundary.at(float(self.exercise_boundary.taus[-1]))
            lines.append(f"  exercise boundary today: {final:.6g}")
        lines.append(self.solution.__story__())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PDEPricingResult(price={self.price:.8g}, delta={self.delta:.6g})"
