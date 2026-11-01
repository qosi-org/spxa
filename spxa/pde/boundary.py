r"""Boundary conditions for one-dimensional convection–diffusion operators.

Each class describes how the boundary node of a :class:`~spxa.pde.grid.Grid1D`
is closed. They fall into two groups:

**Algebraic (constraint) conditions.** The boundary value is determined by
the interior values at the same time level, so the boundary row of the
linear system is a constraint rather than a discretised PDE:

* :class:`Dirichlet` — :math:`u(x_b, \tau) = g(\tau)`.
* :class:`Linear` — :math:`u_{xx}(x_b) = 0`, imposed by linear extrapolation
  through the two nearest interior nodes. This is the standard far-field
  condition for option prices, which are asymptotically linear in the
  underlying (zero gamma).

**Dynamic conditions.** The PDE itself is discretised at the boundary node:

* :class:`Neumann` — :math:`u_x(x_b, \tau) = g(\tau)`, using a mirrored ghost
  node, which keeps the boundary row second-order accurate.
* :class:`Natural` — the PDE is applied with the diffusion term dropped and a
  second-order one-sided first derivative. This is exact at degenerate
  boundaries where the diffusion coefficient vanishes, such as
  :math:`S = 0` for Black–Scholes or :math:`r = 0` for the CIR model
  (Fichera theory: no boundary condition may be imposed there).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

BoundaryValue = Union[float, Callable[[float], float]]


def _eval(value: BoundaryValue, tau: float) -> float:
    v = float(value(tau)) if callable(value) else float(value)
    if not np.isfinite(v):
        raise ValueError("boundary value evaluated to a non-finite number")
    return v


class BoundaryCondition:
    """Base class for boundary conditions."""

    #: True when the boundary row is an algebraic constraint.
    is_constraint: bool = False

    def describe(self) -> str:
        """Short human-readable description."""
        return type(self).__name__


@dataclass(frozen=True)
class Dirichlet(BoundaryCondition):
    r"""Prescribed boundary value :math:`u(x_b, \tau) = g(\tau)`.

    Parameters
    ----------
    value : float or callable
        Constant value or function of the time to maturity ``tau``.
    """

    value: BoundaryValue = 0.0
    is_constraint = True

    def __call__(self, tau: float) -> float:
        return _eval(self.value, tau)

    def describe(self) -> str:
        return "Dirichlet(u = g(tau))"


@dataclass(frozen=True)
class Neumann(BoundaryCondition):
    r"""Prescribed derivative :math:`u_x(x_b, \tau) = g(\tau)` (ghost-node closure).

    Parameters
    ----------
    derivative : float or callable
        Constant slope or function of ``tau``. The slope is taken with
        respect to the grid variable :math:`x`, regardless of the side.
    """

    derivative: BoundaryValue = 0.0
    is_constraint = False

    def __call__(self, tau: float) -> float:
        return _eval(self.derivative, tau)

    def describe(self) -> str:
        return "Neumann(u_x = g(tau))"


@dataclass(frozen=True)
class Linear(BoundaryCondition):
    r"""Zero-curvature far-field condition :math:`u_{xx}(x_b) = 0`."""

    is_constraint = True

    @staticmethod
    def weights(x_b: float, x_1: float, x_2: float) -> tuple[float, float]:
        r"""Extrapolation weights :math:`u_b = w_1 u_1 + w_2 u_2`.

        :math:`x_1` is the interior node adjacent to the boundary node
        :math:`x_b` and :math:`x_2` the next one.
        """
        w2 = (x_b - x_1) / (x_2 - x_1)
        return 1.0 - w2, w2

    def describe(self) -> str:
        return "Linear(u_xx = 0)"


@dataclass(frozen=True)
class Natural(BoundaryCondition):
    """PDE applied at the boundary with one-sided convection and no diffusion."""

    is_constraint = False

    def describe(self) -> str:
        return "Natural(PDE at boundary, degenerate diffusion)"
