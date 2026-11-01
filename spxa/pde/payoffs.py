r"""Terminal payoffs and their projection onto finite-difference grids.

Why smoothing matters
---------------------
Sampling a non-smooth payoff pointwise at the nodes makes the leading error
term of a second-order scheme depend on where the kink or jump falls
relative to the nodes. Convergence then becomes erratic, and for
discontinuous payoffs it drops to first order. Pooley, Vetzal & Forsyth
(2003) showed that replacing the nodal value by the cell average

.. math::
    \bar g_i = \frac{1}{x_{i+1/2} - x_{i-1/2}} \int_{x_{i-1/2}}^{x_{i+1/2}} g(x)\,dx,
    \qquad x_{i\pm1/2} = \tfrac12 (x_i + x_{i\pm1}),

together with Rannacher time stepping restores clean second-order
convergence for prices *and* Greeks.

:func:`project_payoff` evaluates these averages exactly for piecewise
smooth payoffs. Each control volume is split at the declared kink and jump
points and integrated with Gauss–Legendre quadrature, so piecewise
polynomial payoffs of degree :math:`\le 2 n_{gauss} - 1` are integrated
exactly. A grid in a transformed variable (e.g. :math:`x = \log S`) is
handled by passing the transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.grid import Grid1D


@dataclass(frozen=True)
class Payoff:
    """A payoff ``g(S)`` together with its points of non-smoothness.

    Parameters
    ----------
    fn : callable
        Vectorised payoff function of the underlying.
    kinks : tuple of float
        Points where ``g`` or its derivative is discontinuous.
    name : str
        Description used in reports.
    """

    fn: Callable[[np.ndarray], np.ndarray]
    kinks: tuple[float, ...] = field(default_factory=tuple)
    name: str = "payoff"

    def __call__(self, s: np.ndarray | float) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(s, dtype=float)), dtype=float)

    def __add__(self, other: "Payoff") -> "Payoff":
        if not isinstance(other, Payoff):
            return NotImplemented
        f, g = self.fn, other.fn
        return Payoff(
            lambda s: f(s) + g(s),
            tuple(sorted(set(self.kinks) | set(other.kinks))),
            f"{self.name} + {other.name}",
        )

    def __mul__(self, scalar: float) -> "Payoff":
        c = float(scalar)
        f = self.fn
        return Payoff(lambda s: c * f(s), self.kinks, f"{c:g} * {self.name}")

    __rmul__ = __mul__

    def __neg__(self) -> "Payoff":
        return self * -1.0

    def __sub__(self, other: "Payoff") -> "Payoff":
        return self + (-other)


def vanilla(strike: float, option: str = "call") -> Payoff:
    """European call ``(S - K)^+`` or put ``(K - S)^+``."""
    K = check_positive("strike", strike)
    if check_option_type(option) == "call":
        return Payoff(lambda s: np.maximum(s - K, 0.0), (K,), f"call(K={K:g})")
    return Payoff(lambda s: np.maximum(K - s, 0.0), (K,), f"put(K={K:g})")


def digital(strike: float, option: str = "call", cash: float = 1.0) -> Payoff:
    """Cash-or-nothing digital paying ``cash`` if ``S > K`` (call) or ``S < K`` (put).

    The value exactly at the strike is ``cash / 2`` (the cell-average limit).
    """
    K = check_positive("strike", strike)
    c = float(cash)
    if check_option_type(option) == "call":
        fn = lambda s: c * np.where(s > K, 1.0, np.where(s == K, 0.5, 0.0))  # noqa: E731
        return Payoff(fn, (K,), f"digital_call(K={K:g})")
    fn = lambda s: c * np.where(s < K, 1.0, np.where(s == K, 0.5, 0.0))  # noqa: E731
    return Payoff(fn, (K,), f"digital_put(K={K:g})")


def asset_or_nothing(strike: float, option: str = "call") -> Payoff:
    """Asset-or-nothing digital paying ``S`` if ``S > K`` (call) or ``S < K`` (put)."""
    K = check_positive("strike", strike)
    if check_option_type(option) == "call":
        fn = lambda s: s * np.where(s > K, 1.0, np.where(s == K, 0.5, 0.0))  # noqa: E731
        return Payoff(fn, (K,), f"asset_call(K={K:g})")
    fn = lambda s: s * np.where(s < K, 1.0, np.where(s == K, 0.5, 0.0))  # noqa: E731
    return Payoff(fn, (K,), f"asset_put(K={K:g})")


def project_payoff(
    payoff: Payoff | Callable[[np.ndarray], np.ndarray],
    grid: Grid1D,
    method: str = "average",
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    kinks: Optional[tuple[float, ...]] = None,
    n_gauss: int = 6,
) -> np.ndarray:
    r"""Project a payoff onto the grid nodes.

    Parameters
    ----------
    payoff : Payoff or callable
        Payoff of the *underlying* :math:`S`.
    grid : Grid1D
        Grid in the PDE variable :math:`x` (with :math:`S = T(x)`).
    method : {'average', 'kink', 'none'}
        ``'average'`` returns control-volume averages (in :math:`x`) at
        interior nodes and samples at the two end points (whose control
        volumes are one-sided half cells). ``'kink'`` averages only the
        control volumes containing a kink and samples everywhere else, which
        removes the :math:`O(h^2)` averaging bias of smooth (e.g. linear)
        parts on stretched grids. ``'none'`` samples pointwise.
    transform, inverse : callable, optional
        :math:`T` and :math:`T^{-1}` when the grid variable is not :math:`S`
        (for example ``np.exp`` / ``np.log``). Kinks are mapped through
        ``inverse``.
    kinks : tuple of float, optional
        Overrides the payoff's own kinks (given in :math:`S`).
    n_gauss : int
        Gauss–Legendre nodes per sub-interval.
    """
    if (transform is None) != (inverse is None):
        raise ValueError("transform and inverse must be given together")

    def g(xv: np.ndarray) -> np.ndarray:
        s = transform(xv) if transform is not None else xv
        return np.asarray(payoff(s), dtype=float)

    x = grid.nodes
    if method == "none":
        return g(x)
    if method not in ("average", "kink"):
        raise ValueError("method must be 'average', 'kink' or 'none'")
    k_s = kinks if kinks is not None else getattr(payoff, "kinks", ())
    k_x = np.asarray(
        [float(inverse(np.asarray(k))) if inverse is not None else float(k) for k in k_s],
        dtype=float,
    )
    faces = grid.faces
    inner = k_x[(k_x > faces[0]) & (k_x < faces[-1])]
    breaks = np.union1d(faces, inner)
    a, b = breaks[:-1], breaks[1:]
    xi, wi = np.polynomial.legendre.leggauss(int(n_gauss))
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    pts = mid[:, None] + half[:, None] * xi[None, :]
    integrals = (g(pts.ravel()).reshape(pts.shape) * wi[None, :]).sum(axis=1) * half
    # map each sub-interval to the control volume containing it
    owner = np.searchsorted(faces, mid, side="right") - 1
    totals = np.bincount(owner, weights=integrals, minlength=x.size)
    out = totals / grid.cell_widths
    # end-point control volumes are half cells lying on one side of the node,
    # whose average is an O(h) biased nodal value; sample there instead
    out[[0, -1]] = g(x[[0, -1]])
    if method == "kink":
        smooth = np.ones(x.size, dtype=bool)
        if inner.size:
            smooth[np.searchsorted(faces, inner, side="right") - 1] = False
        out[smooth] = g(x[smooth])
    return out
