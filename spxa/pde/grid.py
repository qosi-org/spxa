r"""Spatial and temporal grids for the finite-difference solvers.

Spatial grids
-------------
:class:`Grid1D` stores a strictly increasing set of nodes. Besides uniform
grids it provides the sinh-stretched grids of Tavella & Randall (2000) and
in 't Hout & Foulon (2010),

.. math::
    x(\xi) = c + \alpha \sinh\!\big(c_2 \xi + c_1 (1 - \xi)\big), \qquad
    \xi \in [0, 1],

with :math:`c_1 = \operatorname{asinh}((x_{\min} - c)/\alpha)` and
:math:`c_2 = \operatorname{asinh}((x_{\max} - c)/\alpha)`. Nodes cluster
around the centre :math:`c` (typically the strike or a barrier); a smaller
:math:`\alpha` concentrates them more strongly. Because :math:`x(\xi)` is a
smooth map of a uniform grid, central differences on it remain second-order
accurate.

Points of non-smoothness of the payoff (strikes, barriers) should lie exactly
on nodes or exactly at cell midpoints; :meth:`Grid1D.with_points` snaps the
nearest interior node onto each such point.

Temporal grids
--------------
:class:`TimeGrid` lives in time to maturity :math:`\tau = T - t`. Event dates
(dividends, monitoring dates, Bermudan exercise dates) are inserted exactly
as grid times so that jump conditions are applied at the right instant.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from spxa.pde._utils import check_positive, cubic_interpolate


class Grid1D:
    """A strictly increasing one-dimensional grid.

    Parameters
    ----------
    nodes : array_like
        Strictly increasing node locations (at least three).
    """

    def __init__(self, nodes: Sequence[float] | np.ndarray) -> None:
        arr = np.asarray(nodes, dtype=float).ravel()
        if arr.size < 3:
            raise ValueError("a Grid1D needs at least 3 nodes")
        if not np.all(np.isfinite(arr)):
            raise ValueError("grid nodes must be finite")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("grid nodes must be strictly increasing")
        self._nodes = arr
        self._nodes.setflags(write=False)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, lo: float, hi: float, n_nodes: int) -> "Grid1D":
        """Uniform grid with ``n_nodes`` nodes on ``[lo, hi]``."""
        if hi <= lo:
            raise ValueError("hi must exceed lo")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be at least 3")
        return cls(np.linspace(lo, hi, int(n_nodes)))

    @classmethod
    def sinh(
        cls,
        lo: float,
        hi: float,
        n_nodes: int,
        center: float,
        alpha: Optional[float] = None,
        align_center: bool = True,
        align: Iterable[float] = (),
    ) -> "Grid1D":
        r"""Sinh-stretched grid concentrated around ``center``.

        Parameters
        ----------
        lo, hi : float
            Domain end points.
        n_nodes : int
            Number of nodes.
        center : float
            Concentration point :math:`c` (need not lie inside ``[lo, hi]``).
        alpha : float, optional
            Stretching scale :math:`\alpha > 0`. Defaults to
            ``(hi - lo) / 10``. Smaller values give stronger clustering.
        align_center : bool
            When ``center`` lies inside the domain, place it exactly on a
            node. The map is applied separately on either side of the node
            nearest to :math:`\xi_c = -c_1/(c_2 - c_1)`, which keeps the
            spacing ratio across the centre at :math:`1 + O(1/n)`. This
            avoids snapping a node onto the strike afterwards, which would
            perturb the local spacing by up to half a cell and make the
            convergence of digital payoffs erratic.
        align : iterable of float
            Further points (strikes, barriers) to place exactly on nodes in
            the same way: the computational coordinate :math:`\xi` is split
            at every aligned point, each piece receives a node count
            proportional to its length in :math:`\xi`, and nodes are uniform
            in :math:`\xi` within each piece. The spacing therefore stays a
            smooth function of the index up to :math:`O(1/n)` jumps in the
            ratio at the aligned points.
        """
        if hi <= lo:
            raise ValueError("hi must exceed lo")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be at least 3")
        a = (hi - lo) / 10.0 if alpha is None else check_positive("alpha", alpha)
        c1 = np.arcsinh((lo - center) / a)
        c2 = np.arcsinh((hi - center) / a)
        n = int(n_nodes)
        pts = sorted({float(p) for p in align if lo < float(p) < hi})
        if align_center and lo < center < hi:
            pts = sorted(set(pts) | {float(center)})
        xi_pts = [(np.arcsinh((p - center) / a) - c1) / (c2 - c1) for p in pts]
        breaks = [0.0, *xi_pts, 1.0]
        idx = [0]
        for k, b in enumerate(breaks[1:-1], start=1):
            remaining = len(breaks) - 1 - k
            if idx[-1] + 1 > n - 1 - remaining:
                raise ValueError("too few nodes to align all points; increase n_nodes")
            j = min(max(int(round(b * (n - 1))), idx[-1] + 1), n - 1 - remaining)
            idx.append(j)
        idx.append(n - 1)
        xi = np.empty(n)
        for k in range(len(breaks) - 1):
            i0, i1 = idx[k], idx[k + 1]
            xi[i0 : i1 + 1] = np.linspace(breaks[k], breaks[k + 1], i1 - i0 + 1)
        nodes = center + a * np.sinh(c1 + (c2 - c1) * xi)
        for j, p in zip(idx[1:-1], pts):
            nodes[j] = p
        nodes[0], nodes[-1] = lo, hi
        return cls(nodes)

    # ------------------------------------------------------------------ properties
    @property
    def nodes(self) -> np.ndarray:
        """Node locations (read-only array)."""
        return self._nodes

    @property
    def n(self) -> int:
        """Number of nodes."""
        return int(self._nodes.size)

    @property
    def lo(self) -> float:
        """Left end point."""
        return float(self._nodes[0])

    @property
    def hi(self) -> float:
        """Right end point."""
        return float(self._nodes[-1])

    @property
    def spacing(self) -> np.ndarray:
        """Interval lengths ``h_i = x_{i+1} - x_i`` (length ``n - 1``)."""
        return np.diff(self._nodes)

    @property
    def is_uniform(self) -> bool:
        """True when all intervals have the same length (to rounding)."""
        h = self.spacing
        return bool(np.allclose(h, h[0], rtol=1e-10, atol=0.0))

    @property
    def faces(self) -> np.ndarray:
        """Control-volume faces: domain ends and interval midpoints (length ``n + 1``)."""
        x = self._nodes
        return np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])

    @property
    def cell_widths(self) -> np.ndarray:
        """Width of the control volume around each node (length ``n``)."""
        return np.diff(self.faces)

    # ------------------------------------------------------------------ operations
    def with_points(self, points: Iterable[float]) -> "Grid1D":
        """Return a grid in which the nearest interior node is moved onto each point.

        Points outside the open interval ``(lo, hi)`` are ignored. Raises if
        two points would claim the same node or if snapping would break
        monotonicity (refine the grid in that case).
        """
        x = self._nodes.copy()
        claimed: dict[int, float] = {}
        for p in sorted(float(q) for q in points):
            if not (x[0] < p < x[-1]):
                continue
            j = int(np.argmin(np.abs(self._nodes[1:-1] - p))) + 1
            if j in claimed and not np.isclose(claimed[j], p):
                raise ValueError(
                    f"points {claimed[j]} and {p} snap to the same node; refine the grid"
                )
            claimed[j] = p
            x[j] = p
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("snapping broke grid monotonicity; refine the grid")
        return Grid1D(x)

    def refine(self) -> "Grid1D":
        """Insert the midpoint of every interval (doubles the resolution)."""
        x = self._nodes
        out = np.empty(2 * x.size - 1)
        out[0::2] = x
        out[1::2] = 0.5 * (x[1:] + x[:-1])
        return Grid1D(out)

    def locate(self, x: float) -> int:
        """Index of the node nearest to ``x``."""
        return int(np.argmin(np.abs(self._nodes - float(x))))

    def contains(self, x: float) -> bool:
        """True when ``lo <= x <= hi``."""
        return bool(self.lo <= float(x) <= self.hi)

    def interpolate(self, values: np.ndarray, x: float | np.ndarray) -> np.ndarray:
        """Local cubic interpolation of nodal ``values`` at ``x``."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._nodes.shape:
            raise ValueError("values must have one entry per node")
        return cubic_interpolate(self._nodes, values, np.asarray(x, dtype=float))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        kind = "uniform" if self.is_uniform else "non-uniform"
        return f"Grid1D({kind}, n={self.n}, [{self.lo:.6g}, {self.hi:.6g}])"


class TimeGrid:
    r"""Grid in time to maturity :math:`\tau \in [0, T]` with exact event times.

    Parameters
    ----------
    times : array_like
        Strictly increasing times starting at ``0``.
    event_taus : iterable of float, optional
        Times (in :math:`\tau`) at which event hooks are applied. Each must
        coincide with a grid time.
    """

    def __init__(
        self,
        times: Sequence[float] | np.ndarray,
        event_taus: Iterable[float] = (),
    ) -> None:
        arr = np.asarray(times, dtype=float).ravel()
        if arr.size < 2 or arr[0] != 0.0:
            raise ValueError("a TimeGrid must start at tau=0 and contain at least one step")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("time grid must be strictly increasing")
        self._times = arr
        self._times.setflags(write=False)
        idx: list[int] = []
        for e in sorted(set(float(v) for v in event_taus)):
            j = int(np.argmin(np.abs(arr - e)))
            if not np.isclose(arr[j], e, rtol=0.0, atol=1e-12 * max(1.0, arr[-1])):
                raise ValueError(f"event tau={e} is not a grid time")
            if j == 0:
                raise ValueError("events at tau=0 are the terminal condition, not events")
            idx.append(j)
        self._event_indices = tuple(idx)

    @classmethod
    def uniform(cls, maturity: float, n_steps: int, event_taus: Iterable[float] = ()) -> "TimeGrid":
        """Piecewise-uniform grid with ``~n_steps`` steps and exact event times.

        The interval ``[0, T]`` is split at the event times and each piece
        receives a number of steps proportional to its length (at least one),
        so that every event is hit exactly.
        """
        T = check_positive("maturity", maturity)
        n = int(n_steps)
        if n < 1:
            raise ValueError("n_steps must be at least 1")
        events = sorted(set(float(e) for e in event_taus))
        for e in events:
            if not (0.0 < e <= T + 1e-14):
                raise ValueError(f"event tau={e} must lie in (0, T]")
        breaks = [0.0] + [e for e in events if e < T * (1 - 1e-14)] + [T]
        pieces = [np.array([0.0])]
        for a, b in zip(breaks[:-1], breaks[1:]):
            m = max(1, int(round(n * (b - a) / T)))
            pieces.append(np.linspace(a, b, m + 1)[1:])
        times = np.concatenate(pieces)
        times[-1] = T
        events = [min(e, T) for e in events]
        return cls(times, events)

    @classmethod
    def from_calendar_events(
        cls, maturity: float, n_steps: int, event_times: Iterable[float] = ()
    ) -> "TimeGrid":
        """Like :meth:`uniform` but with events given in calendar time ``t = T - tau``.

        Calendar times must lie in ``[0, T)``.
        """
        T = check_positive("maturity", maturity)
        taus = []
        for t in event_times:
            t = float(t)
            if not (0.0 <= t < T):
                raise ValueError(f"event time t={t} must lie in [0, T)")
            taus.append(T - t)
        return cls.uniform(T, n_steps, taus)

    @property
    def times(self) -> np.ndarray:
        """Grid times :math:`0 = \\tau_0 < \\dots < \\tau_N = T`."""
        return self._times

    @property
    def maturity(self) -> float:
        """Final time :math:`T`."""
        return float(self._times[-1])

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self._times.size - 1)

    @property
    def steps(self) -> np.ndarray:
        """Step sizes."""
        return np.diff(self._times)

    @property
    def event_indices(self) -> tuple[int, ...]:
        """Indices ``k`` such that an event is applied once ``tau_k`` is reached."""
        return self._event_indices

    @property
    def event_taus(self) -> np.ndarray:
        """Event times in :math:`\\tau`."""
        return self._times[list(self._event_indices)]

    def refine(self) -> "TimeGrid":
        """Halve every time step (events are preserved)."""
        t = self._times
        out = np.empty(2 * t.size - 1)
        out[0::2] = t
        out[1::2] = 0.5 * (t[1:] + t[:-1])
        return TimeGrid(out, self.event_taus)

    def __repr__(self) -> str:
        return (
            f"TimeGrid(T={self.maturity:.6g}, n_steps={self.n_steps}, "
            f"events={len(self._event_indices)})"
        )
