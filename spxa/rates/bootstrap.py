r"""Bootstrapping discount and projection curves from quoted instruments.

:func:`bootstrap_curve` builds one :class:`~spxa.rates.interpolation.PiecewiseCurve`
from a set of helpers. Every other curve the helpers read (for example
the OIS discount curve when a EURIBOR projection curve is built) is
passed in ``known``.

The helpers are sorted by pillar, and each pillar becomes a node. Node
:math:`i` is solved so that helper :math:`i` reprices exactly on the
curve through nodes :math:`0, \dots, i`; the unknown is
:math:`x_i = -\log P(t_i)`, found with Brent's method. For a *local*
scheme (:class:`LogLinearDiscount`, :class:`LinearZero`) later nodes do
not move the curve before :math:`t_i`, so one pass reprices every helper
exactly. For non-local schemes the passes are repeated (``max_passes``),
each re-solving every node on the full curve, until the node values stop
moving.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.rates.helpers import CurveHelper
from spxa.rates.interpolation import InterpolationScheme, LogLinearDiscount, PiecewiseCurve

__all__ = ["BootstrapResult", "bootstrap_curve", "CurveSpec", "GlobalFitResult", "fit_curves"]


@dataclass
class BootstrapResult:
    """Output of :func:`bootstrap_curve`."""

    curve: PiecewiseCurve
    helpers: list[CurveHelper]
    pillars: list[_dt.date]
    residuals: np.ndarray
    passes: int
    name: str = "discount"
    known: dict[str, YieldCurve] = field(default_factory=dict)

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    @property
    def curves(self) -> dict[str, YieldCurve]:
        """All curves, the bootstrapped one under its name."""
        return {**self.known, self.name: self.curve}


def _curve(
    times: np.ndarray,
    x: np.ndarray,
    scheme: InterpolationScheme,
    ref: _dt.date,
    dc: DayCount,
) -> PiecewiseCurve:
    return PiecewiseCurve(times, np.exp(-x), scheme, reference_date=ref, day_count=dc)


def bootstrap_curve(
    helpers: Sequence[CurveHelper],
    reference_date: DateLike,
    scheme: Optional[InterpolationScheme] = None,
    name: str = "discount",
    known: Optional[Mapping[str, YieldCurve]] = None,
    day_count: str = "ACT/365F",
    max_passes: Optional[int] = None,
    tol: float = 1e-13,
) -> BootstrapResult:
    r"""Bootstrap the curve ``name`` so that every helper reprices.

    Parameters
    ----------
    helpers
        Instruments whose model quotes read the curve ``name`` (and possibly
        curves in ``known``). Pillars must be distinct and after
        ``reference_date``.
    scheme
        Interpolation scheme (default :class:`LogLinearDiscount`).
    known
        Other curves by name.
    max_passes
        Passes over all nodes. The default is 1 for local schemes and 50
        for non-local ones. Passes stop once the node values move by less
        than ``tol``.
    tol
        Convergence tolerance on :math:`-\log P` between passes.
    """
    scheme = scheme if scheme is not None else LogLinearDiscount()
    known = dict(known or {})
    if name in known:
        raise ValueError(f"curve {name!r} is both known and bootstrapped")
    if not helpers:
        raise ValueError("no helpers")
    ref = to_date(reference_date)
    dc = DayCount(day_count)
    hs = sorted(helpers, key=lambda h: h.pillar)
    for h in hs:
        if name not in h.curves:
            raise ValueError(f"{type(h).__name__} does not read curve {name!r}")
        missing = set(h.curves) - set(known) - {name}
        if missing:
            raise ValueError(
                f"{type(h).__name__} also reads {sorted(missing)}: pass them as known "
                "or fit the curves together with fit_curves"
            )
    pillars = [h.pillar for h in hs]
    if len(set(pillars)) != len(pillars):
        raise ValueError("two helpers share a pillar; drop one or move a pillar")
    if pillars[0] <= ref:
        raise ValueError("pillars must be after the reference date")
    times = np.array([dc.year_fraction(ref, d) for d in pillars])
    n = times.size
    passes = max_passes if max_passes is not None else (1 if scheme.local else 50)
    if passes < 1:
        raise ValueError("max_passes must be at least 1")

    x = np.zeros(n)
    done = 0
    for p in range(passes):
        old = x.copy()
        for i, h in enumerate(hs):
            m = i + 1 if p == 0 else n
            if p == 0:
                # flat-forward guess from the previous node
                prev_t = times[i - 1] if i else 0.0
                prev_x = x[i - 1] if i else 0.0
                fwd = prev_x / prev_t if i else 0.03
                x[i] = prev_x + fwd * (times[i] - prev_t)

            def f(xi: float, i: int = i, h: CurveHelper = h, m: int = m) -> float:
                trial = x[:m].copy()
                trial[i] = xi
                curves = {**known, name: _curve(times[:m], trial, scheme, ref, dc)}
                return h.residual(curves)

            x[i] = _solve(f, x[i], times[i])
        done = p + 1
        if p > 0 and np.max(np.abs(x - old)) < tol:
            break
    curve = _curve(times, x, scheme, ref, dc)
    curves = {**known, name: curve}
    res = np.array([h.residual(curves) for h in hs])
    return BootstrapResult(curve, hs, pillars, res, done, name, known)


def _solve(f: Callable[[float], float], guess: float, t: float) -> float:
    """Root of ``f`` near ``guess`` in :math:`x = -\\log P` (the bracket grows outward)."""
    step = max(0.01 * t, 1e-4)
    lo, hi = guess - step, guess + step
    flo, fhi = f(lo), f(hi)
    k = 0
    while flo * fhi > 0:
        k += 1
        if k > 60:
            raise RuntimeError("could not bracket the bootstrap root")
        step *= 2.0
        if abs(flo) < abs(fhi):
            lo -= step
            flo = f(lo)
        else:
            hi += step
            fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200))


# ----------------------------------------------------------------------------
# simultaneous (global) calibration
# ----------------------------------------------------------------------------
@dataclass
class CurveSpec:
    """One curve of a simultaneous fit: its helpers set its nodes (one per pillar)."""

    name: str
    helpers: Sequence[CurveHelper]
    scheme: InterpolationScheme = field(default_factory=LogLinearDiscount)


@dataclass
class GlobalFitResult:
    """Output of :func:`fit_curves`."""

    curves: dict[str, YieldCurve]
    helpers: list[CurveHelper]
    residuals: np.ndarray
    iterations: int
    #: d(residual_i)/d(x_j) at the solution, x = -log P at the stacked nodes
    jacobian: np.ndarray
    #: (curve name, pillar) of each column of the Jacobian
    nodes: list[tuple[str, _dt.date]]

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))


def fit_curves(
    specs: Sequence[CurveSpec],
    reference_date: DateLike,
    known: Optional[Mapping[str, YieldCurve]] = None,
    day_count: str = "ACT/365F",
    tol: float = 1e-14,
    max_iter: int = 30,
    bump: float = 1e-7,
) -> GlobalFitResult:
    r"""Fit several curves at once by Newton's method on all helper residuals.

    Every helper of ``specs[c]`` puts a node at its pillar on curve ``c``,
    so the system is square. The unknowns :math:`x = -\log P` at all
    nodes are solved jointly. That handles instruments reading several
    curves being built (for example, swaps discounted on an OIS curve
    fitted in the same pass, or basis swaps) and non-local interpolation
    without Gauss–Seidel passes.

    The Jacobian uses forward differences of size ``bump``; a column is
    computed only for helpers that read the bumped curve. Steps are
    halved until the residual norm decreases. The iteration stops when
    :math:`\max|r| <` ``tol`` or the step is below ``tol``.
    """
    known = dict(known or {})
    ref = to_date(reference_date)
    dc = DayCount(day_count)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("curve names must be distinct")
    if set(names) & set(known):
        raise ValueError("a curve is both known and fitted")
    helpers: list[CurveHelper] = []
    layout: list[tuple[str, np.ndarray, InterpolationScheme, list[_dt.date]]] = []
    nodes: list[tuple[str, _dt.date]] = []
    for s in specs:
        hs = sorted(s.helpers, key=lambda h: h.pillar)
        if not hs:
            raise ValueError(f"curve {s.name!r} has no helpers")
        pil = [h.pillar for h in hs]
        if len(set(pil)) != len(pil):
            raise ValueError(f"two helpers of {s.name!r} share a pillar")
        if pil[0] <= ref:
            raise ValueError("pillars must be after the reference date")
        layout.append((s.name, np.array([dc.year_fraction(ref, d) for d in pil]), s.scheme, pil))
        helpers.extend(hs)
        nodes.extend((s.name, d) for d in pil)
    available = set(names) | set(known)
    for h in helpers:
        missing = set(h.curves) - available
        if missing:
            raise ValueError(f"{type(h).__name__} reads unknown curves {sorted(missing)}")
    sizes = [t.size for _, t, _, _ in layout]
    offs = np.cumsum([0] + sizes)
    n = int(offs[-1])
    reads = np.array([[name in h.curves for name in names] for h in helpers])

    def build(x: np.ndarray) -> dict[str, YieldCurve]:
        out: dict[str, YieldCurve] = dict(known)
        for c, (name, t, scheme, _) in enumerate(layout):
            out[name] = _curve(t, x[offs[c] : offs[c + 1]], scheme, ref, dc)
        return out

    def residuals(x: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        curves = build(x)
        idx = range(n) if rows is None else np.flatnonzero(rows)
        r = np.zeros(n)
        for i in idx:
            r[i] = helpers[i].residual(curves)
        return r

    x = np.concatenate([0.03 * t for _, t, _, _ in layout])
    r = residuals(x)
    J = np.zeros((n, n))
    it = 0
    for it in range(1, max_iter + 1):
        J = np.zeros((n, n))
        for c in range(len(layout)):
            rows = reads[:, c]
            for j in range(int(offs[c]), int(offs[c + 1])):
                xb = x.copy()
                xb[j] += bump
                J[:, j] = (residuals(xb, rows) - r * rows) / bump
        step = np.linalg.solve(J, -r)
        lam, norm0 = 1.0, float(np.linalg.norm(r))
        while True:
            xn = x + lam * step
            rn = residuals(xn)
            if float(np.linalg.norm(rn)) < norm0 or lam < 1e-6:
                break
            lam *= 0.5
        x, r = xn, rn
        if np.max(np.abs(r)) < tol or np.max(np.abs(lam * step)) < tol:
            break
    curves = build(x)
    return GlobalFitResult(curves, helpers, r, it, J, nodes)
