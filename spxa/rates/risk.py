r"""Curve risk: node deltas, quote (par-point) deltas and key-rate deltas.

A fitted curve is a function of its node values :math:`x_j = -\log P(t_j)`,
and the nodes are a function of the market quotes :math:`q_i`. For a
price :math:`V(x)`:

* **node deltas** :math:`\partial V/\partial x_j` are central differences in
  each node, with the interpolation scheme left in place;
* **quote deltas** follow from the implicit function theorem. The fit
  makes the helper model quotes :math:`m(x)` equal to :math:`q`, so
  :math:`\partial x/\partial q = J^{-1}` with :math:`J = \partial m/\partial x`, and

  .. math:: \frac{\partial V}{\partial q} = J^{-\top}\,\frac{\partial V}{\partial x}.

  This is the sensitivity to each quote with the others held fixed, which is
  what a full re-bootstrap after bumping one quote gives, but costs one
  Jacobian instead of one bootstrap per quote.

**Key-rate deltas** (Ho, 1992) need no fit at all: the zero curve of any
:class:`~spxa.market.curves.YieldCurve` is moved by triangular hats
:math:`h_k(t)` centred on key tenors, flat beyond the first and last keys,

.. math:: P'(t) = P(t)\exp\Bigl(-t\sum_k b_k h_k(t)\Bigr).

The hats sum to one, so the key-rate deltas add up to the delta for a
parallel zero-rate shift to first order.

A price is any function of a mapping of named curves, for example
``lambda c: swap.pv(c["discount"], c["EUR-6M"])``. Several fitted curves
are handled together: nodes and helpers are stacked across them, so the
quote deltas of a multi-curve fit include the cross effects (a 6M swap
quote moving the discount curve through basis swaps, for instance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.interpolation import PiecewiseCurve
from spxa.rates.turns import SpreadAdjustedCurve

__all__ = [
    "PriceFunction",
    "NodeRisk",
    "QuoteRisk",
    "curve_nodes",
    "with_node_values",
    "node_deltas",
    "quote_jacobian",
    "quote_deltas",
    "KeyRateShiftedCurve",
    "KeyRateRisk",
    "key_rate_hats",
    "key_rate_deltas",
    "HedgeResult",
    "hedge_notionals",
    "par_hedge",
    "node_gamma",
    "PnLExplain",
    "pnl_explain",
]

PriceFunction = Callable[[Mapping[str, YieldCurve]], float]


def _base(curve: YieldCurve) -> PiecewiseCurve:
    base = curve.base if isinstance(curve, SpreadAdjustedCurve) else curve
    if not isinstance(base, PiecewiseCurve):
        raise TypeError(f"node risk needs a PiecewiseCurve, got {type(base).__name__}")
    return base


def curve_nodes(curve: YieldCurve) -> tuple[np.ndarray, np.ndarray]:
    """Node times and node values :math:`x = -\\log P` of the total curve."""
    t = _base(curve).node_times
    return t, -np.log(np.asarray(curve.discount(t), float))


def with_node_values(curve: YieldCurve, x: Sequence[float]) -> YieldCurve:
    """The same curve (scheme, dates, spreads) through new node values ``x``."""
    base = _base(curve)
    x = np.asarray(x, float)
    if x.shape != base.node_times.shape:
        raise ValueError("need one value per node")
    if isinstance(curve, SpreadAdjustedCurve):
        x = x - curve.log_adjustment(base.node_times)
    new = PiecewiseCurve(
        base.node_times,
        np.exp(-x),
        base.scheme,
        reference_date=base.reference_date,
        day_count=base.day_count,
    )
    return (
        SpreadAdjustedCurve(new, curve.spreads) if isinstance(curve, SpreadAdjustedCurve) else new
    )


@dataclass(frozen=True)
class NodeRisk:
    """:math:`\\partial V/\\partial x_j` for each ``(curve name, node time)``."""

    nodes: list[tuple[str, float]]
    deltas: np.ndarray

    def for_curve(self, name: str) -> np.ndarray:
        return np.array([d for (n, _), d in zip(self.nodes, self.deltas) if n == name])


@dataclass(frozen=True)
class QuoteRisk:
    """:math:`\\partial V/\\partial q_i` per unit quote, one entry per helper."""

    helpers: list[Any]
    deltas: np.ndarray

    @property
    def per_bp(self) -> np.ndarray:
        """Change in value for a one-basis-point rise in each quote."""
        return 1e-4 * self.deltas

    @property
    def total(self) -> float:
        """Sum of the deltas: the response to an equal move in every quote."""
        return float(np.sum(self.deltas))

    def table(self) -> str:
        rows = [f"{'helper':<22}{'pillar':>12}{'per bp':>16}"]
        for h, d in zip(self.helpers, self.per_bp):
            rows.append(f"{type(h).__name__:<22}{h.pillar.isoformat():>12}{d:16.6e}")
        return "\n".join(rows)


def _names(curves: Mapping[str, YieldCurve], names: Optional[Sequence[str]]) -> list[str]:
    if names is None:
        names = [n for n, c in curves.items() if isinstance(_maybe_base(c), PiecewiseCurve)]
    for n in names:
        if n not in curves:
            raise KeyError(f"no curve named {n!r}")
    return list(names)


def _maybe_base(c: YieldCurve) -> YieldCurve:
    return c.base if isinstance(c, SpreadAdjustedCurve) else c


def _stack(curves: Mapping[str, YieldCurve], names: Sequence[str]) -> tuple[list, np.ndarray]:
    labels: list[tuple[str, float]] = []
    xs = []
    for n in names:
        t, x = curve_nodes(curves[n])
        labels.extend((n, float(ti)) for ti in t)
        xs.append(x)
    return labels, np.concatenate(xs) if xs else np.zeros(0)


def _bumped(
    curves: Mapping[str, YieldCurve], names: Sequence[str], x: np.ndarray
) -> dict[str, YieldCurve]:
    out = dict(curves)
    i = 0
    for n in names:
        k = _base(curves[n]).node_times.size
        out[n] = with_node_values(curves[n], x[i : i + k])
        i += k
    return out


def node_deltas(
    pv: PriceFunction,
    curves: Mapping[str, YieldCurve],
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-7,
) -> NodeRisk:
    """Central-difference sensitivities of ``pv`` to every node of the named curves.

    ``names`` defaults to every piecewise curve in ``curves``.
    """
    names = _names(curves, names)
    labels, x = _stack(curves, names)
    d = np.empty(x.size)
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = bump
        d[j] = (pv(_bumped(curves, names, x + e)) - pv(_bumped(curves, names, x - e))) / (2 * bump)
    return NodeRisk(labels, d)


def quote_jacobian(
    helpers: Sequence[Any],
    curves: Mapping[str, YieldCurve],
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-7,
) -> np.ndarray:
    """:math:`J_{ij} = \\partial m_i/\\partial x_j`: helper model quotes against stacked nodes."""
    names = _names(curves, names)
    _, x = _stack(curves, names)
    J = np.empty((len(helpers), x.size))
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = bump
        up, dn = _bumped(curves, names, x + e), _bumped(curves, names, x - e)
        J[:, j] = [(h.implied_quote(up) - h.implied_quote(dn)) / (2 * bump) for h in helpers]
    return J


def _from_result(result: Any) -> tuple[list, dict[str, YieldCurve], list[str]]:
    if hasattr(result, "nodes"):  # GlobalFitResult
        names = list(dict.fromkeys(n for n, _ in result.nodes))
        return list(result.helpers), dict(result.curves), names
    return list(result.helpers), dict(result.curves), [result.name]


def quote_deltas(
    pv: PriceFunction,
    result: Any = None,
    *,
    helpers: Optional[Sequence[Any]] = None,
    curves: Optional[Mapping[str, YieldCurve]] = None,
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-7,
    jacobian: Optional[np.ndarray] = None,
) -> QuoteRisk:
    """Sensitivities of ``pv`` to each helper quote, others held fixed.

    Pass a :class:`~spxa.rates.bootstrap.BootstrapResult` or a
    :class:`~spxa.rates.bootstrap.GlobalFitResult` as ``result``, or the
    ``helpers``, ``curves`` and fitted curve ``names`` directly. The system
    must be square (one node per helper), as in both fits. A precomputed
    ``jacobian`` from :func:`quote_jacobian` can be reused across prices.
    """
    if result is not None:
        helpers, curves, names = _from_result(result)
    if helpers is None or curves is None:
        raise ValueError("pass a fit result, or helpers and curves")
    names = _names(curves, names)
    J = quote_jacobian(helpers, curves, names, bump) if jacobian is None else np.asarray(jacobian)
    if J.shape[0] != J.shape[1]:
        raise ValueError(f"need as many helpers as nodes, got {J.shape}")
    dv = node_deltas(pv, curves, names, bump).deltas
    return QuoteRisk(list(helpers), np.linalg.solve(J.T, dv))


# ----------------------------------------------------------------- key rates
def key_rate_hats(keys: Sequence[float], t: Any) -> np.ndarray:
    """Triangular weights :math:`h_k(t)`, shape ``(len(keys),) + shape(t)``; columns sum to one."""
    k = np.asarray(keys, float)
    if k.ndim != 1 or k.size == 0 or np.any(np.diff(k) <= 0) or k[0] < 0:
        raise ValueError("keys must be non-negative and strictly increasing")
    t = np.asarray(t, float)
    eye = np.eye(k.size)
    return np.stack([np.interp(t, k, eye[i]) for i in range(k.size)])


class KeyRateShiftedCurve(YieldCurve):
    """``base`` with zero-rate shifts ``shifts[k]`` spread over key tenors by triangular hats."""

    def __init__(self, base: YieldCurve, keys: Sequence[float], shifts: Sequence[float]) -> None:
        super().__init__(base.reference_date, base.day_count)
        self.base = base
        self.keys = np.asarray(keys, float)
        self.shifts = np.asarray(shifts, float)
        if self.shifts.shape != self.keys.shape:
            raise ValueError("need one shift per key")
        key_rate_hats(self.keys, 0.0)  # validates the keys

    def _discount(self, t: np.ndarray) -> np.ndarray:
        dz = np.tensordot(self.shifts, key_rate_hats(self.keys, t), axes=1)
        return np.asarray(self.base.discount(t), float) * np.exp(-dz * t)


@dataclass(frozen=True)
class KeyRateRisk:
    r""":math:`\partial V/\partial z_k` per unit zero-rate move at each key tenor."""

    keys: np.ndarray
    deltas: np.ndarray

    @property
    def per_bp(self) -> np.ndarray:
        return 1e-4 * self.deltas

    @property
    def total(self) -> float:
        return float(np.sum(self.deltas))


def key_rate_deltas(
    pv: PriceFunction,
    curves: Mapping[str, YieldCurve],
    keys: Sequence[float],
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-4,
) -> KeyRateRisk:
    """Central-difference key-rate deltas, shifting the named curves together.

    ``names`` defaults to every curve, which is the usual single-currency
    convention of moving discount and projection zero rates in parallel.
    """
    names = list(curves) if names is None else list(names)
    for n in names:
        if n not in curves:
            raise KeyError(f"no curve named {n!r}")
    k = np.asarray(keys, float)
    d = np.empty(k.size)
    for i in range(k.size):
        e = np.zeros(k.size)
        e[i] = bump
        up = {**curves, **{n: KeyRateShiftedCurve(curves[n], k, e) for n in names}}
        dn = {**curves, **{n: KeyRateShiftedCurve(curves[n], k, -e) for n in names}}
        d[i] = (pv(up) - pv(dn)) / (2 * bump)
    return KeyRateRisk(k, d)


# ------------------------------------------------------------------- hedging
@dataclass(frozen=True)
class HedgeResult:
    """Hedge notionals and the risk left after hedging."""

    notionals: np.ndarray
    residual: np.ndarray

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))


def hedge_notionals(
    target: Sequence[float],
    hedges: Any,
    weights: Optional[Sequence[float]] = None,
) -> HedgeResult:
    r"""Notionals :math:`n` minimising :math:`\lVert W^{1/2}(d + H n)\rVert_2`.

    ``target`` is the risk vector :math:`d` of the book (quote, node or
    key-rate deltas), and column :math:`j` of ``hedges`` is the same risk
    of one unit of hedge :math:`j`. With as many independent hedges as risk
    buckets the hedge is exact; with fewer, the residual is orthogonal (in
    the :math:`W` inner product) to every hedge.
    """
    d = np.asarray(target, float)
    H = np.asarray(hedges, float)
    if H.ndim == 1:
        H = H[:, None]
    if H.shape[0] != d.size:
        raise ValueError("hedges need one row per risk bucket")
    w = np.ones(d.size) if weights is None else np.asarray(weights, float)
    if w.shape != d.shape or np.any(w <= 0):
        raise ValueError("need one positive weight per bucket")
    r = np.sqrt(w)
    n, *_ = np.linalg.lstsq(H * r[:, None], -d * r, rcond=None)
    return HedgeResult(n, d + H @ n)


def par_hedge(
    pv: PriceFunction,
    result: Any,
    hedges: Sequence[PriceFunction],
    bump: float = 1e-7,
) -> HedgeResult:
    """Hedge the quote deltas of ``pv`` with instruments given as price functions.

    The Jacobian of ``result`` is computed once and reused for the book
    and every hedge.
    """
    helpers, curves, names = _from_result(result)
    J = quote_jacobian(helpers, curves, names, bump)
    book = quote_deltas(pv, result, bump=bump, jacobian=J).deltas
    H = np.column_stack([quote_deltas(h, result, bump=bump, jacobian=J).deltas for h in hedges])
    return hedge_notionals(book, H)


# ------------------------------------------------------------- P&L explain
def node_gamma(
    pv: PriceFunction,
    curves: Mapping[str, YieldCurve],
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-5,
) -> np.ndarray:
    r"""Matrix :math:`\partial^2 V/\partial x_i \partial x_j` by central differences.

    The diagonal uses the three-point stencil and the off-diagonal the
    four-point one, each with error :math:`O(\text{bump}^2)`.
    """
    names = _names(curves, names)
    _, x = _stack(curves, names)
    n = x.size
    v0 = pv(curves)
    E = np.eye(n) * bump

    def f(dx: np.ndarray) -> float:
        return pv(_bumped(curves, names, x + dx))

    up = np.array([f(E[i]) for i in range(n)])
    dn = np.array([f(-E[i]) for i in range(n)])
    G = np.diag((up - 2 * v0 + dn) / bump**2)
    for i in range(n):
        for j in range(i + 1, n):
            g = (f(E[i] + E[j]) - f(E[i] - E[j]) - f(E[j] - E[i]) + f(-E[i] - E[j])) / (4 * bump**2)
            G[i, j] = G[j, i] = g
    return G


@dataclass(frozen=True)
class PnLExplain:
    """Full revaluation against first- and second-order node expansions."""

    full: float
    first_order: float
    second_order: float

    @property
    def explained(self) -> float:
        return self.first_order + self.second_order

    @property
    def unexplained(self) -> float:
        return self.full - self.explained


def pnl_explain(
    pv: PriceFunction,
    before: Mapping[str, YieldCurve],
    after: Mapping[str, YieldCurve],
    names: Optional[Sequence[str]] = None,
    bump: float = 1e-7,
    gamma_bump: float = 1e-5,
) -> PnLExplain:
    r"""Explain :math:`V(\text{after}) - V(\text{before})` by node moves.

    With :math:`\Delta x` the change in node values between two fits on the
    same nodes and scheme, the first-order term is
    :math:`\delta^\top\Delta x` and the second-order term
    :math:`\tfrac12\Delta x^\top\Gamma\,\Delta x`, both at ``before``.
    The unexplained remainder is third order in :math:`\Delta x`.
    """
    names = _names(before, names)
    labels0, x0 = _stack(before, names)
    labels1, x1 = _stack(after, names)
    if labels0 != labels1:
        raise ValueError("the two curve sets must share their nodes")
    dx = x1 - x0
    delta = node_deltas(pv, before, names, bump).deltas
    G = node_gamma(pv, before, names, gamma_bump)
    return PnLExplain(
        full=pv(after) - pv(before),
        first_order=float(delta @ dx),
        second_order=float(0.5 * dx @ G @ dx),
    )
