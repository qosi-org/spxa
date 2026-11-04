r"""Detecting and repairing static arbitrage in a grid of call prices.

This follows Cohen, Reisinger & Wang, *Detecting and repairing arbitrage in
traded option prices* (Applied Mathematical Finance, 2020). Work in
normalised units :math:`c = C/(D F)` and :math:`\kappa = K/F` for each
expiry, which removes rates and proportional dividends. A grid of calls
:math:`c_i(\kappa_{ij})` then extends to a surface free of static
arbitrage, linear in :math:`\kappa` between quotes, if:

* **Bounds.** :math:`\max(1 - \kappa, 0) \le c \le 1`.
* **Vertical spreads.** With the anchor :math:`c(0) = 1`, each slice has
  slopes in :math:`[-1, 0]`.
* **Butterflies.** The slopes are non-decreasing (convexity).
* **Calendar spreads.** At fixed :math:`\kappa`, :math:`c` is
  non-decreasing in :math:`T`. Piecewise-linear slices differ by a
  piecewise-linear function on the union of their strike grids, so the
  condition is checked at every strike of either slice that lies inside
  both ranges.

Every condition is linear in the prices. :func:`repair_call_prices`
therefore finds the smallest weighted :math:`\ell^1` perturbation that
satisfies them all by solving a linear programme (HiGHS). Optionally,
moves that stay inside the bid-ask spread cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "ArbitrageConstraints",
    "RepairResult",
    "call_arbitrage_constraints",
    "repair_call_prices",
]


@dataclass
class ArbitrageConstraints:
    """Linear constraints :math:`A c \\ge b` on stacked normalised call prices."""

    A: np.ndarray
    b: np.ndarray
    kinds: list[str]
    #: (expiry index, strike index) of each stacked price
    index: list[tuple[int, int]]

    def violations(self, c: np.ndarray, tol: float = 1e-12) -> dict[str, int]:
        """Number of violated constraints by kind."""
        gap = self.A @ c - self.b
        out: dict[str, int] = {}
        for kind, g in zip(self.kinds, gap):
            if g < -tol:
                out[kind] = out.get(kind, 0) + 1
        return out


def _normalise(
    strikes: Sequence[Any], forwards: Sequence[float]
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    kap: list[np.ndarray] = []
    idx: list[tuple[int, int]] = []
    for i, (K, F) in enumerate(zip(strikes, forwards)):
        k = np.asarray(K, float) / float(F)
        if k.ndim != 1 or k.size < 1 or np.any(np.diff(k) <= 0) or k[0] <= 0:
            raise ValueError("strikes must be positive and strictly increasing per expiry")
        kap.append(k)
        idx.extend((i, j) for j in range(k.size))
    return kap, idx


def call_arbitrage_constraints(
    strikes: Sequence[Any], forwards: Sequence[float]
) -> ArbitrageConstraints:
    r"""Static-arbitrage constraints for call grids ``strikes[i]`` at expiries ``i``.

    Expiries must be ordered by increasing maturity. The returned rows act
    on normalised prices :math:`c = C/(DF)` stacked expiry by expiry.
    """
    kap, idx = _normalise(strikes, forwards)
    offs = np.cumsum([0] + [k.size for k in kap])
    n = int(offs[-1])
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    kinds: list[str] = []

    def add(coef: dict[int, float], b: float, kind: str) -> None:
        r = np.zeros(n)
        for j, v in coef.items():
            r[j] += v
        rows.append(r)
        rhs.append(b)
        kinds.append(kind)

    for i, k in enumerate(kap):
        o = int(offs[i])
        m = k.size
        for j in range(m):
            add({o + j: 1.0}, max(1.0 - k[j], 0.0), "lower bound")
            add({o + j: -1.0}, -1.0, "upper bound")
        # slopes with the anchor (0, 1): s_0 = (c_0 - 1)/k_0, s_j = (c_j - c_{j-1})/(k_j - k_{j-1})
        # s_0 >= -1  <=>  c_0 >= 1 - k_0 (already a bound); s_last <= 0
        if m >= 2:
            add({o + m - 1: -1.0, o + m - 2: 1.0}, 0.0, "vertical spread")
        # convexity: s_j >= s_{j-1} for consecutive slopes, including the anchor slope
        pts = [(0.0, None)] + [(k[j], o + j) for j in range(m)]
        for a in range(1, len(pts) - 1):
            (x0, i0), (x1, i1), (x2, i2) = pts[a - 1], pts[a], pts[a + 1]
            h0, h1 = x1 - x0, x2 - x1
            # (c2 - c1)/h1 - (c1 - c0)/h0 >= 0
            coef: dict[int, float] = {}
            b = 0.0
            coef[i2] = 1.0 / h1
            coef[i1] = coef.get(i1, 0.0) - 1.0 / h1 - 1.0 / h0
            if i0 is None:
                b -= 1.0 / h0  # c0 = 1 moves to the right-hand side
            else:
                coef[i0] = coef.get(i0, 0.0) + 1.0 / h0
            add(coef, b, "butterfly")

    for i in range(len(kap) - 1):
        ka, kb = kap[i], kap[i + 1]
        oa, ob = int(offs[i]), int(offs[i + 1])
        lo, hi = max(ka[0], kb[0]), min(ka[-1], kb[-1])
        if lo > hi:
            continue
        for x in np.union1d(ka, kb):
            if x < lo or x > hi:
                continue
            coef = {}
            for k, o, sgn in ((kb, ob, 1.0), (ka, oa, -1.0)):
                j = int(np.searchsorted(k, x, side="left"))
                if j < k.size and k[j] == x:
                    coef[o + j] = coef.get(o + j, 0.0) + sgn
                else:
                    w = (x - k[j - 1]) / (k[j] - k[j - 1])
                    coef[o + j - 1] = coef.get(o + j - 1, 0.0) + sgn * (1 - w)
                    coef[o + j] = coef.get(o + j, 0.0) + sgn * w
            add(coef, 0.0, "calendar spread")
    return ArbitrageConstraints(np.array(rows), np.array(rhs), kinds, idx)


@dataclass
class RepairResult:
    """Output of :func:`repair_call_prices`."""

    prices: list[np.ndarray]
    perturbation: list[np.ndarray]
    l1: float
    violations_before: dict[str, int]
    violations_after: dict[str, int]


def repair_call_prices(
    strikes: Sequence[Any],
    prices: Sequence[Any],
    forwards: Sequence[float],
    discounts: Sequence[float],
    weights: Optional[Sequence[Any]] = None,
    bid: Optional[Sequence[Any]] = None,
    ask: Optional[Sequence[Any]] = None,
) -> RepairResult:
    r"""Smallest weighted :math:`\ell^1` change that removes static arbitrage.

    Parameters
    ----------
    strikes, prices
        Per expiry, ordered by maturity: strictly increasing strikes and
        the discounted call prices.
    forwards, discounts
        Forward and discount factor per expiry.
    weights
        Per-price weights on :math:`|\Delta c|` (default 1).
    bid, ask
        If given, moves that keep a price inside :math:`[\text{bid}, \text{ask}]`
        are free and only the distance outside the spread is penalised.

    Returns
    -------
    RepairResult
        Repaired prices in the original units.
    """
    n_exp = len(strikes)
    if not (len(prices) == len(forwards) == len(discounts) == n_exp):
        raise ValueError("strikes, prices, forwards and discounts must have one entry per expiry")
    cons = call_arbitrage_constraints(strikes, forwards)
    scale = np.concatenate(
        [
            np.full(len(np.atleast_1d(K)), float(F) * float(D))
            for K, F, D in zip(strikes, forwards, discounts)
        ]
    )
    c0 = np.concatenate([np.asarray(p, float).ravel() for p in prices]) / scale
    if c0.size != scale.size:
        raise ValueError("need one price per strike")
    n = c0.size
    w = (
        np.ones(n)
        if weights is None
        else np.concatenate([np.asarray(x, float).ravel() for x in weights])
    )
    if w.shape != (n,) or np.any(w < 0):
        raise ValueError("weights must be non-negative, one per price")
    if (bid is None) != (ask is None):
        raise ValueError("give both bid and ask, or neither")
    if bid is not None and ask is not None:
        lo = np.concatenate([np.asarray(x, float).ravel() for x in bid]) / scale - c0
        hi = np.concatenate([np.asarray(x, float).ravel() for x in ask]) / scale - c0
        if np.any(lo > 1e-15) or np.any(hi < -1e-15):
            raise ValueError("prices must lie inside [bid, ask]")
        lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    else:
        lo = hi = np.zeros(n)
    # variables: d (free) plus u, v >= 0 with lo - v <= d <= hi + u; cost w (u + v)
    A, b = cons.A, cons.b
    m = A.shape[0]
    Z = np.zeros((m, n))
    eye = np.eye(n)
    A_ub = np.vstack(
        [
            np.hstack([-A, Z, Z]),  # A (c0 + d) >= b
            np.hstack([eye, -eye, np.zeros((n, n))]),  # d - u <= hi
            np.hstack([-eye, np.zeros((n, n)), -eye]),  # -d - v <= -lo
        ]
    )
    b_ub = np.concatenate([A @ c0 - b, hi, -lo])
    cost = np.concatenate([np.zeros(n), w, w])
    bounds = [(None, None)] * n + [(0, None)] * (2 * n)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"linear programme failed: {res.message}")
    d = res.x[:n]
    d = np.where(np.abs(d) < 1e-15, 0.0, d)
    c1 = c0 + d
    out_p, out_d = [], []
    o = 0
    for K in strikes:
        m_i = len(np.atleast_1d(K))
        out_p.append(c1[o : o + m_i] * scale[o : o + m_i])
        out_d.append(d[o : o + m_i] * scale[o : o + m_i])
        o += m_i
    return RepairResult(
        out_p,
        out_d,
        float(np.sum(w * np.abs(d))),
        cons.violations(c0),
        cons.violations(c1, tol=1e-10),
    )
