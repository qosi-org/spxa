r"""Model-free static-arbitrage checks on option prices.

For one expiry with forward :math:`F` and discount factor :math:`P`,
call prices :math:`C(K)` admit no static arbitrage
if and only if (Carr–Madan; Davis–Hobson)

* **bounds**: :math:`P(F-K)^+ \le C(K) \le PF`;
* **vertical spreads**: :math:`0 \le C(K_1) - C(K_2) \le P(K_2-K_1)` for
  :math:`K_1 < K_2`;
* **butterflies**: :math:`C` is convex in :math:`K`.

Across expiries, when forwards scale proportionally (no cash dividends),
the normalised price :math:`c(k, T) = C(kF_T, T)/(P_T F_T)` must be
non-decreasing in :math:`T` at fixed forward moneyness :math:`k`
(Carr & Madan 2005, Gatheral 2006).

Every reported violation corresponds to an explicit portfolio of the
*quoted* options with non-negative payoff and negative cost; the calendar
check interpolates the later expiry only by chords, which are themselves
replicable (a convex combination of calls dominates the call at the
combined strike), so it never reports false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from spxa.market.quotes import OptionChain, OptionSurfaceQuotes

__all__ = [
    "ArbitrageReport",
    "ArbitrageViolation",
    "call_prices_from_chain",
    "check_calendar_arbitrage",
    "check_call_prices",
    "check_surface_quotes",
]


@dataclass(frozen=True)
class ArbitrageViolation:
    """One static-arbitrage violation.

    ``kind`` is one of ``"lower_bound"``, ``"upper_bound"``,
    ``"call_spread"`` (prices increasing in strike), ``"put_spread"``
    (prices falling faster than the discount factor), ``"butterfly"`` or
    ``"calendar"``. ``amount`` is the arbitrage profit per unit of the
    offending portfolio, in price units (positive).
    """

    kind: str
    expiry: float
    strikes: tuple[float, ...]
    amount: float


@dataclass
class ArbitrageReport:
    """Collection of violations; truthy when arbitrage-free."""

    violations: list[ArbitrageViolation] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.is_free

    def __len__(self) -> int:
        return len(self.violations)

    def by_kind(self, kind: str) -> list[ArbitrageViolation]:
        return [v for v in self.violations if v.kind == kind]

    def extend(self, other: "ArbitrageReport") -> None:
        self.violations.extend(other.violations)

    def summary(self) -> str:
        if self.is_free:
            return "no static arbitrage detected"
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.kind] = counts.get(v.kind, 0) + 1
        worst = max(self.violations, key=lambda v: v.amount)
        parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        return f"{len(self.violations)} violations ({parts}); worst {worst.kind} {worst.amount:.6g} at K={worst.strikes}"


def check_call_prices(
    strikes: Sequence[float],
    calls: Sequence[float],
    forward: float,
    discount: float = 1.0,
    expiry: float = float("nan"),
    tol: float = 0.0,
) -> ArbitrageReport:
    """Static-arbitrage checks of one expiry's call prices.

    Parameters
    ----------
    strikes, calls : array_like
        Distinct strikes (any order) and the call prices.
    forward, discount : float
    expiry : float
        Label stored in the violations.
    tol : float
        Absolute price tolerance; violations smaller than ``tol`` are ignored.
    """
    K = np.asarray(strikes, float)
    C = np.asarray(calls, float)
    if K.shape != C.shape or K.ndim != 1:
        raise ValueError("strikes and calls must be 1-D arrays of equal length")
    if forward <= 0 or discount <= 0:
        raise ValueError("forward and discount must be positive")
    o = np.argsort(K)
    K, C = K[o], C[o]
    if np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be distinct")
    P = float(discount)
    out = ArbitrageReport()

    def add(kind: str, ks: tuple[float, ...], amt: float) -> None:
        if amt > tol:
            out.violations.append(ArbitrageViolation(kind, expiry, ks, float(amt)))

    for k, c in zip(K, C):
        add("lower_bound", (float(k),), P * max(forward - k, 0.0) - c)
        if k >= 0:
            add("upper_bound", (float(k),), c - P * forward)
    dK = np.diff(K)
    dC = np.diff(C)
    for i in range(K.size - 1):
        ks = (float(K[i]), float(K[i + 1]))
        add("call_spread", ks, dC[i])  # C(K2) > C(K1): buy K1, sell K2
        add("put_spread", ks, -dC[i] - P * dK[i])  # spread worth more than its max payoff
    for i in range(1, K.size - 1):
        wl = (K[i + 1] - K[i]) / (K[i + 1] - K[i - 1])
        bfly = wl * C[i - 1] + (1.0 - wl) * C[i + 1] - C[i]
        add("butterfly", (float(K[i - 1]), float(K[i]), float(K[i + 1])), -bfly)
    return out


def call_prices_from_chain(
    chain: OptionChain,
    forward: Optional[float] = None,
    discount: Optional[float] = None,
    use: str = "otm",
) -> tuple[np.ndarray, np.ndarray]:
    """Call-equivalent mid prices of a chain.

    ``use="otm"`` takes out-of-the-money options and maps puts to calls by
    parity :math:`C = P_{put} + P(F-K)`; ``use="calls"`` takes call mids only.
    """
    F = chain._need_forward(forward)
    P = chain._need_discount(discount)
    if use == "calls":
        return chain.mids("call")
    if use != "otm":
        raise ValueError("use must be 'otm' or 'calls'")
    kc, mc = chain.otm(F).mids("call")
    kp, mp_ = chain.otm(F).mids("put")
    K = np.concatenate([kp, kc])
    C = np.concatenate([mp_ + P * (F - kp), mc])
    o = np.argsort(K)
    return K[o], C[o]


def check_calendar_arbitrage(
    slices: Sequence[tuple[float, np.ndarray, np.ndarray, float, float]],
    tol: float = 0.0,
) -> ArbitrageReport:
    """Calendar checks across expiries at fixed forward moneyness.

    Parameters
    ----------
    slices : sequence of ``(T, strikes, calls, forward, discount)``
    tol : float
        Tolerance on the *normalised* price ``C / (P F)``.

    For consecutive expiries ``T1 < T2`` and each quoted moneyness ``k`` of
    ``T1`` inside the quoted range of ``T2``, the chord of ``T2``'s
    normalised prices at ``k`` must be at least ``c(k, T1)``.
    """
    sl = sorted(slices, key=lambda s: s[0])
    out = ArbitrageReport()
    for (t1, k1, c1, f1, p1), (t2, k2, c2, f2, p2) in zip(sl[:-1], sl[1:]):
        m1 = np.asarray(k1, float) / f1
        n1 = np.asarray(c1, float) / (p1 * f1)
        o2 = np.argsort(k2)
        m2 = np.asarray(k2, float)[o2] / f2
        n2 = np.asarray(c2, float)[o2] / (p2 * f2)
        if m2.size == 0:
            continue
        # moneyness grids built as k = m F agree only to rounding; admit the endpoints
        eps = 1e-12
        inside = (m1 >= m2[0] * (1 - eps)) & (m1 <= m2[-1] * (1 + eps))
        chord = np.interp(m1[inside], m2, n2)
        for k, a, b in zip(np.asarray(k1, float)[inside], n1[inside], chord):
            if a - b > tol:
                out.violations.append(
                    ArbitrageViolation(
                        "calendar",
                        float(t2),
                        (float(k), float(k / f1 * f2)),
                        float((a - b) * p2 * f2),
                    )
                )
    return out


def check_surface_quotes(
    surface: OptionSurfaceQuotes,
    forwards: Optional[Sequence[float]] = None,
    discounts: Optional[Sequence[float]] = None,
    tol: float = 0.0,
    use: str = "otm",
) -> ArbitrageReport:
    """Strike checks on every chain plus calendar checks across chains.

    Forwards and discounts default to those stored on each chain.
    """
    n = len(surface)
    fs = list(forwards) if forwards is not None else [None] * n
    ds = list(discounts) if discounts is not None else [None] * n
    if len(fs) != n or len(ds) != n:
        raise ValueError("forwards/discounts must match the number of chains")
    out = ArbitrageReport()
    slices = []
    for ch, f, d in zip(surface, fs, ds):
        F = ch._need_forward(f)
        P = ch._need_discount(d)
        K, C = call_prices_from_chain(ch, F, P, use)
        out.extend(check_call_prices(K, C, F, P, expiry=ch.T, tol=tol))
        slices.append((ch.T, K, C, F, P))
    out.extend(check_calendar_arbitrage(slices, tol=tol))
    return out
