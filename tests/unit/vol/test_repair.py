"""Static-arbitrage detection and l1 repair of call grids."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.repair import call_arbitrage_constraints, repair_call_prices

SS = SSVI(-0.6, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)
T = [0.25, 0.5, 1.0]
R, Q, S = 0.03, 0.01, 100.0
F = [S * math.exp((R - Q) * t) for t in T]
D = [math.exp(-R * t) for t in T]
K = [np.arange(80.0, 125.0, 5.0), np.arange(75.0, 130.0, 5.0), np.arange(70.0, 140.0, 10.0)]


def _prices() -> list[np.ndarray]:
    return [
        black_price(f, k, t, SS.implied_vol(np.log(k / f), t), d) for f, k, t, d in zip(F, K, T, D)
    ]


def _independent_check(strikes: list, prices: list) -> bool:
    """Convex, decreasing, bounded slices and calendar order on a fine grid."""
    ok = True
    curves = []
    for k, p, f, d in zip(strikes, prices, F, D):
        kap, c = np.concatenate([[0.0], k / f]), np.concatenate([[1.0], p / (f * d)])
        s = np.diff(c) / np.diff(kap)
        ok &= bool(np.all(s >= -1 - 1e-12) and np.all(s <= 1e-12) and np.all(np.diff(s) >= -1e-9))
        curves.append((kap, c))
    for (ka, ca), (kb, cb) in zip(curves[:-1], curves[1:]):
        x = np.linspace(max(ka[1], kb[1]), min(ka[-1], kb[-1]), 2001)
        ok &= bool(np.all(np.interp(x, kb, cb) - np.interp(x, ka, ca) >= -1e-12))
    return ok


def test_arbitrage_free_prices_untouched() -> None:
    p = _prices()
    cons = call_arbitrage_constraints(K, F)
    c = np.concatenate([pi / (f * d) for pi, f, d in zip(p, F, D)])
    assert cons.violations(c) == {}
    res = repair_call_prices(K, p, F, D)
    assert res.l1 == 0.0 and res.violations_before == {} and res.violations_after == {}
    for a, b in zip(res.prices, p):
        assert a == pytest.approx(b, abs=1e-15 * S)


def test_single_butterfly_minimal_repair() -> None:
    p = _prices()
    j = 4
    bumped = [x.copy() for x in p]
    v0 = bumped[1][j] - 0.5 * (bumped[1][j - 1] + bumped[1][j + 1])  # < 0: convex
    bumped[1][j] += -v0 + 0.05  # middle call now 0.05 above the chord
    res = repair_call_prices(K, bumped, F, D)
    assert res.violations_before.get("butterfly", 0) >= 1 and res.violations_after == {}
    # lowering the middle price by the excess is optimal (moving a wing costs twice as much)
    excess = (0.05) / (F[1] * D[1])
    assert res.l1 == pytest.approx(excess, rel=1e-9)
    assert res.perturbation[1][j] == pytest.approx(-0.05, rel=1e-9)
    assert _independent_check(K, res.prices)


def test_calendar_violation_repaired() -> None:
    p = _prices()
    bad = [x.copy() for x in p]
    bad[2] = bad[2] * 0.7  # one-year calls far too cheap
    res = repair_call_prices(K, bad, F, D)
    assert res.violations_before.get("calendar spread", 0) > 0
    assert res.violations_after == {}
    assert _independent_check(K, res.prices)
    assert not _independent_check(K, bad)


def test_noisy_grid_and_weights() -> None:
    rng = np.random.default_rng(3)
    noisy = [x + rng.normal(0, 0.15, x.size) for x in _prices()]
    noisy = [np.maximum(x, 1e-4) for x in noisy]
    res = repair_call_prices(K, noisy, F, D)
    assert res.violations_after == {} and _independent_check(K, res.prices)
    heavy = [np.full(k.size, 1.0) for k in K]
    heavy[0][:] = 1e3  # make the shortest expiry almost immovable
    rw = repair_call_prices(K, noisy, F, D, weights=heavy)
    assert np.max(np.abs(rw.perturbation[0])) <= np.max(np.abs(res.perturbation[0])) + 1e-12
    assert rw.violations_after == {}


def test_bid_ask_moves_are_free() -> None:
    p = _prices()
    bumped = [x.copy() for x in p]
    bumped[1][4] += 0.02
    spread = 0.5
    bid = [x - spread for x in bumped]
    ask = [x + spread for x in bumped]
    res = repair_call_prices(K, bumped, F, D, bid=bid, ask=ask)
    assert res.violations_after == {}
    for r, b_, a_ in zip(res.prices, bid, ask):
        assert np.all(r >= b_ - 1e-12) and np.all(r <= a_ + 1e-12)
    with pytest.raises(ValueError):
        repair_call_prices(K, bumped, F, D, bid=bid)
    with pytest.raises(ValueError):
        repair_call_prices(K, bumped, F, D, bid=ask, ask=ask)


def test_constraint_counts_and_validation() -> None:
    cons = call_arbitrage_constraints([np.array([90.0, 100.0, 110.0])], [100.0])
    assert cons.kinds.count("butterfly") == 2  # at the first two strikes (the anchor is c(0) = 1)
    assert cons.kinds.count("vertical spread") == 1
    assert cons.kinds.count("lower bound") == 3 and cons.index[2] == (0, 2)
    # a call above its upper bound or below intrinsic is caught
    assert cons.violations(np.array([1.2, 0.05, 0.01]))["upper bound"] == 1
    assert "lower bound" in cons.violations(np.array([0.05, 0.04, 0.01]))
    with pytest.raises(ValueError):
        call_arbitrage_constraints([np.array([100.0, 90.0])], [100.0])
    with pytest.raises(ValueError):
        repair_call_prices(K, _prices()[:2], F, D)
