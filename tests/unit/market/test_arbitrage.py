"""Tests for spxa.market.arbitrage."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market import (
    ArbitrageReport,
    OptionChain,
    OptionSurfaceQuotes,
    black_price,
    call_prices_from_chain,
    check_calendar_arbitrage,
    check_call_prices,
    check_surface_quotes,
)

F, P, T = 100.0, 0.97, 1.0
K = np.array([60.0, 70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0])


def svi_like_vol(k: np.ndarray, f: float, t: float) -> np.ndarray:
    x = np.log(k / f)
    w = 0.04 * t * (1 + 0.5 * x * x) - 0.01 * t * x  # convex total variance, increasing in t
    return np.sqrt(w / t)


def calls(k: np.ndarray = K, f: float = F, p: float = P, t: float = T) -> np.ndarray:
    return black_price(f, k, t, svi_like_vol(k, f, t), p, "call")


def payoff_grid(lo: float = 0.0, hi: float = 400.0) -> np.ndarray:
    return np.linspace(lo, hi, 40001)


class TestStrikeChecks:
    def test_black_prices_are_free(self) -> None:
        rep = check_call_prices(K, calls(), F, P, expiry=T)
        assert rep.is_free and bool(rep) and len(rep) == 0
        assert rep.summary() == "no static arbitrage detected"

    def test_unsorted_input(self) -> None:
        o = np.random.default_rng(0).permutation(K.size)
        assert check_call_prices(K[o], calls()[o], F, P).is_free

    def test_lower_bound(self) -> None:
        c = calls()
        c[0] = P * (F - K[0]) - 0.3
        rep = check_call_prices(K, c, F, P)
        (v,) = rep.by_kind("lower_bound")
        assert v.strikes == (60.0,) and v.amount == pytest.approx(0.3, abs=1e-12)

    def test_upper_bound(self) -> None:
        c = calls()
        c[0] = P * F + 0.25
        v = check_call_prices(K, c, F, P).by_kind("upper_bound")
        assert len(v) == 1 and v[0].amount == pytest.approx(0.25, abs=1e-12)

    def test_call_spread(self) -> None:
        c = calls()
        c[6] = c[5] + 0.1  # C(105) > C(100)
        rep = check_call_prices(K, c, F, P)
        v = [x for x in rep.by_kind("call_spread") if x.strikes == (100.0, 105.0)]
        assert len(v) == 1 and v[0].amount == pytest.approx(0.1, abs=1e-12)

    def test_put_spread(self) -> None:
        c = calls()
        c[0] = c[1] + P * 10.0 + 0.2  # C(60) - C(70) exceeds P*(70-60)
        v = [
            x
            for x in check_call_prices(K, c, F, P).by_kind("put_spread")
            if x.strikes == (60.0, 70.0)
        ]
        assert len(v) == 1 and v[0].amount == pytest.approx(0.2, abs=1e-12)

    def test_butterfly_amount_is_portfolio_profit(self) -> None:
        c = calls()
        c[5] += 1.0  # bump ATM call -> concavity at 100
        rep = check_call_prices(K, c, F, P)
        bf = {v.strikes: v for v in rep.by_kind("butterfly")}
        v = bf[(95.0, 100.0, 105.0)]
        # sell the butterfly (short wings, long body is the arbitrage reversed):
        # portfolio = w_l C(95) + w_r C(105) - C(100) has payoff >= 0 and cost -amount
        wl = 0.5
        S = payoff_grid()
        pay = (
            wl * np.maximum(S - 95, 0) + (1 - wl) * np.maximum(S - 105, 0) - np.maximum(S - 100, 0)
        )
        assert pay.min() >= -1e-12
        cost = wl * c[4] + (1 - wl) * c[6] - c[5]
        assert -cost == pytest.approx(v.amount, abs=1e-12)

    def test_nonuniform_butterfly_weights(self) -> None:
        k = np.array([80.0, 90.0, 120.0])
        c = np.array([20.0, 12.0, 0.0]) * P
        # chord at 90: (30*20 + 10*0)/40 = 15 > 12 -> convex, free
        assert check_call_prices(k, c, 80.0, P).is_free
        c2 = np.array([20.0, 15.5, 0.0]) * P
        (v,) = check_call_prices(k, c2, 80.0, P).by_kind("butterfly")
        assert v.amount == pytest.approx(0.5 * P, abs=1e-12)

    def test_tolerance(self) -> None:
        k = np.array([90.0, 100.0, 110.0])
        c = np.array([12.0, 7.0 + 1e-9, 2.0])  # linear prices plus a tiny bump
        (v,) = check_call_prices(k, c, 100.0, 1.0).violations
        assert v.kind == "butterfly" and v.amount == pytest.approx(1e-9, rel=1e-6)
        assert check_call_prices(k, c, 100.0, 1.0, tol=2e-9).is_free

    def test_summary_and_errors(self) -> None:
        c = calls()
        c[5] += 1.0
        s = check_call_prices(K, c, F, P).summary()
        assert "butterfly" in s and "worst" in s
        with pytest.raises(ValueError, match="distinct"):
            check_call_prices([100.0, 100.0], [1.0, 1.0], F)
        with pytest.raises(ValueError, match="equal length"):
            check_call_prices([100.0], [1.0, 2.0], F)
        with pytest.raises(ValueError, match="positive"):
            check_call_prices([100.0], [1.0], -1.0)


class TestCalendar:
    def slices(self) -> list:
        out = []
        for t, f, p in ((0.25, 100.5, 0.99), (0.5, 101.0, 0.98), (1.0, 102.0, 0.96)):
            k = f * np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4])
            out.append((t, k, calls(k, f, p, t), f, p))
        return out

    def test_increasing_total_variance_is_free(self) -> None:
        assert check_calendar_arbitrage(self.slices()).is_free

    def test_detects_inverted_variance(self) -> None:
        sl = self.slices()
        t, k, _, f, p = sl[1]
        # price the middle expiry with more variance than the last one at the money
        v = svi_like_vol(k, f, t) * 1.6
        sl[1] = (t, k, black_price(f, k, t, v, p, "call"), f, p)
        rep = check_calendar_arbitrage(sl)
        assert not rep.is_free
        assert all(x.kind == "calendar" and x.expiry == 1.0 for x in rep.violations)
        # the amount is the cost of the replicating long-T2 portfolio below the T1 call
        x = max(rep.violations, key=lambda v: v.amount)
        t2, k2, c2, f2, p2 = sl[2]
        kk = x.strikes[0]
        i = list(k).index(kk)
        c1 = sl[1][2][i]
        chord = np.interp(kk / f, k2 / f2, c2 / (p2 * f2))
        assert x.amount == pytest.approx((c1 / (p * f) - chord) * p2 * f2, rel=1e-12)

    def test_chord_has_no_false_positive(self) -> None:
        # sparse later expiry: chords overestimate prices, so equal-vol surfaces stay free
        t1, t2 = 0.5, 0.5000001
        k1 = np.linspace(80, 120, 41)
        k2 = np.array([80.0, 100.0, 120.0])
        s = (
            (t1, k1, calls(k1, F, 1.0, t1), F, 1.0),
            (t2, k2, calls(k2, F, 1.0, t2), F, 1.0),
        )
        assert check_calendar_arbitrage(s).is_free

    def test_outside_range_ignored(self) -> None:
        s = (
            (0.5, np.array([50.0]), np.array([60.0]), F, 1.0),
            (1.0, np.array([90.0, 110.0]), np.array([12.0, 3.0]), F, 1.0),
        )
        assert check_calendar_arbitrage(s).is_free


class TestChains:
    def chain(self, t: float = T, f: float = F, p: float = P) -> OptionChain:
        k = f * np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
        v = svi_like_vol(k, f, t)
        c = black_price(f, k, t, v, p, "call")
        q = black_price(f, k, t, v, p, "put")
        return OptionChain.from_arrays(t, k, c, c, q, q, forward=f, discount=p)

    def test_call_prices_from_chain(self) -> None:
        ch = self.chain()
        k, c = call_prices_from_chain(ch)
        assert np.max(np.abs(c - calls(k))) < 1e-12
        k2, c2 = call_prices_from_chain(ch, use="calls")
        assert np.max(np.abs(c2 - calls(k2))) < 1e-13
        with pytest.raises(ValueError, match="use"):
            call_prices_from_chain(ch, use="puts")

    def test_surface_free_and_perturbed(self) -> None:
        s = OptionSurfaceQuotes([self.chain(0.5, 101.0, 0.98), self.chain(1.0, 102.0, 0.96)])
        assert check_surface_quotes(s).is_free
        bad = self.chain(1.0, 102.0, 0.96)
        i = next(n for n, q in enumerate(bad.quotes) if not q.is_call)
        q0 = bad.quotes[i]
        bad.quotes[i] = type(q0)(q0.strike, q0.option, 0.0, 0.0)  # lowest put for free
        s2 = OptionSurfaceQuotes([self.chain(0.5, 101.0, 0.98), bad])
        rep = check_surface_quotes(s2)
        assert not rep.is_free and isinstance(rep, ArbitrageReport)
        assert rep.by_kind("calendar")
        with pytest.raises(ValueError, match="match"):
            check_surface_quotes(s, forwards=[100.0])
