"""Tests for spxa.market.carry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market import (
    EquityForward,
    FlatCurve,
    ImpliedForwardCurve,
    OptionChain,
    OptionSurfaceQuotes,
    black_price,
    implied_carry,
    implied_dividend_yield,
    implied_repo_rate,
)

S, R, Q, B = 100.0, 0.045, 0.015, 0.004


class TestRates:
    def test_round_trip_flat(self) -> None:
        T = np.array([0.25, 1.0, 3.0])
        P = np.exp(-R * T)
        Prepo = np.exp(-(R - B) * T)  # repo below risk-free: borrow fee B
        F = S * np.exp((R - B - Q) * T)
        assert np.allclose(implied_carry(S, F, T), R - B - Q, rtol=0, atol=1e-15)
        assert np.allclose(implied_dividend_yield(S, F, T, Prepo), Q, rtol=0, atol=1e-15)
        # against the risk-free curve the fee shows up as extra yield
        assert np.allclose(implied_dividend_yield(S, F, T, P), Q + B, rtol=0, atol=1e-15)
        assert np.allclose(implied_repo_rate(S, F, T, Q), R - B, rtol=0, atol=1e-15)

    def test_scalar_and_errors(self) -> None:
        assert isinstance(implied_carry(S, 101.0, 1.0), float)
        assert implied_carry(S, 101.0, 1.0) == pytest.approx(math.log(1.01), abs=1e-16)
        with pytest.raises(ValueError, match="positive"):
            implied_carry(S, -1.0, 1.0)
        with pytest.raises(ValueError, match="T must"):
            implied_carry(S, 101.0, 0.0)
        with pytest.raises(ValueError, match="repo_discount"):
            implied_dividend_yield(S, 101.0, 1.0, 0.0)

    def test_matches_equity_forward(self) -> None:
        ef = EquityForward(S, FlatCurve(R), dividend_yield=FlatCurve(Q), repo=FlatCurve(R - B))
        for t in (0.1, 0.7, 2.0):
            f = ef.forward(t)
            assert implied_dividend_yield(S, f, t, math.exp(-(R - B) * t)) == pytest.approx(
                Q, abs=1e-14
            )


class TestCurve:
    times = np.array([0.25, 0.5, 1.0, 2.0])
    carries = np.array([0.03, 0.025, 0.02, 0.028])  # average carry to each node
    discs = np.exp(-R * times)

    def curve(self) -> ImpliedForwardCurve:
        return ImpliedForwardCurve(S, self.times, S * np.exp(self.carries * self.times), self.discs)

    def test_nodes_and_spot(self) -> None:
        c = self.curve()
        assert np.allclose(c.forward(self.times), S * np.exp(self.carries * self.times), rtol=1e-15)
        assert c.forward(0.0) == S
        assert len(c) == 4 and "nodes=4" in repr(c)

    def test_piecewise_constant_carry(self) -> None:
        c = self.curve()
        lm = np.log(c.forward(np.linspace(0.5, 1.0, 11)) / S)
        assert np.max(np.abs(np.diff(lm, 2))) < 1e-15  # linear in t
        # instantaneous carry on (0.5, 1]: (1*0.02 - 0.5*0.025)/0.5
        h = 1e-6
        inst = (math.log(c.forward(0.75 + h)) - math.log(c.forward(0.75 - h))) / (2 * h)
        assert inst == pytest.approx((0.02 - 0.0125) / 0.5, abs=1e-8)

    def test_extrapolation_and_carry(self) -> None:
        c = self.curve()
        last = (2 * 0.028 - 1 * 0.02) / 1.0
        assert math.log(c.forward(3.0) / c.forward(2.0)) == pytest.approx(last, abs=1e-14)
        assert c.carry(1.0) == pytest.approx(0.02, abs=1e-15)
        assert c.carry(0.0) == pytest.approx(0.03, abs=1e-15)
        assert np.allclose(c.carry(self.times), self.carries, rtol=0, atol=1e-15)
        with pytest.raises(ValueError, match="negative"):
            c.forward(-0.1)

    def test_equity_forward_agrees_everywhere(self) -> None:
        c = self.curve()
        ef = c.to_equity_forward()
        t = np.linspace(0.01, 3.0, 57)
        assert np.max(np.abs(ef.forward(t) / c.forward(t) - 1)) < 1e-14
        assert np.allclose(ef.discount(self.times), self.discs, rtol=1e-15)
        # dividend yield = r - carry at the nodes
        assert np.allclose(c.dividend_yield(self.times), R - self.carries, rtol=0, atol=1e-14)

    def test_requires_discounts(self) -> None:
        c = ImpliedForwardCurve(S, [1.0], [102.0])
        assert c.forward(0.5) == pytest.approx(S * math.sqrt(1.02), rel=1e-15)
        with pytest.raises(ValueError, match="without discount"):
            c.to_equity_forward()

    @pytest.mark.parametrize(
        "args",
        [
            (S, [0.0, 1.0], [100.0, 101.0]),
            (S, [1.0, 0.5], [100.0, 101.0]),
            (S, [1.0], [-1.0]),
            (-S, [1.0], [101.0]),
            (S, [1.0, 2.0], [101.0]),
        ],
    )
    def test_validation(self, args: tuple) -> None:
        with pytest.raises(ValueError):
            ImpliedForwardCurve(*args)
        with pytest.raises(ValueError, match="discounts"):
            ImpliedForwardCurve(S, [1.0], [101.0], [0.0])


def test_from_surface_end_to_end() -> None:
    ef = EquityForward(S, FlatCurve(R), dividend_yield=FlatCurve(Q))
    chains = []
    for t in (0.25, 0.5, 1.0):
        f, d = ef.forward(t), ef.discount(t)
        k = np.linspace(80.0, 120.0, 9)
        c = black_price(f, k, t, 0.2, d, "call")
        p = black_price(f, k, t, 0.2, d, "put")
        chains.append(OptionChain.from_arrays(t, k, c, c, p, p))
    curve = ImpliedForwardCurve.from_surface(OptionSurfaceQuotes(chains, spot=S))
    assert np.allclose(curve.forward(curve.times), ef.forward(curve.times), rtol=1e-12)
    assert np.allclose(curve.dividend_yield(curve.times), Q, rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="spot"):
        ImpliedForwardCurve.from_surface(OptionSurfaceQuotes(chains))
