"""Exactness tests for the Black-Scholes reference formulas."""

import numpy as np
import pytest

from spxa.pde.reference import (
    bs_asset_or_nothing_price,
    bs_digital_price,
    bs_greeks,
    bs_price,
)

pytestmark = pytest.mark.exactness

S, K, T, R, SIG, Q = 100.0, 95.0, 0.75, 0.04, 0.3, 0.015


class TestBlackScholes:
    def test_textbook_value(self):
        # Hull, Options Futures and Other Derivatives, Example 15.6
        assert bs_price(42, 40, 0.5, 0.1, 0.2, 0.0, "call") == pytest.approx(4.7594, abs=1e-4)
        assert bs_price(42, 40, 0.5, 0.1, 0.2, 0.0, "put") == pytest.approx(0.8086, abs=1e-4)

    def test_put_call_parity(self):
        strikes = np.linspace(60, 140, 9)
        c = bs_price(S, strikes, T, R, SIG, Q, "call")
        p = bs_price(S, strikes, T, R, SIG, Q, "put")
        np.testing.assert_allclose(c - p, S * np.exp(-Q * T) - strikes * np.exp(-R * T))

    @pytest.mark.parametrize("option", ["call", "put"])
    def test_greeks_match_finite_differences(self, option):
        g = bs_greeks(S, K, T, R, SIG, Q, option)
        h = 1e-3

        def f(**kw):
            args = dict(S=S, K=K, T=T, r=R, sigma=SIG, q=Q)
            args.update(kw)
            return float(bs_price(option=option, **args))

        assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-7)
        assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
        assert g["vega"] == pytest.approx((f(sigma=SIG + h) - f(sigma=SIG - h)) / (2 * h), rel=1e-6)
        assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-6)
        assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)
        assert g["price"] == pytest.approx(f())

    def test_digital_decomposition(self):
        call = bs_price(S, K, T, R, SIG, Q, "call")
        asset = bs_asset_or_nothing_price(S, K, T, R, SIG, Q, "call")
        cash = bs_digital_price(S, K, T, R, SIG, Q, "call")
        assert call == pytest.approx(asset - K * cash)
        put = bs_price(S, K, T, R, SIG, Q, "put")
        assert put == pytest.approx(
            K * bs_digital_price(S, K, T, R, SIG, Q, "put")
            - bs_asset_or_nothing_price(S, K, T, R, SIG, Q, "put")
        )
        total = cash + bs_digital_price(S, K, T, R, SIG, Q, "put")
        assert total == pytest.approx(np.exp(-R * T))
        assert bs_digital_price(S, K, T, R, SIG, Q, "call", cash=3.0) == pytest.approx(3 * cash)

    def test_validation(self):
        with pytest.raises(ValueError):
            bs_price(S, K, 0.0, R, SIG)
        with pytest.raises(ValueError):
            bs_price(S, K, T, R, -0.1)
        with pytest.raises(ValueError):
            bs_price(S, K, T, R, SIG, option="chooser")
