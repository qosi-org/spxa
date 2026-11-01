"""Unit tests for Dupire helpers: implied volatility and local-vol extraction."""

import math

import numpy as np
import pytest

from spxa.pde.local_vol import DupirePDE, DupireSurface, implied_volatility, local_vol_from_prices
from spxa.pde.reference import bs_price


class TestImpliedVolatility:
    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.8, 2.0])
    @pytest.mark.parametrize("K", [75.0, 100.0, 130.0])
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_round_trip(self, vol, K, option):
        p = float(bs_price(100.0, K, 1.0, 0.03, vol, 0.01, option))
        assert implied_volatility(p, 100.0, K, 1.0, 0.03, 0.01, option) == pytest.approx(
            vol, abs=1e-8
        )

    def test_low_volatility_at_the_money(self):
        p = float(bs_price(100.0, 100.0, 1.0, 0.0, 0.02))
        assert implied_volatility(p, 100.0, 100.0, 1.0, 0.0) == pytest.approx(0.02, abs=1e-9)

    def test_outside_no_arbitrage_bounds(self):
        assert math.isnan(implied_volatility(120.0, 100.0, 100.0, 1.0, 0.0))
        assert math.isnan(implied_volatility(0.0, 100.0, 150.0, 1.0, 0.0))
        assert math.isnan(implied_volatility(-1.0, 100.0, 100.0, 1.0, 0.0))


class TestLocalVolFromPrices:
    def test_constant_volatility_surface(self):
        K = np.linspace(50.0, 200.0, 301)
        T = np.linspace(0.5, 1.5, 21)
        C = np.array([bs_price(100.0, K, t, 0.04, 0.25, 0.01) for t in T])
        lv = local_vol_from_prices(K, T, C, 0.04, 0.01)
        # away from the far wings, where C_KK underflows, the three-point
        # stencils are accurate to O(h^2) ~ 3e-4 in volatility
        m = (K >= 70.0) & (K <= 150.0)
        np.testing.assert_allclose(lv[1:-1][:, m], 0.25, atol=5e-4)
        assert np.all(np.isnan(lv[0])) and np.all(np.isnan(lv[:, -1]))

    def test_arbitrage_gives_nan(self):
        K = np.linspace(80.0, 120.0, 11)
        T = np.array([0.5, 1.0, 1.5])
        C = np.vstack([np.maximum(100.0 - K, 0.0) - 0.1 * (K - 100.0) ** 2 / 100] * 3)
        lv = local_vol_from_prices(K, T, C, 0.0)
        assert np.all(np.isnan(lv))

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            local_vol_from_prices(np.arange(5.0), np.arange(3.0), np.zeros((5, 3)), 0.0)
        with pytest.raises(ValueError):
            local_vol_from_prices(np.arange(2.0), np.arange(3.0), np.zeros((3, 2)), 0.0)


class TestDupireInterface:
    def test_surface_access(self):
        s = DupirePDE(0.05, 0.2).solve(100.0, 1.0, [0.5], n_space=101, n_time=50)
        assert isinstance(s, DupireSurface)
        assert s.prices.shape == (2, 101)
        np.testing.assert_allclose(s.maturities, [0.5, 1.0])
        k = np.array([90.0, 110.0])
        np.testing.assert_allclose(
            s.put(k, 1.0), s.call(k, 1.0) - 100.0 + k * math.exp(-0.05), atol=1e-12
        )
        with pytest.raises(ValueError):
            s.call(100.0, 0.75)
        assert isinstance(s.call(100.0, 1.0), float)

    def test_validation_and_repr(self):
        with pytest.raises(ValueError):
            DupirePDE(0.05, -0.2)
        d = DupirePDE(0.05, lambda k, t: 0.2 + 0 * k)
        with pytest.raises(ValueError):
            d.solve(100.0, 1.0, [1.5])
        assert "callable" in repr(d)
        assert not d.pde().time_homogeneous
        assert DupirePDE(0.05, 0.2).pde().time_homogeneous
