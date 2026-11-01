"""Exactness tests: Dupire's forward equation."""

import math

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.local_vol import DupirePDE
from spxa.pde.reference import bs_price
from spxa.zoo.beyond import CEVProcess

pytestmark = pytest.mark.exactness

STRIKES = np.array([70.0, 90.0, 100.0, 110.0, 140.0])


def smile(K, t):
    return 0.15 + 0.1 * np.exp(-np.asarray(K) / 100.0) + 0.05 * t


class TestForwardEquation:
    @pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
    def test_constant_vol_surface_is_black_scholes(self, T):
        s = DupirePDE(0.05, 0.2, 0.02).solve(100.0, 2.0, [0.5, 1.0], n_space=801, n_time=400)
        ref = bs_price(100.0, STRIKES, T, 0.05, 0.2, 0.02)
        np.testing.assert_allclose(s.call(STRIKES, T), ref, atol=2e-4)
        np.testing.assert_allclose(s.implied_vol(STRIKES[1:4], T), 0.2, atol=5e-5)

    def test_second_order_convergence(self):
        ref = bs_price(100.0, STRIKES, 1.0, 0.05, 0.2, 0.02)
        errs = []
        for n in (201, 401, 801):
            s = DupirePDE(0.05, 0.2, 0.02).solve(100.0, 1.0, n_space=n, n_time=n // 2)
            errs.append(np.abs(s.call(STRIKES, 1.0) - ref).max())
        assert np.all(np.log2(np.array(errs[:-1]) / np.array(errs[1:])) > 1.9)

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_duality_with_backward_local_vol_pde(self, K):
        """One forward solve reproduces every backward solve at the spot."""
        fwd = DupirePDE(0.03, smile, 0.01).solve(100.0, 1.0, [0.5], n_space=801, n_time=400)
        bwd = BlackScholesPDE(0.03, smile, 0.01)
        for T in (0.5, 1.0):
            back = bwd.european(K, T, 100.0, n_space=801, n_time=400).price
            # the backward grid is centred on K, so it is the less accurate of
            # the two away from the money (1.3e-4 at K = 80 against 1e-5 forward)
            assert fwd.call(K, T) == pytest.approx(back, abs=2e-4)

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_cev_closed_form(self, K):
        cev = CEVProcess(mu=0.0, sigma=2.0, gamma=0.5, s0=100.0, r=0.05, q=0.0)
        s = DupirePDE(0.05, lambda k, t: 2.0 * np.maximum(k, 1e-12) ** -0.5).solve(
            100.0, 1.0, n_space=801, n_time=400
        )
        assert s.call(K, 1.0) == pytest.approx(cev.call_price(K, 1.0), abs=2e-4)

    def test_no_arbitrage_shape(self):
        s = DupirePDE(0.03, smile, 0.01).solve(100.0, 1.0, [0.25, 0.5])
        for T in (0.25, 0.5, 1.0):
            c = s.call(s.strikes, T)
            assert c[0] == pytest.approx(100.0 * math.exp(-0.01 * T), abs=1e-6)
            assert np.all(np.diff(c) <= 1e-12)  # decreasing in strike
            slopes = np.diff(c) / np.diff(s.strikes)
            assert np.all(np.diff(slopes) >= -1e-9)  # convex in strike
        # calendar monotonicity for q = 0 is not guaranteed; with r >= q it is for
        # forward-moneyness, so check total variance increases at the money
        v = [s.implied_vol(100.0, T) ** 2 * T for T in (0.25, 0.5, 1.0)]
        assert v[0] < v[1] < v[2]


class TestLocalVolRoundTrip:
    def test_extracts_the_input_local_vol(self):
        def sig(K, t):
            return 0.2 + 0.1 * np.exp(-(((np.asarray(K) - 100.0) / 30.0) ** 2)) * (1.0 + t)

        mats = np.linspace(0.05, 1.0, 20)
        s = DupirePDE(0.03, sig, 0.01).solve(100.0, 1.0, list(mats), n_space=801, n_time=800)
        lv = s.local_vol()
        m = (s.strikes > 70.0) & (s.strikes < 140.0)
        for j in range(9, 19):
            err = np.abs(lv[j, m] - sig(s.strikes[m], mats[j]))
            assert err.max() < 1e-3
