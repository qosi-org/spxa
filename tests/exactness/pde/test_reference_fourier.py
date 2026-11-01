"""Exactness tests for the Lewis Fourier formula and Merton's series."""

import numpy as np
import pytest

from spxa.pde.reference import (
    bs_cf,
    bs_price,
    heston_cf,
    levy_cf,
    lewis_call_price,
    merton_price,
)
from spxa.zoo.beyond import HestonProcess
from spxa.zoo.levy import MertonJumpDiffusion

pytestmark = pytest.mark.exactness


class TestLewis:
    @pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 125.0, 180.0])
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_reproduces_black_scholes(self, K, option):
        got = lewis_call_price(bs_cf(0.25, 0.8), 100.0, K, 0.8, 0.04, 0.01, option)
        assert got == pytest.approx(
            float(bs_price(100.0, K, 0.8, 0.04, 0.25, 0.01, option)), abs=1e-10
        )

    def test_heston_matches_zoo(self):
        # in 't Hout & Foulon (2010) case 1 parameters
        proc = HestonProcess(
            v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.9, mu=0.025, s0=100
        )
        cf = heston_cf(1.0, 0.04, 1.5, 0.04, 0.3, -0.9)
        for K in (80.0, 100.0, 120.0):
            assert lewis_call_price(cf, 100.0, K, 1.0, 0.025) == pytest.approx(
                proc.call_price(K, T=1.0, r=0.025), abs=1e-5
            )

    def test_heston_degenerates_to_black_scholes(self):
        # sigma_v -> 0 with v0 = theta: constant variance
        cf = heston_cf(1.0, 0.04, 2.0, 0.04, 1e-3, 0.0)
        got = lewis_call_price(cf, 100.0, 100.0, 1.0, 0.03)
        assert got == pytest.approx(float(bs_price(100.0, 100.0, 1.0, 0.03, 0.2)), abs=1e-5)


class TestMerton:
    PARAMS = dict(sigma=0.15, lam=0.1, jump_mean=-0.9, jump_std=0.45)

    def test_series_matches_fourier_with_zoo_char_func(self):
        proc = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
        cf = levy_cf(proc, 0.25)
        for K in (80.0, 100.0, 120.0):
            for option in ("call", "put"):
                series = merton_price(100.0, K, 0.25, 0.05, option=option, **self.PARAMS)
                fourier = lewis_call_price(cf, 100.0, K, 0.25, 0.05, 0.0, option)
                assert series == pytest.approx(fourier, abs=1e-10)

    def test_no_jumps_is_black_scholes(self):
        got = merton_price(100, 100, 1.0, 0.05, 0.2, 0.0, -0.1, 0.1, 0.02, "call")
        assert got == pytest.approx(float(bs_price(100, 100, 1.0, 0.05, 0.2, 0.02, "call")))

    def test_put_call_parity(self):
        c = merton_price(100, 95, 0.5, 0.03, 0.2, 1.0, -0.1, 0.15, 0.01, "call")
        p = merton_price(100, 95, 0.5, 0.03, 0.2, 1.0, -0.1, 0.15, 0.01, "put")
        assert c - p == pytest.approx(100 * np.exp(-0.01 * 0.5) - 95 * np.exp(-0.03 * 0.5))


class TestLevyCF:
    def test_martingale_normalisation(self):
        proc = MertonJumpDiffusion(drift=0.3, sigma=0.2, rate=1.0, jump_mean=-0.1, jump_std=0.15)
        cf = levy_cf(proc, 2.0)
        assert abs(cf(-1j) - 1.0) < 1e-12
        assert abs(cf(0.0) - 1.0) < 1e-12

    def test_rejects_infinite_exponential_moment(self):
        class Bad:
            def char_func(self, u, t=1.0):
                return -1.0 + 0j

        with pytest.raises(ValueError):
            levy_cf(Bad(), 1.0)
