"""Exactness tests: arithmetic Asian options via the Vecer PDE."""

import math

import numpy as np
import pytest

from spxa.pde.asian import AsianPDE
from spxa.pde.reference import bs_price

pytestmark = pytest.mark.exactness

# Linetsky (2004), Table 3: continuously averaged calls, q = 0, K = 2.0
LINETSKY = [
    (2.0, 0.02, 0.10, 1.0, 0.0559860415),
    (2.0, 0.18, 0.30, 1.0, 0.2183875466),
    (2.0, 0.0125, 0.25, 2.0, 0.1722687410),
    (1.9, 0.05, 0.50, 1.0, 0.1931737903),
    (2.0, 0.05, 0.50, 1.0, 0.2464156905),
    (2.1, 0.05, 0.50, 1.0, 0.3062203648),
    (2.0, 0.05, 0.50, 2.0, 0.3500952190),
]


class TestContinuousAveraging:
    @pytest.mark.parametrize("S,r,sigma,T,ref", LINETSKY)
    def test_linetsky_spectral_values(self, S, r, sigma, T, ref):
        coarse = AsianPDE(r, sigma).call(2.0, T, S, n_space=401, n_time=200).price
        fine = AsianPDE(r, sigma).call(2.0, T, S, n_space=801, n_time=400).price
        assert coarse == pytest.approx(ref, abs=1e-4)
        assert abs(fine - ref) < 0.3 * abs(coarse - ref) + 1e-9  # second order
        assert (4 * fine - coarse) / 3 == pytest.approx(ref, abs=2e-7)

    def test_put_call_parity_is_exact(self):
        m = AsianPDE(0.05, 0.3, 0.02)
        c, p = m.call(100.0, 1.0, 100.0), m.put(100.0, 1.0, 100.0)
        q0 = c.solution.info["q0"]
        assert c.price - p.price == pytest.approx(100 * q0 - math.exp(-0.05) * 100, abs=1e-10)
        assert c.delta - p.delta == pytest.approx(q0, abs=1e-10)
        assert c.gamma == pytest.approx(p.gamma, abs=1e-10)

    def test_greeks_match_bumped_prices(self):
        m = AsianPDE(0.05, 0.3)
        res = m.call(100.0, 1.0, 100.0)
        h = 0.5
        up, dn = m.call(100.0, 1.0, 100.0 + h).price, m.call(100.0, 1.0, 100.0 - h).price
        assert res.delta == pytest.approx((up - dn) / (2 * h), abs=1e-4)
        assert res.gamma == pytest.approx((up - 2 * res.price + dn) / h**2, abs=1e-4)

    def test_asian_cheaper_than_european(self):
        m = AsianPDE(0.05, 0.3)
        assert m.call(100.0, 1.0, 100.0).price < float(bs_price(100, 100, 1.0, 0.05, 0.3))


class TestDiscreteAveraging:
    def test_single_fixing_is_european(self):
        m = AsianPDE(0.05, 0.3, 0.02)
        for opt in ("call", "put"):
            pde = m.price(100.0, 1.0, 100.0, opt, fixings=[1.0], n_space=801, n_time=400).price
            ref = float(bs_price(100, 100, 1.0, 0.05, 0.3, 0.02, opt))
            assert pde == pytest.approx(ref, rel=2e-5)

    def test_monthly_against_monte_carlo(self):
        r, q, sig = 0.05, 0.02, 0.3
        fx = np.arange(1, 13) / 12
        pde = AsianPDE(r, sig, q).call(100.0, 1.0, 100.0, fixings=fx).price
        rng = np.random.default_rng(20261030)
        n = 400_000
        z = rng.standard_normal((n, 12))
        log_s = np.log(100.0) + np.cumsum(
            (r - q - 0.5 * sig**2) / 12 + sig * np.sqrt(1 / 12) * z, 1
        )
        pay = math.exp(-r) * np.maximum(np.exp(log_s).mean(1) - 100.0, 0.0)
        # antithetic-free plain estimator, 3.5 standard errors
        assert pde == pytest.approx(pay.mean(), abs=3.5 * pay.std() / math.sqrt(n))

    def test_converges_to_continuous_at_rate_one_over_n(self):
        m = AsianPDE(0.05, 0.3, 0.02)
        cont = m.call(100.0, 1.0, 100.0, n_space=401, n_time=400).price
        errs = []
        for nf in (25, 50, 100):
            fx = np.arange(1, nf + 1) / nf
            errs.append(m.call(100.0, 1.0, 100.0, fixings=fx, n_space=401, n_time=400).price - cont)
        assert all(e > 0 for e in errs)
        ratios = np.array(errs[:-1]) / np.array(errs[1:])
        np.testing.assert_allclose(ratios, 2.0, atol=0.1)

    def test_seasoned_deep_in_the_money_is_forward(self):
        # past fixings already exceed the strike: payoff is linear, price = S0 z0
        m = AsianPDE(0.04, 0.25)
        fx = [0.25, 0.5, 0.75, 1.0]
        res = m.call(100.0, 1.0, 100.0, fixings=fx, past_average=110.0)
        q0 = res.solution.info["q0"]
        assert res.price == pytest.approx(100 * q0 - math.exp(-0.04) * (100.0 - 110.0), abs=1e-8)
        assert res.delta == pytest.approx(q0, abs=1e-8)

    def test_past_average_shifts_strike(self):
        m = AsianPDE(0.04, 0.25)
        fx = [0.5, 1.0]
        a = m.call(100.0, 1.0, 100.0, fixings=fx, past_average=20.0).price
        b = m.call(80.0, 1.0, 100.0, fixings=fx).price
        assert a == pytest.approx(b, abs=1e-12)
