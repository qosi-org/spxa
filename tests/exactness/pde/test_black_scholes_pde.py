"""Exactness tests: Black-Scholes PDE pricer against closed forms."""

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.convergence import observed_order
from spxa.pde.reference import bs_digital_price, bs_greeks, bs_price

pytestmark = pytest.mark.exactness

CASES = [
    # S, K, T, r, sigma, q, option
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call"),
    (97.0, 110.0, 0.5, 0.05, 0.2, 0.0, "put"),
    (100.0, 100.0, 0.1, 0.03, 0.2, 0.0, "call"),
    (100.0, 120.0, 5.0, 0.03, 0.4, 0.01, "put"),
    (100.0, 80.0, 0.25, 0.05, 0.6, 0.0, "call"),
    (42.0, 40.0, 0.5, 0.1, 0.2, 0.0, "call"),
    (100.0, 90.0, 2.0, 0.0, 0.3, 0.05, "put"),
]


class TestEuropean:
    @pytest.mark.parametrize("S,K,T,r,sigma,q,option", CASES)
    def test_price_and_grid_greeks(self, S, K, T, r, sigma, q, option):
        res = BlackScholesPDE(r, sigma, q).european(K, T, S, option)
        ref = bs_greeks(S, K, T, r, sigma, q, option)
        assert res.price == pytest.approx(ref["price"], rel=1e-4, abs=2e-4)
        assert res.delta == pytest.approx(ref["delta"], abs=2e-4)
        assert res.gamma == pytest.approx(ref["gamma"], rel=1e-3)
        assert res.theta == pytest.approx(ref["theta"], rel=5e-3, abs=2e-3)

    def test_bump_vega_and_rho(self):
        res = BlackScholesPDE(0.05, 0.25, 0.01).european(105, 1.0, 100, "put", bump_greeks=True)
        ref = bs_greeks(100, 105, 1.0, 0.05, 0.25, 0.01, "put")
        assert res.vega == pytest.approx(ref["vega"], rel=1e-3)
        assert res.rho == pytest.approx(ref["rho"], rel=1e-3)
        assert set(res.as_dict()) == {"price", "delta", "gamma", "theta", "vega", "rho"}

    def test_whole_curve(self):
        res = BlackScholesPDE(0.05, 0.2).european(100, 1.0, 100, "call")
        s = np.linspace(40.0, 250.0, 43)
        np.testing.assert_allclose(
            res.solution.value(s), bs_price(s, 100, 1.0, 0.05, 0.2), rtol=2e-5, atol=5e-4
        )

    def test_second_order_convergence_price_and_gamma(self):
        model = BlackScholesPDE(0.05, 0.2)
        ref = bs_greeks(100, 100, 1.0, 0.05, 0.2)
        e_p, e_g = [], []
        for k in range(4):
            res = model.european(100, 1.0, 100, n_space=50 * 2**k + 1, n_time=25 * 2**k)
            e_p.append(res.price - ref["price"])
            e_g.append(res.gamma - ref["gamma"])
        # the strike-aligned grid family varies slightly with n, so judge the
        # order over the whole refinement range and bound the per-level orders
        overall = np.log2(abs(e_p[0] / e_p[-1])) / 3
        assert abs(overall - 2.0) < 0.1
        assert np.all(observed_order(e_p) > 1.6)
        assert np.all(np.abs(observed_order(e_g) - 2.0) < 0.15)

    def test_put_call_parity_on_grid(self):
        m = BlackScholesPDE(0.04, 0.3, 0.02)
        c = m.european(100, 1.5, 95, "call").price
        p = m.european(100, 1.5, 95, "put").price
        assert c - p == pytest.approx(
            95 * np.exp(-0.02 * 1.5) - 100 * np.exp(-0.04 * 1.5), abs=2e-4
        )


class TestDigital:
    def test_price_with_smoothing(self):
        res = BlackScholesPDE(0.05, 0.2).digital(100, 1.0, 100, "call", cash=10.0)
        assert res.price == pytest.approx(
            bs_digital_price(100, 100, 1.0, 0.05, 0.2, 0, "call", 10), abs=2e-4
        )

    def test_second_order_despite_discontinuity(self):
        model = BlackScholesPDE(0.05, 0.2)
        ref = bs_digital_price(100, 100, 1.0, 0.05, 0.2)
        errs = [
            model.digital(100, 1.0, 100, n_space=50 * 2**k + 1, n_time=25 * 2**k).price - ref
            for k in range(4)
        ]
        assert np.all(observed_order(errs) > 1.7)


class TestLocalVolatility:
    def test_constant_callable_matches_constant(self):
        a = BlackScholesPDE(0.05, 0.2).european(100, 1.0, 100).price
        b = BlackScholesPDE(0.05, lambda s, t: 0.2 + 0.0 * s).european(100, 1.0, 100).price
        assert a == pytest.approx(b, abs=1e-10)

    def test_time_dependent_volatility(self):
        # deterministic sigma(t): Black-Scholes with the average variance
        T = 2.0
        sig = lambda s, t: (0.15 + 0.1 * t) + 0.0 * s  # noqa: E731
        avg_var = (0.15**2 * T + 0.15 * 0.1 * T**2 + 0.01 * T**3 / 3) / T
        res = BlackScholesPDE(0.04, sig, 0.01).european(100, T, 100)
        assert res.price == pytest.approx(
            bs_price(100, 100, T, 0.04, np.sqrt(avg_var), 0.01), abs=5e-4
        )

    def test_validation(self):
        with pytest.raises(ValueError):
            BlackScholesPDE(0.05, -0.2)
        m = BlackScholesPDE(0.05, 0.2)
        with pytest.raises(ValueError):
            m.european(100, 1.0, -5.0)
        with pytest.raises(ValueError):
            m.european(100, 1.0, 100, grid="chebyshev")
        with pytest.raises(ValueError):
            m.european(100, 1.0, 100, s_min=10.0)
        with pytest.raises(ValueError):
            m.european(100, 1.0, 100, s_max=50.0)
        assert "European call" in m.european(100, 1.0, 100).__story__()
