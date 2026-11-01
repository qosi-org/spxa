"""Exactness tests for binomial trees and the perpetual American put."""

import pytest

from spxa.pde.reference import (
    binomial_price,
    bs_price,
    perpetual_american_put,
    perpetual_put_boundary,
)

pytestmark = pytest.mark.exactness


class TestBinomial:
    def test_leisen_reimer_european_accuracy(self):
        for option in ("call", "put"):
            tree = binomial_price(100, 105, 1.0, 0.05, 0.25, 0.02, option, False, 501, "lr")
            exact = float(bs_price(100, 105, 1.0, 0.05, 0.25, 0.02, option))
            assert tree == pytest.approx(exact, abs=2e-6)

    def test_leisen_reimer_second_order(self):
        exact = float(bs_price(100, 105, 1.0, 0.05, 0.25, 0.0, "put"))
        errs = [
            abs(binomial_price(100, 105, 1.0, 0.05, 0.25, 0.0, "put", False, n) - exact)
            for n in (51, 101, 201)
        ]
        assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5

    def test_crr_converges(self):
        exact = float(bs_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call"))
        assert binomial_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", False, 2000, "crr") == (
            pytest.approx(exact, abs=2e-3)
        )

    def test_american_put_benchmarks(self):
        # Longstaff & Schwartz (2001), Table 1 (K=40, r=0.06, sigma=0.2, T=1):
        # finite-difference values 4.478, 2.314, 1.110
        vals = [binomial_price(s, 40, 1.0, 0.06, 0.2, 0.0, "put", True, 2001) for s in (36, 40, 44)]
        for v, ref in zip(vals, (4.478, 2.314, 1.110)):
            assert v == pytest.approx(ref, abs=1e-2)
        # LR and CRR agree on the American put
        crr = binomial_price(36, 40, 1.0, 0.06, 0.2, 0.0, "put", True, 4000, "crr")
        assert vals[0] == pytest.approx(crr, abs=5e-4)

    def test_american_call_without_dividends_is_european(self):
        am = binomial_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", True, 501)
        eu = binomial_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", False, 501)
        assert am == pytest.approx(eu, abs=1e-12)
        assert binomial_price(100, 100, 1.0, 0.05, 0.2, 0.08, "call", True, 501) > (
            binomial_price(100, 100, 1.0, 0.05, 0.2, 0.08, "call", False, 501)
        )

    def test_validation(self):
        with pytest.raises(ValueError):
            binomial_price(100, 100, 1.0, 0.05, 0.2, tree="trinomial")
        with pytest.raises(ValueError):
            binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=0)


class TestPerpetualPut:
    def test_long_dated_tree_converges(self):
        perp = perpetual_american_put(100, 100, 0.05, 0.3)
        tree = binomial_price(100, 100, 100.0, 0.05, 0.3, 0.0, "put", True, 8001, "lr")
        assert tree == pytest.approx(perp, abs=1e-2)
        assert tree < perp

    def test_smooth_pasting(self):
        K, r, sig, q = 100.0, 0.05, 0.3, 0.02
        s_star = perpetual_put_boundary(K, r, sig, q)
        h = 1e-5
        slope = (perpetual_american_put(s_star + h, K, r, sig, q) - (K - s_star)) / h
        assert slope == pytest.approx(-1.0, abs=1e-4)
        assert perpetual_american_put(0.5 * s_star, K, r, sig, q) == pytest.approx(K - 0.5 * s_star)

    def test_closed_form_without_dividends(self):
        # q = 0: beta = -2r/sigma^2, S* = 2rK / (2r + sigma^2)
        K, r, sig = 100.0, 0.05, 0.3
        assert perpetual_put_boundary(K, r, sig) == pytest.approx(2 * r * K / (2 * r + sig**2))
