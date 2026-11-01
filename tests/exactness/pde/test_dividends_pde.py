"""Exactness tests: discrete dividends as jump conditions."""

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.pde.black_scholes import BlackScholesPDE, DiscreteDividend
from spxa.pde.reference import bs_price

pytestmark = pytest.mark.exactness


def cash_dividend_call(S, K, T, r, sigma, D, td):
    """Piecewise-lognormal model: integrate the post-dividend BS price over S_{td}."""
    m = np.log(S) + (r - 0.5 * sigma**2) * td
    s = sigma * np.sqrt(td)

    def integrand(z):
        st = np.exp(m + s * z)
        after = max(st - D, 1e-300)
        return float(bs_price(after, K, T - td, r, sigma)) * np.exp(-0.5 * z * z)

    val, _ = quad(integrand, -12, 12, limit=400, epsabs=1e-12)
    return np.exp(-r * td) * val / np.sqrt(2 * np.pi)


class TestDiscreteDividends:
    def test_proportional_dividend_is_spot_scaling(self):
        m = BlackScholesPDE(0.05, 0.25)
        res = m.european(
            100, 1.0, 100, "call", dividends=[DiscreteDividend(0.4, proportional=0.03)]
        )
        assert res.price == pytest.approx(float(bs_price(97.0, 100, 1.0, 0.05, 0.25)), abs=3e-4)

    def test_cash_dividend_against_semi_analytic(self):
        m = BlackScholesPDE(0.05, 0.3)
        D, td = 4.0, 0.5
        res = m.european(100, 1.0, 100, "call", dividends=[DiscreteDividend(td, amount=D)])
        assert res.price == pytest.approx(
            cash_dividend_call(100, 100, 1.0, 0.05, 0.3, D, td), abs=1e-3
        )

    def test_two_dividends_and_put(self):
        m = BlackScholesPDE(0.03, 0.2)
        divs = [
            DiscreteDividend(0.25, proportional=0.02),
            DiscreteDividend(0.75, proportional=0.01),
        ]
        res = m.european(95, 1.0, 100, "put", dividends=divs)
        assert res.price == pytest.approx(
            float(bs_price(100 * 0.98 * 0.99, 95, 1.0, 0.03, 0.2, 0, "put")), abs=3e-4
        )

    def test_dividend_today(self):
        m = BlackScholesPDE(0.05, 0.2)
        res = m.european(100, 1.0, 100, "call", dividends=[DiscreteDividend(0.0, amount=2.0)])
        assert res.price == pytest.approx(float(bs_price(98.0, 100, 1.0, 0.05, 0.2)), abs=3e-4)

    def test_validation(self):
        with pytest.raises(ValueError):
            DiscreteDividend(0.5, amount=-1.0)
        with pytest.raises(ValueError):
            DiscreteDividend(0.5, proportional=1.0)
        with pytest.raises(ValueError):
            BlackScholesPDE(0.05, 0.2).european(
                100, 1.0, 100, dividends=[DiscreteDividend(1.5, 1.0)]
            )
        np.testing.assert_allclose(DiscreteDividend(0.1, 5.0).apply(np.array([3.0, 10.0])), [0, 5])
