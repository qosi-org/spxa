"""Exactness tests: European options under jump diffusions via the PIDE."""

import math

import numpy as np
import pytest

from spxa.pde.pide import LevyPIDE
from spxa.pde.reference import lewis_call_price, merton_price
from spxa.zoo.levy import KouJumpDiffusion, MertonJumpDiffusion

pytestmark = pytest.mark.exactness

# d'Halluin, Forsyth & Vetzal (2005) / Andersen & Andreasen (2000) test case
MERTON = dict(sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
R, T = 0.05, 0.25


@pytest.fixture(scope="module")
def merton() -> LevyPIDE:
    proc = MertonJumpDiffusion(drift=0.0, **MERTON)
    return LevyPIDE.from_process(proc, R)


class TestMerton:
    def test_published_put(self, merton):
        # European put at S = K = 100: 3.149026 (d'Halluin et al., 2005)
        res = merton.european(100.0, T, 100.0, "put", n_space=801, n_time=200)
        assert res.price == pytest.approx(3.149026, abs=2e-4)

    @pytest.mark.parametrize("S", [80.0, 90.0, 100.0, 110.0, 125.0])
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_series_solution(self, merton, S, option):
        ref = merton_price(S, 100.0, T, R, 0.15, 0.1, -0.9, 0.45, option=option)
        res = merton.european(100.0, T, S, option, n_space=801, n_time=200)
        assert res.price == pytest.approx(ref, abs=5e-4)

    def test_second_order_convergence(self, merton):
        ref = merton_price(100.0, 100.0, T, R, 0.15, 0.1, -0.9, 0.45, option="put")
        errs = [
            abs(merton.european(100.0, T, 100.0, "put", n_space=n, n_time=n // 4).price - ref)
            for n in (201, 401, 801)
        ]
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.8)

    def test_greeks_against_series(self, merton):
        def price(s):
            return merton_price(s, 100.0, T, R, 0.15, 0.1, -0.9, 0.45, option="call")

        e = 1e-2
        delta = (price(100 + e) - price(100 - e)) / (2 * e)
        gamma = (price(100 + e) - 2 * price(100.0) + price(100 - e)) / e**2
        res = merton.european(100.0, T, 100.0, "call", n_space=1601, n_time=400)
        assert res.delta == pytest.approx(delta, abs=2e-5)
        assert res.gamma == pytest.approx(gamma, abs=2e-5)

    def test_dividend_yield(self):
        proc = MertonJumpDiffusion(drift=0.0, **MERTON)
        P = LevyPIDE.from_process(proc, R, q=0.03)
        ref = merton_price(100.0, 95.0, 1.0, R, 0.15, 0.1, -0.9, 0.45, q=0.03, option="call")
        assert P.european(95.0, 1.0, 100.0, "call", n_space=801).price == pytest.approx(
            ref, abs=3e-4
        )


class TestKou:
    kou = KouJumpDiffusion(drift=0.0, sigma=0.16, rate=1.0, p=0.4, eta1=10.0, eta2=5.0)

    def _cf(self, T):
        om = np.log(self.kou.characteristic_function(-1j, 1.0)).real

        def cf(u):
            return self.kou.characteristic_function(u, T) * np.exp(-1j * u * om * T)

        return cf

    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_closed_form_calls(self, K):
        ref = float(self.kou.call_price([K], T=1.0, S0=100.0, r=0.05)[0])
        res = LevyPIDE.from_process(self.kou, 0.05).european(K, 1.0, 100.0, n_space=801)
        assert res.price == pytest.approx(ref, abs=1.5e-4)

    def test_puts_and_greeks_against_lewis(self):
        cf = self._cf(1.0)
        P = LevyPIDE.from_process(self.kou, 0.05)
        res = P.european(100.0, 1.0, 100.0, "put", n_space=801)
        assert res.price == pytest.approx(
            lewis_call_price(cf, 100.0, 100.0, 1.0, 0.05, 0.0, "put"), abs=2e-4
        )
        e = 1e-2
        up, mid, dn = (
            lewis_call_price(cf, s, 100.0, 1.0, 0.05, 0.0, "put") for s in (100 + e, 100.0, 100 - e)
        )
        assert res.delta == pytest.approx((up - dn) / (2 * e), abs=2e-5)
        assert res.gamma == pytest.approx((up - 2 * mid + dn) / e**2, abs=5e-6)
        assert math.isfinite(res.theta)
