"""Exactness tests: short-rate closed forms in spxa.pde.reference."""

import math

import numpy as np
import pytest

from spxa.pde import reference as R
from spxa.zoo.beyond import CIRProcess, OrnsteinUhlenbeckProcess

pytestmark = pytest.mark.exactness


class TestBondFormulas:
    @pytest.mark.parametrize("tau", [0.5, 2.0, 10.0])
    def test_against_zoo(self, tau):
        cir = CIRProcess(kappa=0.3, theta=0.05, sigma=0.1, r0=0.04)
        ou = OrnsteinUhlenbeckProcess(theta=0.3, mu=0.05, sigma=0.02, x0=0.04)
        assert R.cir_bond(0.04, tau, 0.3, 0.05, 0.1) == pytest.approx(
            cir.bond_price(tau), rel=1e-12
        )
        assert R.vasicek_bond(0.04, tau, 0.3, 0.05, 0.02) == pytest.approx(
            ou.bond_price(tau), rel=1e-12
        )

    def test_zero_maturity_and_vectorisation(self):
        assert R.vasicek_bond(0.03, 0.0, 0.3, 0.05, 0.02) == 1.0
        np.testing.assert_array_equal(R.cir_bond(np.array([0.01, 0.02]), 0.0, 0.3, 0.05, 0.1), 1.0)
        v = R.vasicek_bond(np.array([0.01, 0.05]), 3.0, 0.3, 0.05, 0.02)
        assert v.shape == (2,) and v[0] > v[1]

    def test_small_volatility_limit(self):
        # deterministic rate path r(t) = theta + (r0 - theta) e^{-kappa t}
        k, th, r0, tau = 0.4, 0.06, 0.02, 3.0
        integral = th * tau + (r0 - th) * (1 - math.exp(-k * tau)) / k
        # convexity corrections are O(sigma^2 tau^3); the CIR exponent loses
        # ~1e-16 / sigma^2 to cancellation, so sigma = 1e-4 balances both
        assert R.vasicek_bond(r0, tau, k, th, 1e-4) == pytest.approx(math.exp(-integral), rel=2e-6)
        assert R.cir_bond(r0, tau, k, th, 1e-4) == pytest.approx(math.exp(-integral), rel=2e-6)


class TestBondOptionFormulas:
    @pytest.mark.parametrize("K", [0.75, 0.8, 0.85])
    def test_put_call_parity(self, K):
        for f, bond, s in (
            (R.vasicek_bond_option, R.vasicek_bond, 0.02),
            (R.cir_bond_option, R.cir_bond, 0.1),
        ):
            c = f(0.04, K, 1.0, 5.0, 0.3, 0.05, s, "call")
            p = f(0.04, K, 1.0, 5.0, 0.3, 0.05, s, "put")
            fwd = bond(0.04, 5.0, 0.3, 0.05, s) - K * bond(0.04, 1.0, 0.3, 0.05, s)
            assert c - p == pytest.approx(fwd, abs=1e-13)
            assert c > 0 and p > 0

    @pytest.mark.parametrize("model", ["vasicek", "cir"])
    def test_against_monte_carlo(self, model):
        k, th, r0, T, S, K = 0.3, 0.05, 0.04, 1.0, 5.0, 0.8
        s = 0.02 if model == "vasicek" else 0.1
        bond = R.vasicek_bond if model == "vasicek" else R.cir_bond
        price = (R.vasicek_bond_option if model == "vasicek" else R.cir_bond_option)(
            r0, K, T, S, k, th, s
        )
        rng = np.random.default_rng(7)
        n_paths, n_steps = 60_000, 100
        dt = T / n_steps
        r = np.full(n_paths, r0)
        integral = np.zeros(n_paths)
        for _ in range(n_steps):
            vol = s if model == "vasicek" else s * np.sqrt(np.maximum(r, 0.0))
            r_new = r + k * (th - r) * dt + vol * math.sqrt(dt) * rng.standard_normal(n_paths)
            if model == "cir":
                r_new = np.maximum(r_new, 0.0)
            integral += 0.5 * (r + r_new) * dt
            r = r_new
        pay = np.exp(-integral) * np.maximum(bond(r, S - T, k, th, s) - K, 0.0)
        assert price == pytest.approx(pay.mean(), abs=4 * pay.std() / math.sqrt(n_paths))
