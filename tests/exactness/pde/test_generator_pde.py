"""Exactness tests: zoo processes solved through spxa.pde.to_pde."""

import math

import numpy as np
import pytest

from spxa.pde.generator import to_pde
from spxa.pde.payoffs import vanilla
from spxa.zoo.beyond import CEVProcess, CIRProcess, OrnsteinUhlenbeckProcess
from spxa.zoo.levy import BrownianMotion

pytestmark = pytest.mark.exactness


class TestOrnsteinUhlenbeck:
    ou = OrnsteinUhlenbeckProcess(theta=1.5, mu=0.5, sigma=0.4, x0=0.0)

    def test_second_moment(self):
        sol = to_pde(self.ou).expectation(lambda x: x**2, 1.0)
        m = 0.5 * (1 - math.exp(-1.5))
        v = 0.16 / 3 * (1 - math.exp(-3.0))
        assert sol.value(0.0) == pytest.approx(m * m + v, abs=5e-6)

    def test_density_matches_zoo(self):
        den = to_pde(self.ou).density(1.0)
        y = np.array([-0.2, 0.0, 0.3, 0.6, 0.9])
        np.testing.assert_allclose(den.pdf(y), self.ou.transition_density(y, 1.0), atol=3e-4)
        assert den.mass() == pytest.approx(1.0, abs=1e-12)


class TestCIR:
    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0])
    def test_bond_with_state_discount(self, T):
        cir = CIRProcess(kappa=0.8, theta=0.05, sigma=0.15, r0=0.03)
        sol = to_pde(cir).expectation(lambda x: np.ones_like(x), T, discount="state")
        assert sol.value(0.03) == pytest.approx(cir.bond_price(T), abs=1e-6)

    def test_density_matches_zoo(self):
        cir = CIRProcess(kappa=0.8, theta=0.05, sigma=0.15, r0=0.03)
        den = to_pde(cir).density(1.0, n_space=801, n_time=400)
        y = np.array([0.01, 0.03, 0.05, 0.08])
        np.testing.assert_allclose(den.pdf(y), cir.transition_density(y, 1.0), rtol=5e-4, atol=2e-3)


class TestCEV:
    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_call_closed_form(self, K):
        cev = CEVProcess(mu=0.0, sigma=2.0, gamma=0.5, s0=100.0, r=0.05, q=0.0)
        sol = to_pde(cev).expectation(vanilla(K), 1.0, discount=0.05, n_space=801, n_time=400)
        assert sol.value(100.0) == pytest.approx(cev.call_price(K, 1.0), abs=1e-4)


class TestBrownianMotion:
    def test_exponential_moment(self):
        sol = to_pde(BrownianMotion(mu=0.1, sigma=0.3)).expectation(np.exp, 1.0)
        assert sol.value(0.0) == pytest.approx(math.exp(0.1 + 0.045), rel=5e-6)

    def test_discounted_digital_has_aligned_kink(self):
        from scipy.stats import norm

        from spxa.pde.payoffs import digital

        sol = to_pde(BrownianMotion(mu=0.0, sigma=1.0)).expectation(
            digital(0.5), 1.0, discount=0.02
        )
        assert sol.value(0.0) == pytest.approx(math.exp(-0.02) * norm.sf(0.5), abs=2e-5)


class TestLevyBridge:
    def test_merton_through_to_pde(self):
        from spxa.pde.reference import merton_price
        from spxa.zoo.levy import MertonJumpDiffusion

        proc = MertonJumpDiffusion(drift=0.3, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
        # the physical drift is irrelevant: the PIDE is risk neutral
        res = to_pde(proc, r=0.05).european(100.0, 0.25, 100.0, "put", n_space=801)
        ref = merton_price(100.0, 100.0, 0.25, 0.05, 0.15, 0.1, -0.9, 0.45, option="put")
        assert res.price == pytest.approx(ref, abs=2e-4)
