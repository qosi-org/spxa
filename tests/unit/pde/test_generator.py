"""Unit tests for the process-to-PDE bridge."""

import math

import numpy as np
import pytest

from spxa.pde.boundary import Linear, Natural
from spxa.pde.generator import DiffusionPDE, to_pde
from spxa.pde.pide import LevyPIDE
from spxa.zoo.beyond import CIRProcess, HestonProcess, OrnsteinUhlenbeckProcess
from spxa.zoo.levy import BrownianMotion, MertonJumpDiffusion, VarianceGamma


class TestToPde:
    def test_explicit_coefficients(self):
        d = to_pde(drift=0.1, volatility=0.2, x0=1.0)
        assert isinstance(d, DiffusionPDE)
        assert d.time_homogeneous
        assert "x0=1" in repr(d)
        with pytest.raises(TypeError):
            to_pde(drift=0.1)

    def test_process_mapping(self):
        ou = to_pde(OrnsteinUhlenbeckProcess(theta=2.0, mu=0.1, sigma=0.3, x0=0.5))
        np.testing.assert_allclose(
            ou.pde().coefficients(np.array([0.5]), 0.0)[1], 2.0 * (0.1 - 0.5)
        )
        cir = to_pde(CIRProcess(kappa=0.5, theta=0.04, sigma=0.1, r0=0.03))
        assert cir.lower == 0.0 and cir.x0 == 0.03
        bm = to_pde(BrownianMotion(mu=0.1, sigma=0.3))
        assert bm.drift == 0.1 and bm.volatility == 0.3

    def test_overrides(self):
        d = to_pde(BrownianMotion(mu=0.0, sigma=1.0), x0=2.0, domain=lambda T: (-5.0, 9.0))
        assert d.truncated_domain(1.0) == (-5.0, 9.0)

    def test_unsupported_processes(self):
        merton = MertonJumpDiffusion(drift=0.0, sigma=0.2, rate=0.5, jump_mean=-0.1, jump_std=0.1)
        with pytest.raises(ValueError, match="jumps"):
            to_pde(merton)
        with pytest.raises(TypeError, match="unexpected"):
            to_pde(merton, r=0.05, domain=lambda T: (0.0, 1.0))
        with pytest.raises(NotImplementedError):
            to_pde(
                HestonProcess(
                    v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.3, rho=-0.5, mu=0.0, s0=100.0
                )
            )

    def test_jump_process_gives_pide(self):
        merton = MertonJumpDiffusion(drift=0.0, sigma=0.2, rate=0.5, jump_mean=-0.1, jump_std=0.1)
        pide = to_pde(merton, r=0.05, q=0.01)
        assert isinstance(pide, LevyPIDE)
        assert (pide.r, pide.q, pide.sigma) == (0.05, 0.01, pytest.approx(0.2))
        vg = to_pde(VarianceGamma(sigma=0.12, nu=0.2, theta=-0.14), r=0.03)
        assert isinstance(vg, LevyPIDE) and vg.sigma == 0.0


class TestDiffusionPDE:
    def test_default_domain_and_boundaries(self):
        d = DiffusionPDE(drift=0.0, volatility=0.5, x0=0.0)
        lo, hi = d.truncated_domain(1.0)
        assert lo == pytest.approx(-4.0) and hi == pytest.approx(4.0)
        assert all(isinstance(b, Linear) for b in d._boundaries(lo, hi))
        cir = to_pde(CIRProcess(kappa=0.5, theta=0.04, sigma=0.1, r0=0.03))
        left, right = cir._boundaries(*cir.truncated_domain(1.0))
        assert isinstance(left, Natural) and isinstance(right, Linear)

    def test_x0_outside_domain(self):
        d = DiffusionPDE(drift=0.0, volatility=0.5, x0=0.0, domain=lambda T: (1.0, 2.0))
        with pytest.raises(ValueError):
            d.truncated_domain(1.0)

    def test_discount_variants(self):
        d = DiffusionPDE(drift=0.0, volatility=0.1, x0=0.05)
        x = np.array([0.02, 0.05])
        assert np.allclose(d.pde(0.03).coefficients(x, 0.0)[2], -0.03)
        assert np.allclose(d.pde("state").coefficients(x, 0.0)[2], -x)
        assert np.allclose(d.pde(lambda x, t: 2 * x).coefficients(x, 0.0)[2], -2 * x)
        assert not d.pde(lambda x, t: 2 * x).time_homogeneous
        with pytest.raises(ValueError):
            d.pde("rate")

    def test_generator_bridge(self):
        d = to_pde(OrnsteinUhlenbeckProcess(theta=1.5, mu=0.5, sigma=0.4, x0=0.0))
        gen = d.generator()
        # A x^2 = 2 x mu(x) + sigma^2
        assert gen(lambda x: x**2, 0.7) == pytest.approx(
            2 * 0.7 * 1.5 * (0.5 - 0.7) + 0.16, abs=1e-6
        )
        assert gen(math.sin, 0.0) == pytest.approx(1.5 * 0.5, abs=1e-6)
