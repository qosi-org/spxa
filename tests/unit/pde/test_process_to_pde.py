"""Process.to_pde delegates to spxa.pde.to_pde."""

import pytest

from spxa.pde import DiffusionPDE, LevyPIDE, to_pde
from spxa.zoo.beyond import CIRProcess, HestonProcess, OrnsteinUhlenbeckProcess
from spxa.zoo.levy import MertonJumpDiffusion, VarianceGamma


def test_diffusion():
    ou = OrnsteinUhlenbeckProcess(theta=2.0, mu=0.1, sigma=0.3, x0=0.5)
    pde = ou.to_pde()
    assert isinstance(pde, DiffusionPDE)
    ref = to_pde(ou)
    assert (pde.name, pde.x0, pde.lower, pde.upper) == (ref.name, ref.x0, ref.lower, ref.upper)
    for a, b in ((pde.drift, ref.drift), (pde.volatility, ref.volatility)):
        if callable(a):
            assert a(0.37, 0.0) == pytest.approx(b(0.37, 0.0))
        else:
            assert a == pytest.approx(b)
    assert isinstance(CIRProcess(kappa=0.5, theta=0.04, sigma=0.1, r0=0.03).to_pde(), DiffusionPDE)


def test_jump_processes():
    merton = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
    pide = merton.to_pde(r=0.05, q=0.01)
    assert isinstance(pide, LevyPIDE) and (pide.r, pide.q) == (0.05, 0.01)
    assert isinstance(VarianceGamma(sigma=0.12, nu=0.2, theta=-0.14).to_pde(r=0.03), LevyPIDE)
    with pytest.raises(ValueError):
        merton.to_pde()


def test_unregistered():
    heston = HestonProcess(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.9)
    with pytest.raises(NotImplementedError):
        heston.to_pde()
