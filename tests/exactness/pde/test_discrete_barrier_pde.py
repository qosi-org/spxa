"""Exactness tests: discretely monitored barrier options under Black-Scholes."""

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.reference import barrier_price

pytestmark = pytest.mark.exactness

BETA = 0.5825971579390106  # -zeta(1/2)/sqrt(2 pi), Broadie-Glasserman-Kou


def dates(T, m):
    return np.arange(1, m + 1) * T / m


class TestDiscreteBarrier:
    @pytest.mark.parametrize(
        "L,m,ref",
        [(95.0, 25, 6.63156), (99.5, 25, 3.35558), (95.0, 125, 6.16864)],
    )
    def test_fusai_abrahams_sgarra(self, L, m, ref):
        """Wiener-Hopf closed form of Fusai, Abrahams & Sgarra (2006)."""
        pde = BlackScholesPDE(0.1, 0.2).discrete_barrier(
            100.0, 0.5, 100.0, L, dates(0.5, m), n_space=1601, n_time=1000
        )
        assert pde.price == pytest.approx(ref, abs=2e-3)

    @pytest.mark.parametrize(
        "H,bt,opt", [(85.0, "down-and-out", "call"), (120.0, "up-and-out", "put")]
    )
    def test_bgk_continuity_correction(self, H, bt, opt):
        T, r, sig, m = 0.5, 0.05, 0.3, 50
        pde = BlackScholesPDE(r, sig).discrete_barrier(100.0, T, 100.0, H, dates(T, m), bt, opt)
        shift = np.exp((-1 if bt.startswith("down") else 1) * BETA * sig * np.sqrt(T / m))
        ref = barrier_price(100.0, 100.0, H * shift, T, r, sig, 0.0, opt, bt)
        assert pde.price == pytest.approx(ref, rel=5e-3)

    def test_monotone_in_monitoring_frequency(self):
        m = BlackScholesPDE(0.05, 0.3)
        prices = [
            m.discrete_barrier(100.0, 0.5, 100.0, 90.0, dates(0.5, k)).price
            for k in (2, 8, 32, 128)
        ]
        cont = barrier_price(100.0, 100.0, 90.0, 0.5, 0.05, 0.3, 0.0, "call", "down-and-out")
        assert all(a > b for a, b in zip(prices, prices[1:]))
        assert prices[-1] > cont
        euro = m.european(100.0, 0.5, 100.0).price
        assert prices[0] < euro

    def test_space_convergence_is_regular(self):
        m = BlackScholesPDE(0.1, 0.2)
        vals = [
            m.discrete_barrier(
                100.0, 0.5, 100.0, 95.0, dates(0.5, 25), n_space=n, n_time=1000
            ).price
            for n in (201, 401, 801, 1601)
        ]
        d = np.abs(np.diff(vals))
        assert np.all(d[:-1] > d[1:])

    def test_rebate_on_knock_out(self):
        m = BlackScholesPDE(0.05, 0.2)
        no = m.discrete_barrier(100.0, 0.5, 100.0, 95.0, dates(0.5, 10)).price
        with_r = m.discrete_barrier(100.0, 0.5, 100.0, 95.0, dates(0.5, 10), rebate=2.0).price
        assert 0.0 < with_r - no < 2.0

    def test_validation(self):
        with pytest.raises(ValueError):
            BlackScholesPDE(0.05, 0.2).discrete_barrier(100, 0.5, 100, 95, [0.25], "down-and-in")
