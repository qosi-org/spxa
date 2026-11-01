"""Exactness tests: continuously monitored barrier options under Black-Scholes."""

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.reference import barrier_price, bs_price, double_barrier_price

pytestmark = pytest.mark.exactness

S, T, R, Q, SIG, REBATE = 100.0, 0.5, 0.08, 0.04, 0.25, 3.0
CASES = [
    ("down-and-out", 95.0, "call"),
    ("up-and-out", 105.0, "call"),
    ("down-and-in", 95.0, "call"),
    ("up-and-in", 105.0, "call"),
    ("down-and-out", 95.0, "put"),
    ("up-and-out", 105.0, "put"),
    ("down-and-in", 95.0, "put"),
    ("up-and-in", 105.0, "put"),
]


class TestSingleBarrier:
    @pytest.mark.parametrize("bt,H,opt", CASES)
    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_haug_table(self, bt, H, opt, K):
        """Haug's barrier table (rebate 3) against Reiner-Rubinstein."""
        pde = BlackScholesPDE(R, SIG, Q).barrier(K, T, S, H, bt, opt, REBATE).price
        ref = barrier_price(S, K, H, T, R, SIG, Q, opt, bt, REBATE)
        assert pde == pytest.approx(ref, abs=5e-4)

    @pytest.mark.parametrize("bt,H", [("down-and-out", 95.0), ("up-and-in", 105.0)])
    def test_rebate_at_expiry(self, bt, H):
        pde = BlackScholesPDE(R, SIG, Q).barrier(100.0, T, S, H, bt, "call", REBATE, "expiry")
        ref = barrier_price(S, 100.0, H, T, R, SIG, Q, "call", bt, REBATE, "expiry")
        assert pde.price == pytest.approx(ref, abs=5e-4)

    def test_in_out_parity_and_greeks(self):
        m = BlackScholesPDE(R, SIG, Q)
        out = m.barrier(100.0, T, S, 90.0, "down-and-out", "put")
        inn = m.barrier(100.0, T, S, 90.0, "down-and-in", "put")
        van = m.european(100.0, T, S, "put")
        assert out.price + inn.price == pytest.approx(van.price, abs=1e-10)
        assert out.delta + inn.delta == pytest.approx(van.delta, abs=1e-8)
        # finite-difference delta of the reference
        h = 0.01
        fd = (
            barrier_price(S + h, 100, 90, T, R, SIG, Q, "put", "down-and-out")
            - barrier_price(S - h, 100, 90, T, R, SIG, Q, "put", "down-and-out")
        ) / (2 * h)
        assert out.delta == pytest.approx(fd, abs=2e-4)

    def test_value_near_barrier(self):
        res = BlackScholesPDE(R, SIG, Q).barrier(100.0, T, S, 95.0, "down-and-out", "call", 2.0)
        assert res.solution.value(95.0) == pytest.approx(2.0, abs=1e-12)
        s = np.array([96.0, 98.0, 102.0])
        ref = [barrier_price(x, 100.0, 95.0, T, R, SIG, Q, "call", "down-and-out", 2.0) for x in s]
        np.testing.assert_allclose(res.solution.value(s), ref, atol=1e-3)

    def test_convergence(self):
        m = BlackScholesPDE(R, SIG, Q)
        ref = barrier_price(S, 100.0, 95.0, T, R, SIG, Q, "call", "down-and-out")
        errs = [
            abs(m.barrier(100.0, T, S, 95.0, n_space=n, n_time=n // 2).price - ref)
            for n in (101, 201, 401, 801)
        ]
        assert np.log2(errs[0] / errs[-1]) / 3 > 1.7

    def test_american_knock_out_bounds(self):
        m = BlackScholesPDE(0.05, 0.3)
        euro = m.barrier(100.0, 1.0, 100.0, 130.0, "up-and-out", "put").price
        amer = m.barrier(100.0, 1.0, 100.0, 130.0, "up-and-out", "put", american=True).price
        vanilla_am = m.american(100.0, 1.0, 100.0, "put").price
        assert euro < amer < vanilla_am
        far = m.barrier(100.0, 1.0, 100.0, 1000.0, "up-and-out", "put", american=True).price
        assert far == pytest.approx(vanilla_am, abs=5e-3)

    def test_validation(self):
        m = BlackScholesPDE(R, SIG, Q)
        with pytest.raises(ValueError):
            m.barrier(100.0, T, 90.0, 95.0, "down-and-out")
        with pytest.raises(ValueError):
            m.barrier(100.0, T, S, 95.0, "sideways")
        with pytest.raises(ValueError):
            m.barrier(100.0, T, S, 95.0, rebate_timing="later")
        with pytest.raises(NotImplementedError):
            m.barrier(100.0, T, S, 95.0, "down-and-in", american=True)


class TestDoubleBarrier:
    @pytest.mark.parametrize("sigma", [0.15, 0.25, 0.35])
    @pytest.mark.parametrize("L,U", [(50, 150), (70, 130), (80, 120), (90, 110)])
    def test_against_series(self, sigma, L, U):
        pde = BlackScholesPDE(0.1, sigma).double_barrier(100.0, 0.25, 100.0, L, U).price
        ref = double_barrier_price(100.0, 100.0, L, U, 0.25, 0.1, sigma)
        assert pde == pytest.approx(ref, abs=5e-4)

    def test_wide_barriers_reduce_to_vanilla(self):
        pde = BlackScholesPDE(0.05, 0.2).double_barrier(100.0, 0.5, 100.0, 20.0, 500.0).price
        assert pde == pytest.approx(float(bs_price(100, 100, 0.5, 0.05, 0.2)), abs=2e-4)

    def test_rebate_and_validation(self):
        m = BlackScholesPDE(0.05, 0.2)
        res = m.double_barrier(100.0, 0.5, 100.0, 90.0, 110.0, "put", rebate=1.0)
        assert res.solution.value(90.0) == pytest.approx(1.0)
        assert res.solution.value(110.0) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            m.double_barrier(100.0, 0.5, 100.0, 110.0, 120.0)
