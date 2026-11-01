"""Exactness tests for the double-barrier eigenfunction expansion."""

import pytest

from spxa.pde.reference import barrier_price, bs_price, double_barrier_price

pytestmark = pytest.mark.exactness


class TestDoubleBarrier:
    def test_wide_corridor_recovers_vanilla(self):
        for option in ("call", "put"):
            got = double_barrier_price(100, 100, 1e-3, 1e5, 1.0, 0.05, 0.2, 0.01, option)
            assert got == pytest.approx(float(bs_price(100, 100, 1.0, 0.05, 0.2, 0.01, option)))

    def test_one_sided_limits_match_reiner_rubinstein(self):
        dko = double_barrier_price(100, 100, 90, 1e5, 1.0, 0.05, 0.2, 0.02, "call")
        rr = barrier_price(100, 100, 90, 1.0, 0.05, 0.2, 0.02, "call", "down-and-out")
        assert dko == pytest.approx(rr, rel=1e-9)
        uko = double_barrier_price(100, 100, 1e-4, 115, 0.5, 0.03, 0.25, 0.0, "put")
        rr = barrier_price(100, 100, 115, 0.5, 0.03, 0.25, 0.0, "put", "up-and-out")
        assert uko == pytest.approx(rr, rel=1e-9)

    def test_bounded_by_single_barriers(self):
        args = (100, 100, 85, 120, 0.75, 0.04, 0.3, 0.0, "call")
        dko = double_barrier_price(*args)
        down = barrier_price(100, 100, 85, 0.75, 0.04, 0.3, 0.0, "call", "down-and-out")
        up = barrier_price(100, 100, 120, 0.75, 0.04, 0.3, 0.0, "call", "up-and-out")
        assert 0.0 < dko < min(down, up)

    def test_short_maturity_convergence(self):
        # many terms are needed for small T; far barriers are then irrelevant
        val = double_barrier_price(100, 100, 80, 120, 0.01, 0.05, 0.2, 0.0, "call")
        assert val == pytest.approx(
            float(bs_price(100, 100, 0.01, 0.05, 0.2, 0.0, "call")), abs=1e-10
        )
        tight = double_barrier_price(100, 100, 97, 103, 0.05, 0.05, 0.2, 0.0, "call")
        assert 0.0 < tight < float(bs_price(100, 100, 0.05, 0.05, 0.2, 0.0, "call"))

    def test_degenerate_cases(self):
        assert double_barrier_price(130, 100, 80, 120, 1.0, 0.05, 0.2) == 0.0
        assert double_barrier_price(100, 130, 80, 120, 1.0, 0.05, 0.2, 0.0, "call") == 0.0
        assert double_barrier_price(100, 70, 80, 120, 1.0, 0.05, 0.2, 0.0, "put") == 0.0
        with pytest.raises(ValueError):
            double_barrier_price(100, 100, 120, 80, 1.0, 0.05, 0.2)
