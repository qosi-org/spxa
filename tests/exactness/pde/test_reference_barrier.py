"""Exactness tests for the Reiner-Rubinstein barrier formulas.

Reference values: Haug, *The Complete Guide to Option Pricing Formulas*
(2nd ed.), Table 4-13: S=100, T=0.5, r=0.08, b=0.04 (q=0.04), sigma=0.25,
cash rebate 3.
"""

import numpy as np
import pytest

from spxa.pde.reference import barrier_price, bs_price

pytestmark = pytest.mark.exactness

S, T, R, Q, SIG, REBATE = 100.0, 0.5, 0.08, 0.04, 0.25, 3.0

HAUG_TABLE = [
    ("down-and-out", 95, "call", [9.0246, 6.7924, 4.8759]),
    ("down-and-out", 100, "call", [3.0000, 3.0000, 3.0000]),
    ("up-and-out", 105, "call", [2.6789, 2.3580, 2.3453]),
    ("down-and-in", 95, "call", [7.7627, 4.0109, 2.0576]),
    ("down-and-in", 100, "call", [13.8333, 7.8494, 3.9795]),
    ("up-and-in", 105, "call", [14.1112, 8.4482, 4.5910]),
    ("down-and-out", 95, "put", [2.2798, 2.2947, 2.6252]),
    ("down-and-out", 100, "put", [3.0000, 3.0000, 3.0000]),
    ("up-and-out", 105, "put", [3.7760, 5.4932, 7.5187]),
    ("down-and-in", 95, "put", [2.9586, 6.5677, 11.9752]),
    ("down-and-in", 100, "put", [2.2845, 5.9085, 11.6465]),
    ("up-and-in", 105, "put", [1.4653, 3.3721, 7.0846]),
]


class TestReinerRubinstein:
    @pytest.mark.parametrize("barrier_type,H,option,expected", HAUG_TABLE)
    def test_haug_table(self, barrier_type, H, option, expected):
        got = [
            barrier_price(S, K, H, T, R, SIG, Q, option, barrier_type, REBATE)
            for K in (90.0, 100.0, 110.0)
        ]
        np.testing.assert_allclose(got, expected, atol=5e-5)

    @pytest.mark.parametrize("option", ["call", "put"])
    @pytest.mark.parametrize("H", [80.0, 97.0, 104.0, 125.0])
    @pytest.mark.parametrize("K", [90.0, 100.0, 112.0])
    def test_in_out_parity(self, option, H, K):
        direction = "down" if H < S else "up"
        out = barrier_price(S, K, H, T, R, SIG, Q, option, f"{direction}-and-out")
        knock_in = barrier_price(S, K, H, T, R, SIG, Q, option, f"{direction}-and-in")
        assert out + knock_in == pytest.approx(float(bs_price(S, K, T, R, SIG, Q, option)))

    def test_rebate_timing(self):
        hit = barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-out", REBATE, "hit")
        exp_ = barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-out", REBATE, "expiry")
        core = barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-out", 0.0)
        assert core < exp_ < hit
        # rebate at expiry if hit + rebate at expiry if not hit = discounted cash
        in_reb = barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-in", REBATE) - (
            barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-in", 0.0)
        )
        assert (exp_ - core) + in_reb == pytest.approx(REBATE * np.exp(-R * T))

    def test_far_barrier_limits(self):
        vanilla = float(bs_price(S, 100, T, R, SIG, Q, "call"))
        assert barrier_price(S, 100, 1e-3, T, R, SIG, Q, "call", "down-and-out") == pytest.approx(
            vanilla, rel=1e-10
        )
        assert barrier_price(S, 100, 1e4, T, R, SIG, Q, "call", "up-and-in") == pytest.approx(
            0.0, abs=1e-12
        )

    def test_breached_barrier(self):
        assert barrier_price(90, 100, 95, T, R, SIG, Q, "call", "down-and-out", 2.0) == 2.0
        assert barrier_price(
            90, 100, 95, T, R, SIG, Q, "call", "down-and-out", 2.0, "expiry"
        ) == pytest.approx(2.0 * np.exp(-R * T))
        assert barrier_price(110, 100, 105, T, R, SIG, Q, "put", "up-and-in") == pytest.approx(
            float(bs_price(110, 100, T, R, SIG, Q, "put"))
        )

    def test_validation(self):
        with pytest.raises(ValueError):
            barrier_price(S, 100, 95, T, R, SIG, Q, "call", "sideways-and-out")
        with pytest.raises(ValueError):
            barrier_price(S, 100, 95, T, R, SIG, Q, "call", "down-and-out", rebate_timing="never")
        with pytest.raises(ValueError):
            barrier_price(S, 100, -1.0, T, R, SIG, Q)
