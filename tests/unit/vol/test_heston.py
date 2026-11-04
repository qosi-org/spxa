"""Heston model in forward coordinates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.pde.reference import bs_price
from spxa.vol.heston import Heston

CASE1 = Heston(0.04, 1.5, 0.04, 0.3, -0.9)


def test_in_t_hout_foulon_case_one() -> None:
    r, T = 0.025, 1.0
    got = CASE1.price(100.0, T, F=100 * math.exp(r * T), df=math.exp(-r * T))
    assert got == pytest.approx(8.8948693600540, abs=1e-9)


def test_parity_and_skew() -> None:
    K = np.array([80.0, 100.0, 125.0])
    c = CASE1.price(K, 1.0, F=100.0, df=0.97)
    p = CASE1.price(K, 1.0, F=100.0, df=0.97, option="put")
    np.testing.assert_allclose(c - p, 0.97 * (100.0 - K), atol=1e-10)
    vols = CASE1.implied_vol(K, 1.0, F=100.0)
    assert vols[0] > vols[1] > vols[2]  # rho < 0 gives a negative skew


def test_small_vol_of_variance_is_black_with_mean_variance() -> None:
    m = Heston(0.09, 2.0, 0.04, 1e-3, 0.0)
    T = 1.5
    w = float(m.expected_total_variance(T))
    assert m.implied_vol(110.0, T, F=100.0) == pytest.approx(math.sqrt(w / T), abs=1e-6)
    assert m.price(110.0, T, F=100.0) == pytest.approx(
        float(bs_price(100.0, 110.0, T, 0.0, math.sqrt(w / T))), abs=1e-5
    )


def test_mixing_and_validation() -> None:
    assert CASE1.feller and not Heston(0.04, 0.5, 0.04, 0.3, 0.0).feller
    assert CASE1.scaled(0.5).xi == pytest.approx(0.15)
    assert CASE1.mean_variance(0.0) == pytest.approx(0.04)
    assert CASE1.implied_vol(100.0, 1e-3, F=100.0) == pytest.approx(CASE1.atm_short_vol(), abs=2e-3)
    with pytest.raises(ValueError):
        Heston(0.04, 1.0, 0.04, 0.3, 1.0)
