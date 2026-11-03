"""Tests for spxa.vol.conventions."""

from __future__ import annotations

import math

import numpy as np
import pytest

import spxa
from spxa.market import black_price, normalised_black_call
from spxa.vol import (
    implied_vol_from_total_variance,
    log_moneyness,
    normalised_call_from_total_variance,
    strike_from_log_moneyness,
    total_variance,
    total_variance_from_price,
)


def test_lazy_import() -> None:
    assert spxa.vol.total_variance is total_variance
    assert "vol" in dir(spxa)


def test_moneyness_round_trip() -> None:
    K = np.array([50.0, 100.0, 250.0])
    k = log_moneyness(K, 100.0)
    assert k[1] == 0.0 and k[0] == pytest.approx(math.log(0.5), abs=1e-16)
    assert np.allclose(strike_from_log_moneyness(k, 100.0), K, rtol=1e-15, atol=0)
    assert isinstance(log_moneyness(110.0, 100.0), float)
    with pytest.raises(ValueError):
        log_moneyness(-1.0, 100.0)


def test_total_variance_round_trip() -> None:
    assert total_variance(0.25, 2.0) == 0.125  # exact in binary
    T = np.array([0.25, 1.0, 5.0])
    s = np.array([0.3, 0.2, 0.15])
    assert np.allclose(implied_vol_from_total_variance(total_variance(s, T), T), s, rtol=1e-15)
    assert math.isnan(implied_vol_from_total_variance(-0.01, 1.0))
    with pytest.raises(ValueError):
        total_variance(0.2, -1.0)
    with pytest.raises(ValueError):
        implied_vol_from_total_variance(0.04, 0.0)


def test_normalised_call() -> None:
    k = np.linspace(-1.0, 1.0, 9)
    w = 0.09
    c = normalised_call_from_total_variance(k, w)
    F = 123.0
    assert np.allclose(c * F, black_price(F, F * np.exp(k), 1.0, math.sqrt(w)), rtol=1e-14)
    # c = e^{k/2} b(-k, sqrt w)
    assert np.allclose(c, np.exp(k / 2) * normalised_black_call(-k, math.sqrt(w)), rtol=1e-13)
    # w = 0: intrinsic (1 - e^k)^+
    assert np.allclose(
        normalised_call_from_total_variance(k, 0.0), np.maximum(1 - np.exp(k), 0), atol=1e-16
    )
    with pytest.raises(ValueError):
        normalised_call_from_total_variance(0.0, -0.1)


def test_total_variance_from_price() -> None:
    F, T, df = 100.0, 0.5, 0.97
    K = np.array([80.0, 100.0, 130.0])
    vols = np.array([0.31, 0.22, 0.19])
    p = black_price(F, K, T, vols, df, "put")
    w = total_variance_from_price(p, F, K, T, df, "put")
    assert np.allclose(w, vols**2 * T, rtol=1e-12)
    assert math.isnan(total_variance_from_price(150.0, F, 100.0, T))
