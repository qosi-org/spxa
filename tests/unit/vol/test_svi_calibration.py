"""Quasi-explicit SVI calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from spxa.vol import (
    SVIRaw,
    calibrate_svi,
    calibrate_svi_butterfly_free,
    check_butterfly,
    svi_inner_fit,
)

K = np.linspace(-1.0, 0.6, 25)
TRUE = SVIRaw(0.02, 0.4, -0.5, 0.1, 0.3)


def test_exact_recovery() -> None:
    fit = calibrate_svi(K, TRUE(K))
    p = fit.params
    assert fit.success and fit.rmse < 1e-11
    assert (p.a, p.b, p.rho, p.m, p.sigma) == pytest.approx((0.02, 0.4, -0.5, 0.1, 0.3), abs=1e-8)


def test_noisy_fit_is_within_noise() -> None:
    rng = np.random.default_rng(7)
    noise = 0.002
    w = TRUE(K) * (1 + noise * rng.standard_normal(K.size))
    fit = calibrate_svi(K, w)
    assert fit.rmse < 1.5 * noise * TRUE(K).max()
    assert np.max(np.abs(fit.params(K) / TRUE(K) - 1)) < 5 * noise


def test_inner_problem_matches_constrained_solver() -> None:
    rng = np.random.default_rng(3)
    w = TRUE(K) * (1 + 0.01 * rng.standard_normal(K.size))
    m, s = 0.05, 0.25
    p, sse = svi_inner_fit(K, w, m, s)
    y = (K - m) / s
    z = np.sqrt(y * y + 1)

    def f(x: np.ndarray) -> float:
        a, c, d = x
        return float(np.sum((a + d * y + c * z - w) ** 2))

    cons = [
        {"type": "ineq", "fun": lambda x: x[1] - abs(x[2])},
        {"type": "ineq", "fun": lambda x: 2 * s - x[1] - abs(x[2])},
        {"type": "ineq", "fun": lambda x: x[0]},
        {"type": "ineq", "fun": lambda x: w.max() - x[0]},
    ]
    ref = minimize(f, [0.01, 0.1, 0.0], constraints=cons, method="SLSQP", options={"ftol": 1e-15})
    assert sse == pytest.approx(ref.fun, rel=1e-6)
    assert sse <= ref.fun * (1 + 1e-9)
    assert p.m == m and p.sigma == s


def test_slope_bound() -> None:
    steep = SVIRaw(0.01, 1.5, -0.9, 0.0, 0.2)  # left wing slope b(1 + |rho|) = 2.85
    lee = calibrate_svi(K, steep(K))
    assert lee.params.b * (1 + abs(lee.params.rho)) <= 2 + 1e-12
    assert lee.params.satisfies_lee_bound
    assert lee.rmse > 1e-3  # cannot fit beyond the bound
    rt = calibrate_svi(K, steep(K), max_slope=4.0)  # Rogers-Tehranchi bound (Zeliade)
    assert rt.rmse < 1e-10
    assert rt.params.b == pytest.approx(1.5, rel=1e-8)


def test_weights_and_flat_smile() -> None:
    w = TRUE(K).copy()
    w[3] *= 1.5  # outlier
    wt = np.ones_like(K)
    wt[3] = 0.0
    fit = calibrate_svi(K, w, weights=wt)
    assert np.max(np.abs(fit.params(K) - TRUE(K))[wt > 0]) < 1e-9
    flat = calibrate_svi(K, np.full(K.size, 0.04))
    assert np.allclose(flat.params(K), 0.04, atol=1e-12)


def test_validation() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        calibrate_svi(K[:4], TRUE(K[:4]))
    with pytest.raises(ValueError, match="positive"):
        calibrate_svi(K, -TRUE(K))
    with pytest.raises(ValueError, match="weights"):
        calibrate_svi(K, TRUE(K), weights=np.ones(3))
    with pytest.raises(ValueError, match="max_slope"):
        calibrate_svi(K, TRUE(K), max_slope=0.0)


class TestButterflyFree:
    vogt = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    k = np.linspace(-1.5, 1.5, 61)

    def test_negative_a_is_recovered(self) -> None:
        # minimum variance stays positive, so a < 0 is admissible
        fit = calibrate_svi(self.k, self.vogt(self.k))
        assert fit.params.a == pytest.approx(-0.0410, abs=1e-8)
        assert fit.rmse < 1e-11

    def test_vogt_slice_is_repaired(self) -> None:
        fit = calibrate_svi_butterfly_free(self.k, self.vogt(self.k))
        assert fit.success
        dense = check_butterfly(fit.params, np.linspace(-10, 10, 200_001))
        assert dense.arbitrage_free and dense.min_g >= 0
        assert not check_butterfly(self.vogt, np.linspace(-1.5, 1.5, 3001))
        # the repair costs little: about 23 bp of total variance at worst
        assert fit.rmse < 3e-3
        assert np.max(np.abs(fit.params(self.k) - self.vogt(self.k))) < 6e-3

    def test_noisy_vogt(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(3):
            w = self.vogt(self.k) * (1 + 0.003 * rng.standard_normal(self.k.size))
            fit = calibrate_svi_butterfly_free(self.k, w)
            assert fit.success
            assert check_butterfly(fit.params, np.linspace(-10, 10, 100_001))

    def test_arbitrage_free_data_unchanged(self) -> None:
        fit = calibrate_svi_butterfly_free(K, TRUE(K))
        assert fit.success and fit.rmse < 1e-9
        assert fit.params.to_array() == pytest.approx(TRUE.to_array(), abs=1e-7)
