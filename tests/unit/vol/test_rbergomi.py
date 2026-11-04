"""Rough Bergomi by the hybrid scheme."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rough import RoughBergomi
from spxa.vol.rbergomi import hybrid_weights, rbergomi_paths

K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])


def test_scheme_variance_tracks_the_exact_one() -> None:
    w, var = hybrid_weights(0.1, 0.01, 100)
    t = 0.01 * np.arange(101)
    assert var[1] == pytest.approx(0.01**0.2 / 0.2, rel=1e-14)  # first cell is exact
    np.testing.assert_allclose(var[1:], t[1:] ** 0.2 / 0.2, rtol=0.01)
    assert w.size == 99 and np.all(np.diff(w) < 0)


def test_forward_variance_and_martingale() -> None:
    xi = lambda t: 0.04 + 0.01 * t  # noqa: E731
    p = rbergomi_paths(0.1, 1.0, -0.7, xi, 2.0, 200, 40_000, seed=1, keep_variance=True)
    assert p.variance is not None
    for i in (1, 20, 100, 200):
        v = p.variance[:, i]
        assert abs(v.mean() - xi(p.times[i])) < 3 * v.std() / np.sqrt(v.size)
    assert abs(p.spot.mean() - 1.0) < 3 * p.spot.std() / np.sqrt(p.spot.size)
    assert p.int_v.mean() == pytest.approx(0.04 * 2 + 0.01 * 2 - 0.01 * 0.01, rel=0.02)


def test_conditional_estimator_reduces_variance() -> None:
    p = rbergomi_paths(0.1, 1.9, -0.9, 0.04, 1.0, 100, 50_000, seed=2)
    mixed, se_mixed = p.price(K)
    plain, se_plain = p.price(K, conditional=False)
    assert np.all(np.abs(mixed - plain) < 3 * se_plain)
    # only the orthogonal noise is removed: modest at rho = -0.9, larger in the wings
    assert np.all(se_mixed < se_plain) and se_plain[2] / se_mixed[2] > 1.2
    assert se_plain[-1] / se_mixed[-1] > 1.5
    put, _ = p.price(K, "put")
    s_eff = np.exp(-0.9 * p.int_sqrt_v_dw - 0.5 * 0.81 * p.int_v)
    np.testing.assert_allclose(mixed - put, s_eff.mean() - K, atol=1e-12)


def test_hybrid_agrees_with_exact_cholesky() -> None:
    hyb = rbergomi_paths(0.1, 1.9, -0.9, 0.04, 1.0, 100, 100_000, seed=2)
    price, se = hyb.price(K)
    exact = RoughBergomi(0.1, 1.9, -0.9, 0.04, 1.0)
    ST = np.concatenate(
        [exact.simulate_paths(1.0, 100, 20_000, seed=s)["spot_paths"][:, -1] for s in (3, 4, 5)]
    )
    ref = np.array([np.maximum(ST - k, 0).mean() for k in K])
    ref_se = np.array([np.maximum(ST - k, 0).std() / np.sqrt(ST.size) for k in K])
    assert np.all(np.abs(price - ref) < 3 * np.hypot(se, ref_se) + 0.03 * ref)
    vols = hyb.implied_vol(K)
    assert np.all(np.diff(vols) < 0)  # negative skew for rho < 0
    with pytest.raises(ValueError):
        rbergomi_paths(0.5, 1.0, -0.7, 0.04, 1.0, 10, 10)
