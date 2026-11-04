"""PCA of yield-curve changes."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.pca import curve_pca

TENORS = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], float)


def _factors() -> np.ndarray:
    level = np.ones_like(TENORS)
    slope = np.log(TENORS) - np.log(TENORS).mean()
    curv = np.exp(-((np.log(TENORS) - np.log(5.0)) ** 2))
    q, _ = np.linalg.qr(np.column_stack([level, slope, curv]))
    return q


def _exact_changes(n: int, sds: list[float], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Changes = scores V^T with orthogonal, mean-zero scores of exact sample variances."""
    v = _factors()
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, 3))
    raw -= raw.mean(axis=0)
    u, _ = np.linalg.qr(raw)  # orthonormal columns, still mean-zero
    scores = u * np.sqrt(n - 1) * np.array(sds)
    return scores @ v.T, v


def test_recovers_known_factors_exactly() -> None:
    sds = [0.0008, 0.0004, 0.00015]  # 8, 4 and 1.5 bp a day
    dz, v = _exact_changes(500, sds)
    pca = curve_pca(dz, TENORS, differences=False)
    for j in range(3):
        sign = np.sign(pca.components[:, j] @ v[:, j])
        assert sign * pca.components[:, j] == pytest.approx(v[:, j], abs=1e-12)
    assert pca.volatilities == pytest.approx(sds, rel=1e-12)
    assert pca.explained_ratio.sum() == pytest.approx(1.0, abs=1e-12)
    assert pca.explained_ratio == pytest.approx(np.square(sds) / np.sum(np.square(sds)), rel=1e-12)


def test_sign_conventions_and_shapes() -> None:
    dz, _ = _exact_changes(300, [0.0008, 0.0004, 0.00015], seed=3)
    pca = curve_pca(np.cumsum(dz, axis=0) + 0.03, TENORS)  # levels, differenced inside
    level, slope, curv = pca.components.T
    assert np.all(level > 0)  # a level move lifts every tenor
    assert slope[-1] > 0 > slope[0]  # steepener
    assert curv[TENORS.size // 2] > 0
    assert np.all(np.diff(slope) > 0)


def test_scores_reconstruct_and_scenarios() -> None:
    dz, _ = _exact_changes(400, [0.0008, 0.0004, 0.00015], seed=5)
    pca = curve_pca(dz, TENORS, differences=False)
    s = pca.scores(dz)
    assert pca.reconstruct(s) == pytest.approx(dz, abs=1e-15)  # rank three data
    assert np.var(s, axis=0, ddof=1) == pytest.approx(pca.variances, rel=1e-12)
    shock = pca.scenario(0, 2.0)
    assert np.linalg.norm(shock) == pytest.approx(2 * pca.volatilities[0], rel=1e-14)


def test_factor_risk() -> None:
    dz, _ = _exact_changes(400, [0.0008, 0.0004, 0.00015], seed=7)
    pca = curve_pca(dz, TENORS, differences=False)
    rng = np.random.default_rng(1)
    d = rng.normal(0, 100, TENORS.size)  # key-rate deltas of some book
    pnl = dz @ d  # first-order P&L each day
    # with exact rank-3 data the factor model explains the P&L variance entirely
    assert pca.factor_variance(d) == pytest.approx(np.var(pnl, ddof=1), rel=1e-11)
    assert pca.factor_deltas(d) == pytest.approx(d @ pca.components)
    with pytest.raises(ValueError):
        pca.factor_deltas(d[:-1])


def test_noise_lowers_explained_share() -> None:
    dz, _ = _exact_changes(2000, [0.0008, 0.0004, 0.00015], seed=9)
    noisy = dz + np.random.default_rng(2).normal(0, 0.00003, dz.shape)
    pca = curve_pca(noisy, TENORS, differences=False, n_components=10)
    assert 0.98 < pca.explained_ratio[:3].sum() < 1.0
    assert pca.explained_ratio.sum() == pytest.approx(1.0, abs=1e-12)


def test_validation() -> None:
    with pytest.raises(ValueError):
        curve_pca(np.zeros((5, 3)), TENORS)
    with pytest.raises(ValueError):
        curve_pca(np.zeros((2, 2)), [1.0, 2.0])
    with pytest.raises(ValueError):
        curve_pca(np.zeros((5, 2)), [2.0, 1.0])
    with pytest.raises(ValueError):
        curve_pca(np.zeros((5, 2)), [1.0, 2.0], n_components=3)
