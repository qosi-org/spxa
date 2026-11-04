"""VIX in rough Bergomi and Heston."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2

from spxa.vol import Heston
from spxa.vol.vix import VIX_WINDOW, HestonVIX, RBergomiVIX


def xi(t: np.ndarray) -> np.ndarray:
    return 0.04 + 0.01 * np.asarray(t)


M = RBergomiVIX(0.1, 1.9, xi)
T = 0.5


def test_covariance_matches_quadrature() -> None:
    u, w = M.nodes(T)
    C = M.covariance(T)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    for i, j in [(0, 0), (3, 10), (5, 23), (23, 23)]:
        ref = 0.2 * quad(lambda s: (u[i] - s) ** -0.4 * (u[j] - s) ** -0.4, 0, T, limit=200)[0]
        assert C[i, j] == pytest.approx(ref, rel=1e-8)
    assert C[0, 0] == pytest.approx(u[0] ** 0.2 - (u[0] - T) ** 0.2, rel=1e-13)
    assert np.linalg.eigvalsh(C).min() > -1e-12


def test_vix_squared_is_unbiased_and_futures_obey_jensen() -> None:
    exact = quad(xi, T, T + VIX_WINDOW)[0] / VIX_WINDOW
    assert M.expected_vix_squared(T) == pytest.approx(exact, rel=1e-14)
    v2 = M.vix_squared(T, 400_000, seed=1)
    assert abs(v2.mean() - exact) < 3 * v2.std() / np.sqrt(v2.size)
    m1, s2 = M.lognormal_moments(T)
    m2 = m1 * m1 * np.exp(s2)
    assert abs((v2**2).mean() - m2) < 3 * (v2**2).std() / np.sqrt(v2.size)
    fut, se = M.futures(T, 400_000, seed=1)
    assert fut < np.sqrt(exact) - 0.03  # strong concavity adjustment for eta = 1.9
    assert fut == pytest.approx(M.futures_lognormal(T), rel=0.01)
    # quadrature converged: 24 vs 48 nodes
    assert s2 == pytest.approx(
        RBergomiVIX(0.1, 1.9, xi, n_nodes=48).lognormal_moments(T)[1], rel=1e-4
    )
    # eta -> 0: VIX is deterministic
    assert RBergomiVIX(0.1, 1e-6, xi).futures(T, 1000, seed=0)[0] == pytest.approx(
        np.sqrt(exact), rel=1e-9
    )


def test_vix_options_parity_skew_and_term_structure() -> None:
    K = np.array([0.14, 0.16, 0.18, 0.2, 0.24, 0.3, 0.36])
    calls, se, fut = M.options(T, K, seed=1)
    puts, _, fut_p = M.options(T, K, "put", seed=1)
    assert fut == fut_p
    np.testing.assert_allclose(calls - puts, fut - K, atol=1e-12)
    # the lognormal approximation is good: rough Bergomi VIX is close to lognormal
    assert np.all(np.abs(calls - M.options_lognormal(T, K)) < 3 * se + 0.002 * calls)
    iv = M.implied_vols(T, K, seed=1)
    assert np.all(np.diff(iv) > 0)  # positive but nearly flat VIX skew
    assert iv[-1] - iv[0] < 0.01
    atm = [M.implied_vols(t, M.futures(t, seed=1)[0], seed=1)[0] for t in (0.05, 0.25, 1.0, 2.0)]
    assert np.all(np.diff(atm) < 0) and atm[0] > 1.3 and atm[-1] < 0.55


@pytest.mark.parametrize(
    "model",
    [
        Heston(0.04, 1.5, 0.04, 0.5, -0.7),
        Heston(0.02, 0.5, 0.05, 0.6, -0.7),
        Heston(0.01, 0.3, 0.05, 1.5, -0.7),
    ],
)
def test_heston_vix_quadrature_against_exact_sampling(model: Heston) -> None:
    m = HestonVIX(model)
    T = 0.5
    c, d, lam = m.variance_law(T)
    assert c * (d + lam) == pytest.approx(model.mean_variance(T), rel=1e-14)
    # mass and first moment are exact, including d < 1 and the extreme d = 0.027
    assert m._expect(T, lambda x: 1.0 + 0.0 * x) == pytest.approx(1.0, abs=1e-12)
    assert m._expect(T, lambda x: m.a * c * x + m.b) == pytest.approx(
        m.expected_vix_squared(T), rel=1e-9
    )
    x = ncx2(d, lam).rvs(size=1_000_000, random_state=np.random.default_rng(0))
    vix = np.sqrt(m.a * c * x + m.b)
    se = vix.std() / np.sqrt(vix.size)
    assert abs(m.futures(T) - vix.mean()) < 3 * se
    K = np.array([0.12, 0.16, 0.2, 0.24, 0.3])
    pay = np.maximum(vix[:, None] - K, 0.0)
    assert np.all(np.abs(m.options(T, K) - pay.mean(0)) < 3 * pay.std(0) / np.sqrt(vix.size))
    np.testing.assert_allclose(
        m.options(T, K) - m.options(T, K, "put"), m.futures(T) - K, atol=1e-14
    )
    if (
        d > 0.1
    ):  # Heston: negative VIX skew, unlike the market (d = 0.027 piles mass at VIX = sqrt(b))
        assert np.all(np.diff(m.implied_vols(T, K)) < 0)
