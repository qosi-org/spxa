"""VIX in rough Bergomi and Heston."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.vol.vix import VIX_WINDOW, RBergomiVIX


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
