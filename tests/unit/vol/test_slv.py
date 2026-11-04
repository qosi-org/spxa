"""SLV grids and leverage surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ncx2

from spxa.vol.heston import Heston
from spxa.vol.slv import LeverageSurface, slv_grid, variance_quantile, variance_std

M = Heston(0.04, 1.5, 0.04, 0.6, -0.7)


def test_variance_moments_against_noncentral_chi_square() -> None:
    t = 1.3
    c = M.xi**2 * (1 - np.exp(-M.kappa * t)) / (4 * M.kappa)
    df, nc = 4 * M.kappa * M.theta / M.xi**2, M.v0 * np.exp(-M.kappa * t) / c
    assert variance_std(M, t) == pytest.approx(c * np.sqrt(ncx2.var(df, nc)), rel=1e-12)
    assert float(M.mean_variance(t)) == pytest.approx(c * ncx2.mean(df, nc), rel=1e-12)
    assert variance_quantile(M, t, 0.5) == pytest.approx(c * ncx2.median(df, nc), rel=1e-10)
    assert variance_quantile(M, 0.0, 0.9) == M.v0


def test_grid_layout() -> None:
    g = slv_grid(M, 2.0, n_y=100, n_v=41, vol=0.25)
    assert g.shape == (101, 41) and 0.0 in g.y.nodes and M.v0 in g.v.nodes
    h = np.diff(g.v.nodes)
    assert (
        g.v.nodes[0] == 0.0
        and h[0] < 0.1 * M.v0
        and np.all((h[1:] / h[:-1] > 1) & (h[1:] / h[:-1] < 1.2))
    )
    assert g.y.nodes[-1] == pytest.approx(5 * 0.25 * np.sqrt(2.0))
    # the variance grid holds all but a negligible tail of v_T
    c = M.xi**2 * (1 - np.exp(-2 * M.kappa)) / (4 * M.kappa)
    tail = ncx2.sf(
        g.v.nodes[-1] / c, 4 * M.kappa * M.theta / M.xi**2, M.v0 * np.exp(-2 * M.kappa) / c
    )
    assert tail < 1.01e-6
    assert g.weights.sum() == pytest.approx((g.y.hi - g.y.lo) * g.v.hi)


def test_leverage_interpolation() -> None:
    L = LeverageSurface([0.0, 0.5], [-0.2, 0.0, 0.2], [[1.0, 1.2, 1.4], [0.8, 1.0, 1.1]])
    assert L(0.1, 0.0) == pytest.approx(1.3) and L(0.1, 0.49) == pytest.approx(1.3)
    assert L(0.1, 0.5) == pytest.approx(1.05) and L(0.1, 9.0) == pytest.approx(1.05)
    np.testing.assert_allclose(L(np.array([-1.0, 1.0]), 0.2), [1.0, 1.4])
    assert LeverageSurface.constant()(0.3, 2.0) == 1.0
    with pytest.raises(ValueError):
        LeverageSurface([0.0], [0.0, 1.0], [[1.0, np.nan]])


def test_deterministic_variance_limit() -> None:
    m = M.scaled(0.0)
    assert variance_quantile(m, 1.0, 0.99) == pytest.approx(float(m.mean_variance(1.0)), abs=1e-10)
    g = slv_grid(m, 1.0)
    assert np.all(np.isfinite(g.v.nodes)) and g.v.hi == pytest.approx(3 * M.v0)
