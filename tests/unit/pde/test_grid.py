"""Tests for spxa.pde.grid.Grid1D."""

import numpy as np
import pytest

from spxa.pde.grid import Grid1D


class TestGrid1DConstruction:
    def test_uniform(self):
        g = Grid1D.uniform(0.0, 2.0, 5)
        np.testing.assert_allclose(g.nodes, [0, 0.5, 1.0, 1.5, 2.0])
        assert g.is_uniform and g.n == 5 and len(g) == 5
        assert g.lo == 0.0 and g.hi == 2.0

    def test_validation(self):
        with pytest.raises(ValueError):
            Grid1D([0.0, 1.0])
        with pytest.raises(ValueError):
            Grid1D([0.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            Grid1D([0.0, np.nan, 1.0])
        with pytest.raises(ValueError):
            Grid1D.uniform(1.0, 0.0, 10)
        with pytest.raises(ValueError):
            Grid1D.uniform(0.0, 1.0, 2)
        with pytest.raises(ValueError):
            Grid1D.sinh(0.0, 1.0, 10, center=0.5, alpha=-1.0)

    def test_nodes_read_only(self):
        g = Grid1D.uniform(0, 1, 4)
        with pytest.raises(ValueError):
            g.nodes[0] = 5.0

    def test_sinh_endpoints_and_monotone(self):
        g = Grid1D.sinh(0.0, 400.0, 101, center=100.0, alpha=20.0)
        assert g.lo == 0.0 and g.hi == 400.0
        assert np.all(np.diff(g.nodes) > 0)
        assert not g.is_uniform

    def test_sinh_concentrates_at_center(self):
        g = Grid1D.sinh(0.0, 400.0, 201, center=100.0, alpha=10.0)
        h = g.spacing
        mid = 0.5 * (g.nodes[1:] + g.nodes[:-1])
        h_center = h[np.argmin(np.abs(mid - 100.0))]
        assert h_center < 0.25 * h.max()
        assert h_center < (400.0 / 200)

    def test_sinh_places_center_on_node(self):
        g = Grid1D.sinh(0.0, 400.0, 101, center=100.0, alpha=20.0)
        j = int(np.argmin(np.abs(g.nodes - 100.0)))
        assert g.nodes[j] == 100.0
        ratio = g.spacing[j] / g.spacing[j - 1]
        assert abs(ratio - 1.0) < 0.05
        free = Grid1D.sinh(0.0, 400.0, 101, center=100.0, alpha=20.0, align_center=False)
        assert np.all(np.diff(free.nodes) > 0)
        outside = Grid1D.sinh(0.0, 1.0, 11, center=2.0, alpha=0.5)
        assert outside.lo == 0.0 and outside.hi == 1.0

    def test_sinh_is_smooth_map(self):
        # successive spacing ratios tend to 1 under refinement (smooth stretching)
        g = Grid1D.sinh(0.0, 1.0, 401, center=0.3, alpha=0.05)
        ratio = g.spacing[1:] / g.spacing[:-1]
        assert np.max(np.abs(ratio - 1.0)) < 0.05


class TestSinhAlignment:
    def test_points_land_on_nodes(self):
        g = Grid1D.sinh(0.0, 400.0, 201, center=100.0, alpha=100.0, align=[90.0, 117.3])
        for p in (90.0, 100.0, 117.3):
            assert np.min(np.abs(g.nodes - p)) == 0.0
        assert np.all(np.diff(g.nodes) > 0)

    def test_center_only_matches_split_formula(self):
        lo, hi, n, c, a = 0.0, 300.0, 101, 100.0, 50.0
        g = Grid1D.sinh(lo, hi, n, center=c, alpha=a)
        c1, c2 = np.arcsinh((lo - c) / a), np.arcsinh((hi - c) / a)
        j = int(round(-c1 / (c2 - c1) * (n - 1)))
        left = c + a * np.sinh(c1 * (1.0 - np.arange(j + 1) / j))
        right = c + a * np.sinh(c2 * np.arange(1, n - j) / (n - 1 - j))
        np.testing.assert_allclose(g.nodes, np.concatenate([left, right]), atol=1e-12)

    def test_spacing_ratio_smooth_at_aligned_points(self):
        for n in (101, 401, 1601):
            g = Grid1D.sinh(0.0, 400.0, n, center=100.0, alpha=100.0, align=[93.7])
            h = np.diff(g.nodes)
            ratio = np.abs(np.log(h[1:] / h[:-1]))
            # O(1/n) with a constant set by the xi-distance between aligned points
            assert ratio.max() * n < 40.0

    def test_alignment_without_center(self):
        g = Grid1D.sinh(
            95.0, 400.0, 101, center=100.0, alpha=100.0, align_center=False, align=[110.0]
        )
        assert np.min(np.abs(g.nodes - 110.0)) == 0.0
        assert np.min(np.abs(g.nodes - 100.0)) > 0.0

    def test_points_outside_are_ignored(self):
        g = Grid1D.sinh(0.0, 10.0, 21, center=5.0, align=[-1.0, 10.0, 12.0])
        h = Grid1D.sinh(0.0, 10.0, 21, center=5.0)
        np.testing.assert_array_equal(g.nodes, h.nodes)

    def test_too_few_nodes(self):
        with pytest.raises(ValueError):
            Grid1D.sinh(0.0, 10.0, 4, center=5.0, align=[2.0, 3.0, 4.0])


class TestGrid1DOperations:
    def test_faces_and_cell_widths(self):
        g = Grid1D([0.0, 1.0, 3.0, 4.0])
        np.testing.assert_allclose(g.faces, [0.0, 0.5, 2.0, 3.5, 4.0])
        np.testing.assert_allclose(g.cell_widths, [0.5, 1.5, 1.5, 0.5])
        assert np.isclose(g.cell_widths.sum(), g.hi - g.lo)

    def test_with_points_snaps_exactly(self):
        g = Grid1D.sinh(0.0, 300.0, 81, center=100.0, alpha=15.0).with_points([97.3, 120.0])
        assert np.any(g.nodes == 97.3)
        assert np.any(g.nodes == 120.0)
        assert g.n == 81

    def test_with_points_ignores_outside(self):
        g = Grid1D.uniform(0, 1, 11)
        assert np.array_equal(g.with_points([-1.0, 1.0, 2.0]).nodes, g.nodes)

    def test_with_points_conflict(self):
        g = Grid1D.uniform(0, 1, 5)
        with pytest.raises(ValueError):
            g.with_points([0.49, 0.51])

    def test_refine(self):
        g = Grid1D([0.0, 1.0, 3.0])
        np.testing.assert_allclose(g.refine().nodes, [0.0, 0.5, 1.0, 2.0, 3.0])

    def test_locate_contains(self):
        g = Grid1D.uniform(0, 1, 11)
        assert g.locate(0.31) == 3
        assert g.contains(0.5) and not g.contains(1.5)

    def test_interpolate(self):
        g = Grid1D.sinh(0.0, 2.0, 41, center=1.0, alpha=0.3)
        vals = np.exp(g.nodes)
        assert abs(float(g.interpolate(vals, 1.234)[0]) - np.exp(1.234)) < 1e-6
        with pytest.raises(ValueError):
            g.interpolate(vals[:-1], 1.0)

    def test_repr(self):
        assert "uniform" in repr(Grid1D.uniform(0, 1, 3))
