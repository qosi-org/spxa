"""Tests for payoffs and cell-average projection."""

import numpy as np
import pytest

from spxa.pde.grid import Grid1D
from spxa.pde.payoffs import Payoff, asset_or_nothing, digital, project_payoff, vanilla


class TestPayoffs:
    def test_vanilla(self):
        s = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(vanilla(100, "call")(s), [0, 0, 20])
        np.testing.assert_allclose(vanilla(100, "put")(s), [20, 0, 0])
        assert vanilla(100).kinks == (100.0,)

    def test_digitals(self):
        s = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(digital(100, "call", 2.0)(s), [0, 1.0, 2.0])
        np.testing.assert_allclose(digital(100, "put")(s), [1, 0.5, 0])
        np.testing.assert_allclose(asset_or_nothing(100, "call")(s), [0, 50, 110])
        np.testing.assert_allclose(asset_or_nothing(100, "put")(s), [90, 50, 0])

    def test_algebra(self):
        straddle = vanilla(100, "call") + vanilla(100, "put")
        np.testing.assert_allclose(straddle(np.array([90.0, 110.0])), [10, 10])
        spread = vanilla(90) - vanilla(110)
        np.testing.assert_allclose(spread(np.array([80.0, 100.0, 130.0])), [0, 10, 20])
        assert spread.kinks == (90.0, 110.0)
        np.testing.assert_allclose((2 * vanilla(100))(np.array([105.0])), [10.0])
        with pytest.raises(TypeError):
            vanilla(1) + 3

    def test_validation(self):
        with pytest.raises(ValueError):
            vanilla(-1.0)
        with pytest.raises(ValueError):
            vanilla(1.0, "forward")


class TestProjection:
    def test_pointwise(self):
        g = Grid1D.uniform(0, 200, 11)
        np.testing.assert_allclose(
            project_payoff(vanilla(100), g, "none"), np.maximum(g.nodes - 100, 0)
        )

    def test_average_is_exact_cell_mean(self):
        g = Grid1D.uniform(0.0, 4.0, 5)  # faces 0, .5, 1.5, 2.5, 3.5, 4
        out = project_payoff(vanilla(1.7), g)
        # node 2 cell [1.5, 2.5]: (1/1) * int_{1.7}^{2.5} (s - 1.7) ds = 0.32
        assert out[2] == pytest.approx(0.32)
        assert out[0] == 0.0
        # end points are sampled: their control volumes are one-sided half cells
        assert out[4] == pytest.approx(4.0 - 1.7)

    def test_average_preserves_integral(self):
        g = Grid1D.sinh(0.0, 3.0, 57, center=1.1, alpha=0.2)
        p = digital(1.234)
        out = project_payoff(p, g)
        assert np.dot(out, g.cell_widths) == pytest.approx(3.0 - 1.234, abs=1e-12)

    def test_smooth_payoff_unchanged_to_second_order(self):
        g = Grid1D.uniform(0.0, 1.0, 101)
        p = Payoff(np.sin)
        out = project_payoff(p, g)
        assert np.max(np.abs(out[1:-1] - np.sin(g.nodes[1:-1]))) < 1e-5

    def test_log_space_transform(self):
        g = Grid1D.uniform(np.log(50), np.log(200), 41)
        out = project_payoff(vanilla(100), g, transform=np.exp, inverse=np.log)
        # integral of (e^x - K)^+ over the interior control volumes
        hi = g.faces[-2]
        exact = (np.exp(hi) - 100) - 100 * (hi - np.log(100))
        assert np.dot(out[1:-1], g.cell_widths[1:-1]) == pytest.approx(exact, rel=1e-10)
        assert out[-1] == pytest.approx(100.0)
        with pytest.raises(ValueError):
            project_payoff(vanilla(100), g, transform=np.exp)

    def test_kink_override_and_method_validation(self):
        g = Grid1D.uniform(0.0, 2.0, 11)
        fn = lambda s: np.abs(s - 0.95)  # noqa: E731
        out = project_payoff(fn, g, kinks=(0.95,))
        lo, hi = g.faces[1], g.faces[-2]  # interior control volumes: [0.1, 1.9]
        exact = (0.95 - lo) ** 2 / 2 + (hi - 0.95) ** 2 / 2
        assert np.dot(out[1:-1], g.cell_widths[1:-1]) == pytest.approx(exact)
        with pytest.raises(ValueError):
            project_payoff(fn, g, method="spline")

    def test_kink_method_averages_only_kink_cells(self):
        g = Grid1D.sinh(0.0, 4.0, 41, center=1.7, alpha=0.5)
        avg = project_payoff(vanilla(1.7), g)
        kink = project_payoff(vanilla(1.7), g, method="kink")
        j = int(np.argmin(np.abs(g.nodes - 1.7)))
        assert kink[j] == pytest.approx(avg[j])
        others = np.arange(g.n) != j
        np.testing.assert_allclose(kink[others], np.maximum(g.nodes[others] - 1.7, 0.0), atol=1e-15)
        # off-node kink: only its own control volume is averaged
        k2 = project_payoff(vanilla(1.234), g, method="kink")
        diff = np.abs(k2 - np.maximum(g.nodes - 1.234, 0.0)) > 1e-15
        assert diff.sum() == 1
