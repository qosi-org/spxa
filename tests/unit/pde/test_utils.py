"""Tests for the internal helpers of spxa.pde."""

import numpy as np
import pytest

from spxa.pde._utils import (
    check_option_type,
    check_positive,
    cubic_interpolate,
    evaluate_coefficient,
)


class TestEvaluateCoefficient:
    def test_scalar_broadcasts(self):
        x = np.linspace(0, 1, 5)
        out = evaluate_coefficient(2.5, x, 0.3)
        assert out.shape == x.shape
        assert np.all(out == 2.5)

    def test_array_passthrough_is_copy(self):
        x = np.linspace(0, 1, 4)
        arr = np.arange(4.0)
        out = evaluate_coefficient(arr, x, 0.0)
        out[0] = 99.0
        assert arr[0] == 0.0

    def test_callable_receives_tau(self):
        x = np.linspace(0, 1, 3)
        out = evaluate_coefficient(lambda s, t: s + t, x, 2.0)
        np.testing.assert_allclose(out, x + 2.0)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            evaluate_coefficient(lambda s, t: 1.0 / s, np.array([0.0, 1.0]), 0.0)


class TestValidation:
    def test_check_positive(self):
        assert check_positive("a", 1.5) == 1.5
        assert check_positive("a", 0.0, allow_zero=True) == 0.0
        with pytest.raises(ValueError):
            check_positive("a", 0.0)
        with pytest.raises(ValueError):
            check_positive("a", float("nan"))

    def test_option_type(self):
        assert check_option_type("C") == "call"
        assert check_option_type("put") == "put"
        with pytest.raises(ValueError):
            check_option_type("straddle")


class TestCubicInterpolate:
    def test_exact_for_cubics(self):
        nodes = np.sort(np.random.default_rng(0).uniform(0, 1, 12))
        f = lambda z: 1 - 2 * z + 3 * z**2 - z**3  # noqa: E731
        xq = np.linspace(nodes[0], nodes[-1], 37)
        np.testing.assert_allclose(cubic_interpolate(nodes, f(nodes), xq), f(xq), atol=1e-12)

    def test_fourth_order(self):
        errs = []
        for n in (21, 41, 81):
            nodes = np.linspace(0, 2, n)
            xq = np.linspace(0.05, 1.95, 101)
            errs.append(np.max(np.abs(cubic_interpolate(nodes, np.sin(nodes), xq) - np.sin(xq))))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 3.7)

    def test_linear_extrapolation(self):
        nodes = np.linspace(0, 1, 6)
        vals = 2 * nodes + 1
        np.testing.assert_allclose(
            cubic_interpolate(nodes, vals, [-0.5, 1.5]), [0.0, 4.0], atol=1e-12
        )
