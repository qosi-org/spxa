"""Tests for convergence analysis and Richardson extrapolation."""

import numpy as np
import pytest

from spxa.pde.convergence import (
    convergence_study,
    observed_order,
    observed_order_from_values,
    richardson_extrapolate,
)


def model(h, p=2.0, c=0.7, v=1.25):
    return v + c * h**p + 0.1 * h ** (p + 1)


class TestOrders:
    def test_observed_order_from_errors(self):
        errs = [model(0.1 / 2**k) - 1.25 for k in range(4)]
        np.testing.assert_allclose(observed_order(errs), 2.0, atol=0.05)

    def test_observed_order_from_values(self):
        vals = [model(0.1 / 3**k, p=1.0) for k in range(4)]
        np.testing.assert_allclose(observed_order_from_values(vals, ratio=3.0), 1.0, atol=0.05)

    def test_validation(self):
        with pytest.raises(ValueError):
            observed_order([1.0])
        with pytest.raises(ValueError):
            observed_order_from_values([1.0, 2.0])
        with pytest.raises(ValueError):
            richardson_extrapolate(1.0, 2.0, ratio=1.0)


class TestRichardson:
    def test_removes_leading_term(self):
        h = 0.05
        coarse, fine = model(h), model(h / 2)
        extrap = richardson_extrapolate(coarse, fine)
        assert abs(extrap - 1.25) < 0.1 * abs(fine - 1.25)
        assert isinstance(extrap, float)

    def test_exact_for_pure_power(self):
        coarse = 2.0 + 3.0 * 0.2**4
        fine = 2.0 + 3.0 * 0.1**4
        assert richardson_extrapolate(coarse, fine, order=4) == pytest.approx(2.0)

    def test_vectorised(self):
        out = richardson_extrapolate(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        np.testing.assert_allclose(out, [1.5 + 0.5 / 3, 2.5 + 0.5 / 3])


class TestConvergenceStudy:
    def test_with_reference(self):
        study = convergence_study(lambda n: model(1.0 / n), [10, 20, 40, 80], reference=1.25)
        assert study.errors is not None
        np.testing.assert_allclose(study.orders, 2.0, atol=0.1)
        assert abs(study.best_estimate - 1.25) < 1e-5
        text = str(study)
        assert "Richardson" in text and "80" in text

    def test_without_reference(self):
        study = convergence_study(lambda n: model(1.0 / n), [10, 20, 40])
        assert study.errors is None
        assert study.orders.size == 1
        assert "order" in str(study)
        short = convergence_study(lambda n: model(1.0 / n), [10, 20])
        assert short.orders.size == 0

    def test_needs_two_levels(self):
        with pytest.raises(ValueError):
            convergence_study(lambda n: 1.0, [10])
