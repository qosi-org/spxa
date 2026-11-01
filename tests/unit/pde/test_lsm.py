"""Unit tests for least-squares Monte Carlo."""

import numpy as np
import pytest

from spxa.pde.lsm import LSMResult, _basis, gbm_paths, longstaff_schwartz, lsm_american
from spxa.pde.reference import bs_price


def put(K):
    return lambda s: np.maximum(K - s, 0.0)


class TestBasis:
    def test_laguerre(self):
        x = np.array([0.0, 1.0, 2.5])
        B = _basis(x, 3, "laguerre")
        w = np.exp(-x / 2)
        expected = np.column_stack([np.ones(3), w, w * (1 - x), w * (1 - 2 * x + x**2 / 2)])
        np.testing.assert_allclose(B, expected, atol=1e-14)

    def test_monomial(self):
        x = np.array([0.5, 2.0])
        np.testing.assert_allclose(_basis(x, 2, "monomial"), [[1, 0.5, 0.25], [1, 2, 4]])


class TestPaths:
    def test_gbm(self):
        t, p = gbm_paths(100.0, 0.05, 0.2, 2.0, 8, 200_000, q=0.01, seed=3)
        assert p.shape == (200_000, 9) and t[-1] == 2.0 and np.all(p[:, 0] == 100.0)
        # antithetic halves mirror each other in log space
        lr = np.log(p[:, -1] / 100.0)
        mu = (0.05 - 0.01 - 0.02) * 2.0
        np.testing.assert_allclose(lr[:100_000] + lr[100_000:], 2 * mu, atol=1e-10)
        se = p[:, -1].std() / np.sqrt(200_000)
        assert abs(p[:, -1].mean() - 100.0 * np.exp(0.08)) < 4 * se
        with pytest.raises(ValueError, match="even"):
            gbm_paths(100.0, 0.05, 0.2, 1.0, 4, 11)


class TestLongstaffSchwartz:
    def test_validation(self):
        t, p = gbm_paths(40.0, 0.06, 0.2, 1.0, 4, 100, seed=0)
        with pytest.raises(ValueError, match="shape"):
            longstaff_schwartz(p[:, :3], t, put(40.0), 0.06)
        with pytest.raises(ValueError, match="start at 0"):
            longstaff_schwartz(p, t + 0.1, put(40.0), 0.06)
        with pytest.raises(ValueError, match="basis"):
            longstaff_schwartz(p, t, put(40.0), 0.06, basis="hermite")
        with pytest.raises(ValueError, match="degree"):
            longstaff_schwartz(p, t, put(40.0), 0.06, degree=0)
        with pytest.raises(ValueError, match="regression_paths"):
            longstaff_schwartz(p, t, put(40.0), 0.06, regression_paths=p[:, :3])

    def test_single_date_is_european(self):
        t, p = gbm_paths(40.0, 0.06, 0.2, 1.0, 1, 400_000, seed=5)
        res = longstaff_schwartz(p, t, put(40.0), 0.06)
        ref = bs_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.0, option="put")
        assert abs(res.price - ref) < 3 * res.std_error
        assert res.exercise_probability == 0.0

    def test_immediate_exercise(self):
        res = lsm_american(40.0, 1.0, 10.0, 0.06, 0.2, n_paths=2000, n_dates=5)
        assert res.price == pytest.approx(30.0) and res.info["immediate_exercise"]
        assert res.std_error == 0.0

    def test_result_fields(self):
        res = lsm_american(40.0, 1.0, 40.0, 0.06, 0.2, n_paths=20_000, n_dates=10, seed=2)
        assert isinstance(res, LSMResult) and float(res) == res.price
        lo, hi = res.confidence_interval()
        assert lo < res.price < hi and hi - lo == pytest.approx(2 * 1.96 * res.std_error)
        assert 0.0 < res.exercise_probability < 1.0
        assert res.info["two_pass"] and res.info["n_dates"] == 10
        b = res.boundary[1:-1]
        b = b[np.isfinite(b)]
        assert np.all(b < 40.0)

    def test_call_without_dividends_is_not_exercised(self):
        res = lsm_american(100.0, 1.0, 100.0, 0.05, 0.25, option="call", n_paths=100_000, seed=4)
        ref = bs_price(100.0, 100.0, 1.0, 0.05, 0.25, 0.0)
        assert abs(res.price - ref) < 3 * res.std_error + 0.02
        assert res.exercise_probability < 0.01
