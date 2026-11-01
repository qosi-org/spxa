"""Unit tests for the Vecer Asian PDE building blocks."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.pde.asian import AsianPDE, vecer_shares


class TestVecerShares:
    @pytest.mark.parametrize("r,q", [(0.05, 0.0), (0.03, 0.07), (0.04, 0.04)])
    def test_continuous_matches_integral(self, r, q):
        T, t = 2.0, 0.7
        ref = quad(lambda s: math.exp(-r * (T - s) - q * (s - t)), t, T)[0] / T
        assert vecer_shares(t, T, r, q) == pytest.approx(ref, rel=1e-10)

    def test_continuous_branch_is_continuous_at_r_equal_q(self):
        a = vecer_shares(0.0, 1.0, 0.05, 0.05)
        b = vecer_shares(0.0, 1.0, 0.05 + 1e-7, 0.05)
        assert a == pytest.approx(b, rel=1e-6)

    def test_discrete_formula(self):
        r, q, T = 0.05, 0.01, 1.0
        fx = [0.25, 0.5, 0.75, 1.0]
        t = 0.3
        ref = sum(math.exp(-r * (T - ti) - q * (ti - t)) for ti in fx if ti > t) / 4
        assert vecer_shares(t, T, r, q, fx) == pytest.approx(ref, rel=1e-14)

    def test_endpoints(self):
        assert vecer_shares(1.0, 1.0, 0.05) == 0.0
        assert vecer_shares(1.0, 1.0, 0.05, 0.0, [0.5, 1.0]) == 0.0
        # zero rates: one share per unit of remaining averaging window
        assert vecer_shares(0.25, 1.0, 0.0, 0.0) == pytest.approx(0.75)


class TestAsianPDEInterface:
    def test_validation(self):
        m = AsianPDE(0.05, 0.2)
        with pytest.raises(ValueError):
            AsianPDE(0.05, -0.2)
        with pytest.raises(ValueError):
            m.price(100, 1.0, 100, fixings=[])
        with pytest.raises(ValueError):
            m.price(100, 1.0, 100, fixings=[0.0, 1.0])
        with pytest.raises(ValueError):
            m.price(100, 1.0, 100, fixings=[0.5, 1.5])
        with pytest.raises(ValueError):
            m.price(100, 1.0, 100, option="straddle")

    def test_result_metadata(self):
        res = AsianPDE(0.05, 0.2).call(
            100.0, 1.0, 100.0, fixings=[0.5, 1.0], n_space=101, n_time=40
        )
        info = res.solution.info
        assert info["segments"] == 2
        assert info["averaging"] == "2 fixings"
        assert info["time_steps"] == 40
        assert math.isnan(res.theta)
        assert "Vecer" in res.method
        assert "AsianPDE" in repr(AsianPDE(0.05, 0.2))

    def test_phi_includes_dividend_discount(self):
        m = AsianPDE(0.05, 0.2, 0.03)
        assert m.phi(0.4, 1.0) == pytest.approx(
            vecer_shares(0.4, 1.0, 0.05, 0.03) * math.exp(-0.012)
        )

    def test_history_is_kept_per_segment(self):
        res = AsianPDE(0.05, 0.2).put(
            100.0, 1.0, 100.0, fixings=[0.5, 1.0], n_space=51, n_time=20, store_history=True
        )
        hist = res.solution.info["segment_histories"]
        assert len(hist) == 2 and all(h.shape[1] == 51 for h in hist)
        assert np.all(res.solution.values >= -1e-12)
