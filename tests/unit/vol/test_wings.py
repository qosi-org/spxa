"""Lee's moment formula, density tails and the Rogers–Tehranchi bound."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.vol import (
    SSVI,
    PowerLawPhi,
    SVIRaw,
    WingAsymptotics,
    durrleman_g,
    lee_moment_from_slope,
    lee_slope_from_moment,
    risk_neutral_density,
    rogers_tehranchi_slope,
    wing_asymptotics,
)

GOOD = SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2)


class TestLeeFormula:
    def test_textbook_form_and_endpoints(self) -> None:
        for x in (0.0, 0.01, 0.125, 1.0, 3.7, 50.0):
            textbook = 2 - 4 * (math.sqrt(x * x + x) - x)
            assert lee_slope_from_moment(x) == pytest.approx(textbook, abs=1e-14)
        assert lee_slope_from_moment(0.0) == 2.0  # no moment beyond the first: slope 2
        assert lee_slope_from_moment(math.inf) == 0.0
        assert lee_moment_from_slope(2.0) == 0.0
        assert lee_moment_from_slope(0.0) == math.inf
        assert lee_moment_from_slope(1.0) == 0.125  # (2-1)^2 / 8

    def test_inverse_pair(self) -> None:
        for beta in np.linspace(0.01, 2.0, 40):
            x = lee_moment_from_slope(beta)
            assert lee_slope_from_moment(x) == pytest.approx(beta, rel=1e-14)

    def test_large_moments_without_cancellation(self) -> None:
        # psi(x) ~ 1/(2x) - 1/(4x^2): the stable form keeps full relative accuracy
        x = 1e9
        assert lee_slope_from_moment(x) == pytest.approx(1 / (2 * x) - 1 / (4 * x * x), rel=1e-15)
        assert lee_moment_from_slope(lee_slope_from_moment(x)) == pytest.approx(x, rel=1e-14)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            lee_slope_from_moment(-0.1)
        with pytest.raises(ValueError, match="Lee"):
            lee_moment_from_slope(2.1)


class TestWingAsymptotics:
    def test_svi_slopes_exact_and_numerical(self) -> None:
        exact = wing_asymptotics(GOOD)
        assert (exact.left_slope, exact.right_slope) == (GOOD.left_slope, GOOD.right_slope)
        numeric = wing_asymptotics(lambda k: GOOD.total_variance(k))
        assert numeric.left_slope == pytest.approx(0.14, abs=1e-9)
        assert numeric.right_slope == pytest.approx(0.06, abs=1e-9)
        # moments: p = (2 - 0.06)^2 / 0.48, q = (2 - 0.14)^2 / 1.12
        assert exact.right_moment == pytest.approx(1.94**2 / 0.48, rel=1e-15)
        assert exact.left_moment == pytest.approx(1.86**2 / 1.12, rel=1e-15)
        assert exact.right_decay == pytest.approx(1 / 0.12 + 0.5 + 0.06 / 8, rel=1e-14)
        assert exact.left_decay == pytest.approx(1 / 0.28 - 0.5 + 0.14 / 8, rel=1e-14)
        assert exact.satisfies_lee_bound

    def test_ssvi_slice(self) -> None:
        # SSVI wings: theta phi (1 +- rho) / 2
        s = SSVI(rho=-0.5, phi=PowerLawPhi(eta=1.2, gamma=0.3), theta=lambda t: 0.05 * t)
        th = 0.05
        ph = float(PowerLawPhi(eta=1.2, gamma=0.3)(th))
        wa = wing_asymptotics(s.slice(1.0))
        assert wa.right_slope == pytest.approx(th * ph * 0.5 / 2, rel=1e-13)
        assert wa.left_slope == pytest.approx(th * ph * 1.5 / 2, rel=1e-13)

    def test_lee_violation(self) -> None:
        steep = SVIRaw(0.01, 1.8, 0.3, 0.0, 0.1)  # right slope 2.34
        with pytest.raises(ValueError, match="right wing"):
            wing_asymptotics(steep)
        assert not WingAsymptotics(0.5, 2.5).satisfies_lee_bound


def log_density(sl: SVIRaw, k: np.ndarray) -> np.ndarray:
    """log p(k) = log g - log sqrt(2 pi w) - d_-^2 / 2, without underflow."""
    k = np.asarray(k, float)
    w = sl.total_variance(k)
    g = durrleman_g(k, w, sl.dw(k), sl.d2w(k))
    dm = -k / np.sqrt(w) - np.sqrt(w) / 2
    return np.log(g) - 0.5 * np.log(2 * np.pi * w) - 0.5 * dm * dm


class TestDensityTails:
    def test_log_density_slopes(self) -> None:
        # d/dk log p(k) -> -lambda_R (k -> inf) and +lambda_L (k -> -inf);
        # the sub-exponential factor contributes O(log k / k)
        wa = wing_asymptotics(GOOD)
        k1, k2 = 2000.0, 4000.0
        ks = np.array([-30.0, -1.0, 0.5, 30.0])
        assert np.allclose(
            log_density(GOOD, ks), np.log(risk_neutral_density(GOOD, ks)), atol=1e-12
        )
        lp = log_density(GOOD, np.array([k1, k2, -k1, -k2]))
        right = (lp[1] - lp[0]) / (k2 - k1)
        left = (lp[2] - lp[3]) / (k2 - k1)
        assert right == pytest.approx(-wa.right_decay, abs=2e-3)
        assert left == pytest.approx(wa.left_decay, abs=2e-3)

    def test_moments_finite_below_and_infinite_above_critical(self) -> None:
        # E[S^{1+p}] on [0, K]: converges for p < p~, grows without bound for p > p~
        p_crit = wing_asymptotics(GOOD).right_moment

        def partial(p: float, K: float) -> float:
            # integrate in log space to avoid overflow: e^{(1+p)k} p(k)
            def f(k: float) -> float:
                return math.exp((1 + p) * k + float(log_density(GOOD, np.array(k))))

            return quad(f, 0.0, K, limit=400)[0]

        below = [partial(p_crit - 0.5, K) for K in (100.0, 200.0, 300.0)]
        assert abs(below[2] - below[1]) < 1e-12 * below[1]
        above = [partial(p_crit + 0.5, K) for K in (100.0, 200.0, 300.0)]
        assert above[1] > 1e10 * above[0] and above[2] > 1e10 * above[1]


class TestRogersTehranchi:
    def test_bound(self) -> None:
        assert rogers_tehranchi_slope(GOOD) < GOOD.b * (1 + abs(GOOD.rho)) + 1e-12
        assert rogers_tehranchi_slope(SVIRaw(0.01, 2.5, 0.8, 0.0, 0.1)) > 4.0
