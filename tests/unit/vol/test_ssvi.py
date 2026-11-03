"""SSVI surfaces (Gatheral & Jacquier 2014, Section 4)."""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

from spxa.vol import (
    SSVI,
    ATMVarianceCurve,
    HestonLikePhi,
    PowerLawPhi,
    SVINatural,
    check_butterfly,
    check_calendar,
)

THETA = ATMVarianceCurve([0.25, 0.5, 1.0, 2.0], [0.012, 0.022, 0.04, 0.075])
K = np.linspace(-2.0, 2.0, 801)


class TestPhi:
    def test_heston_like_against_mpmath(self) -> None:
        phi = HestonLikePhi(1.3)
        with mp.workdps(50):
            for th in np.geomspace(1e-8, 50, 60):
                x = mp.mpf(1.3) * mp.mpf(th)
                ref = mp.mpf(1.3) * (1 - (1 - mp.e ** (-x)) / x) / x
                dref = (1 - mp.e ** (-x) * (1 + x)) / x**2
                assert abs(phi(th) / float(ref) - 1) < 1e-15
                assert abs(phi.theta_phi_derivative(th) / float(dref) - 1) < 1e-15

    def test_power_law_derivative(self) -> None:
        phi = PowerLawPhi(0.8, 0.4)
        th = np.geomspace(1e-4, 10, 30)
        h = 1e-6 * th
        fd = ((th + h) * phi(th + h) - (th - h) * phi(th - h)) / (2 * h)
        assert np.max(np.abs(phi.theta_phi_derivative(th) / fd - 1)) < 1e-8

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            HestonLikePhi(0.0)
        with pytest.raises(ValueError):
            PowerLawPhi(-1.0, 0.5)
        with pytest.raises(ValueError):
            PowerLawPhi(1.0, 1.0)
        with pytest.raises(ValueError):
            HestonLikePhi(1.0)(0.0)


class TestTheta:
    def test_interpolation(self) -> None:
        assert THETA(0.0) == 0.0
        assert THETA(0.125) == pytest.approx(0.006, abs=1e-17)
        assert THETA(0.75) == pytest.approx(0.031, abs=1e-16)
        assert THETA(3.0) == pytest.approx(0.075 + 0.035, abs=1e-15)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            ATMVarianceCurve([1.0, 2.0], [0.04, 0.03])
        with pytest.raises(ValueError, match="increasing"):
            ATMVarianceCurve([1.0, 1.0], [0.04, 0.05])
        with pytest.raises(ValueError, match="negative"):
            THETA(-1.0)


class TestSurface:
    def test_atm_and_slice(self) -> None:
        s = SSVI(-0.6, HestonLikePhi(1.0), THETA)
        for T in (0.3, 1.0, 2.5):
            assert s.total_variance(0.0, T) == pytest.approx(THETA(T), rel=1e-15)
            th = THETA(T)
            nat = SVINatural(0.0, 0.0, -0.6, th, s.phi(th))
            assert np.allclose(s.total_variance(K, T), nat(K), rtol=1e-14)
            assert np.allclose(s.slice(T)(K), nat(K), rtol=1e-13)
        assert np.allclose(s.implied_vol(K, 1.0) ** 2, s.total_variance(K, 1.0), rtol=1e-14)
        with pytest.raises(ValueError):
            SSVI(1.0, HestonLikePhi(1.0), THETA)
        with pytest.raises(ValueError):
            s.implied_vol(0.0, 0.0)

    def test_atm_skew(self) -> None:
        # d w / dk at k = 0 is rho theta phi(theta)
        s = SSVI(-0.6, PowerLawPhi(0.7, 0.4), THETA)
        th = THETA(1.0)
        h = 1e-6
        fd = (s.total_variance(h, 1.0) - s.total_variance(-h, 1.0)) / (2 * h)
        assert fd == pytest.approx(-0.6 * th * s.phi(th), rel=1e-8)

    def test_heston_like_corollary(self) -> None:
        rho = -0.7
        lam = (1 + abs(rho)) / 4  # boundary of Corollary 4.1
        s = SSVI(rho, HestonLikePhi(lam), THETA)
        ths = np.geomspace(1e-4, 5.0, 200)
        assert s.is_arbitrage_free(ths)
        Ts = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        assert all(check_butterfly(s.slice(T), K) for T in Ts)
        assert check_calendar([s.slice(T) for T in Ts], K)

    def test_power_law_remark(self) -> None:
        rho = 0.3
        s = SSVI(rho, PowerLawPhi(2.0 / (1 + rho), 0.5), THETA)
        ths = np.geomspace(1e-4, 5.0, 200)
        assert s.is_arbitrage_free(ths)
        assert all(check_butterfly(s.slice(T), K) for T in (0.1, 1.0, 3.0))

    def test_violations_detected(self) -> None:
        # very steep curvature: theta phi^2 (1 + |rho|) > 4 at short expiries
        s = SSVI(-0.5, PowerLawPhi(5.0, 0.5), THETA)
        assert not np.all(s.butterfly_conditions(np.array([0.001, 0.01])))
        assert not check_butterfly(s.slice(0.05), K)
        # calendar: theta phi decreasing in theta
        bad = SSVI(-0.5, lambda th: 1.0 / np.asarray(th) ** 1.5, THETA)
        assert not np.all(bad.calendar_conditions(np.array([0.01, 0.1])))

    def test_rho_zero_calendar_bound_is_infinite(self) -> None:
        s = SSVI(0.0, HestonLikePhi(2.0), THETA)
        assert np.all(s.calendar_conditions(np.geomspace(1e-3, 1, 10)))
        assert math.isfinite(s.total_variance(1.0, 1.0))
