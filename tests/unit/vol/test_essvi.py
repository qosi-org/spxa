"""eSSVI slices, calendar conditions and arbitrage-free interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol import (
    ESSVI,
    ESSVISlice,
    SVINatural,
    anchored_theta,
    calibrate_essvi,
    check_butterfly,
    check_calendar,
    essvi_calendar_free,
)

# Corbetta, Cohort, Laachir & Martini (2019), Table 1: SPX, 8 January 2018
TABLE1 = [
    (0.030137, 0.0001, 0.012, -0.224),
    (0.106849, 0.0006, 0.032, -0.453),
    (0.183562, 0.0014, 0.049, -0.495),
    (0.279452, 0.0025, 0.066, -0.578),
    (0.432877, 0.0049, 0.089, -0.610),
    (0.701370, 0.0100, 0.116, -0.672),
    (0.950685, 0.0158, 0.131, -0.704),
    (1.027397, 0.0174, 0.134, -0.704),
    (1.180822, 0.0215, 0.145, -0.725),
    (1.449315, 0.0292, 0.165, -0.725),
    (1.947945, 0.0444, 0.191, -0.746),
    (2.945205, 0.0750, 0.243, -0.724),
]
TIMES = [r[0] for r in TABLE1]
SLICES = [ESSVISlice(th, ps, rh) for _, th, ps, rh in TABLE1]
SURF = ESSVI(TIMES, SLICES)
K = np.linspace(-1.5, 1.0, 501)


def test_slice_is_natural_svi() -> None:
    s = ESSVISlice(0.04, 0.2, -0.6)
    nat = SVINatural(0.0, 0.0, -0.6, 0.04, 5.0)
    assert np.allclose(s(K), nat(K), rtol=1e-14)
    assert np.allclose(s.to_raw()(K), nat(K), rtol=1e-13)
    assert s.total_variance(0.0) == pytest.approx(0.04, rel=1e-15)
    with pytest.raises(ValueError):
        ESSVISlice(0.04, 0.0, 0.1)
    with pytest.raises(ValueError):
        ESSVISlice(0.04, 0.1, -1.0)


def test_butterfly_bounds() -> None:
    # psi^2 (1 + |rho|) <= 4 theta binds for small theta
    th, rho = 0.01, 0.5
    edge = 2 * np.sqrt(th / (1 + rho))
    assert ESSVISlice(th, edge * (1 - 1e-12), rho).butterfly_free()
    assert not ESSVISlice(th, edge * (1 + 1e-9), rho).butterfly_free()
    # psi (1 + |rho|) < 4 binds for large theta
    assert not ESSVISlice(10.0, 4 / 1.5, rho).butterfly_free()
    assert ESSVISlice(10.0, 4 / 1.5 * (1 - 1e-12), rho).butterfly_free()


def test_published_spx_calibration_is_arbitrage_free() -> None:
    assert SURF.is_arbitrage_free()
    assert SURF.violations() == []
    assert all(check_butterfly(s, K) for s in SLICES)
    assert check_calendar(SLICES, K)


def test_interpolation_scheme() -> None:
    for T, s in zip(TIMES, SLICES):
        p = SURF.params(T)
        assert (p.theta, p.psi, p.rho) == pytest.approx((s.theta, s.psi, s.rho), abs=1e-15)
    a, b = SLICES[5], SLICES[6]
    mid = SURF.params(0.5 * (TIMES[5] + TIMES[6]))
    assert mid.theta == pytest.approx(0.5 * (a.theta + b.theta), rel=1e-14)
    assert mid.psi == pytest.approx(0.5 * (a.psi + b.psi), rel=1e-14)
    assert mid.rho * mid.psi == pytest.approx(0.5 * (a.rho * a.psi + b.rho * b.psi), rel=1e-14)
    short = SURF.params(TIMES[0] / 4)
    assert (short.theta, short.psi, short.rho) == pytest.approx(
        (0.000025, 0.003, -0.224), rel=1e-12
    )
    slope = (0.0750 - 0.0444) / (TIMES[-1] - TIMES[-2])
    long_ = SURF.params(4.0)
    assert long_.theta == pytest.approx(0.0750 + slope * (4.0 - TIMES[-1]), rel=1e-14)
    assert (long_.psi, long_.rho) == (0.243, -0.724)
    assert np.allclose(SURF.implied_vol(K, 1.0) ** 2, SURF.total_variance(K, 1.0), rtol=1e-14)
    assert np.allclose(SURF.slice(1.0)(K), SURF.total_variance(K, 1.0), rtol=1e-13)


def test_interpolated_surface_has_no_static_arbitrage() -> None:
    # Corbetta et al. Section 5: the interpolation preserves both conditions
    Ts = np.concatenate([np.linspace(0.005, 3.0, 120), [3.5, 5.0]])
    params = [SURF.params(T) for T in Ts]
    assert all(p.butterfly_free() for p in params)
    assert all(essvi_calendar_free(a, b, tol=1e-15) for a, b in zip(params, params[1:]))
    assert all(check_butterfly(p, K) for p in params)
    assert check_calendar(params, K, tol=1e-15)


def test_calendar_violation_crosses() -> None:
    s1 = ESSVISlice(0.04, 0.2, -0.7)
    s2 = ESSVISlice(0.041, 0.21, 0.5)  # skew flips: left wing of s2 falls below s1
    assert not essvi_calendar_free(s1, s2)
    assert not check_calendar([s1, s2], np.linspace(-3, 3, 601))
    surf = ESSVI([0.5, 1.0], [s1, s2])
    assert not surf.is_arbitrage_free()
    assert surf.violations() == ["calendar between T=0.5 and T=1"]
    assert not essvi_calendar_free(s2, s1)  # theta and psi decrease


def test_validation() -> None:
    with pytest.raises(ValueError, match="one slice per expiry"):
        ESSVI([1.0, 2.0], SLICES[:1])
    with pytest.raises(ValueError, match="increasing"):
        ESSVI([1.0, 1.0], SLICES[:2])
    with pytest.raises(ValueError, match="non-negative"):
        ESSVI([1.0], SLICES[:1], long_theta_slope=-0.1)
    with pytest.raises(ValueError):
        SURF.params(0.0)
    single = ESSVI([1.0], [ESSVISlice(0.04, 0.2, -0.5)])
    assert single.long_theta_slope == pytest.approx(0.04, abs=1e-17)
    bad = ESSVI([1.0], [ESSVISlice(0.0004, 0.2, -0.5)])
    assert bad.violations() == ["butterfly at T=1"] and not bad.is_arbitrage_free()


class TestCalibration:
    ks = [np.linspace(-3 * np.sqrt(th), 1.5 * np.sqrt(th), 21) for _, th, _, _ in TABLE1]

    def test_anchored_theta_is_exact(self) -> None:
        s = ESSVISlice(0.04, 0.2, -0.6)
        for k_star in (-0.3, -0.05, 0.0, 0.13):
            th = anchored_theta(-0.6, 0.2, k_star, s(k_star))
            assert th == pytest.approx(0.04, rel=1e-14)
            assert ESSVISlice(th, 0.2, -0.6)(k_star) == pytest.approx(s(k_star), rel=1e-14)

    def test_recovers_published_surface(self) -> None:
        ws = [SURF.total_variance(k, t) for k, t in zip(self.ks, TIMES)]
        fit = calibrate_essvi(TIMES, self.ks, ws)
        assert fit.is_arbitrage_free()
        for got, (_, th, ps, rh) in zip(fit.slices, TABLE1):
            assert (got.theta, got.psi, got.rho) == pytest.approx((th, ps, rh), rel=1e-7, abs=1e-8)

    def test_noisy_quotes_stay_arbitrage_free(self) -> None:
        rng = np.random.default_rng(11)
        ws = [
            SURF.total_variance(k, t) * (1 + 0.02 * rng.standard_normal(k.size))
            for k, t in zip(self.ks, TIMES)
        ]
        fit = calibrate_essvi(TIMES, self.ks, ws)
        assert fit.is_arbitrage_free()
        params = [fit.params(T) for T in np.linspace(0.01, 3.5, 80)]
        assert all(p.butterfly_free() for p in params)
        assert check_calendar(params, K, tol=1e-15)
        for got, (_, th, ps, rh) in zip(fit.slices, TABLE1):
            assert abs(got.rho - rh) < 0.15 and abs(got.psi / ps - 1) < 0.2

    def test_infeasible_slice(self) -> None:
        k = np.array([-0.2, 0.0, 0.2])
        s1 = ESSVISlice(0.04, 0.2, -0.5)
        s2 = ESSVISlice(0.02, 0.2, -0.5)  # ATM variance falls
        with pytest.raises(ValueError, match="previous expiry"):
            calibrate_essvi([0.5, 1.0], [k, k], [s1(k), s2(k)])
        with pytest.raises(ValueError, match="one"):
            calibrate_essvi([0.5], [k, k], [s1(k)])
        with pytest.raises(ValueError, match="at least 3"):
            calibrate_essvi([0.5], [k[:2]], [s1(k[:2])])
