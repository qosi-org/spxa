"""Hagan's SABR implied volatilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.market.implied import implied_normal_volatility
from spxa.rates.sabr import SABR, sabr_lognormal_vol, sabr_normal_vol, z_over_chi
from spxa.zoo.beyond.sabr import SABRProcess

F = 0.03
K = F * np.exp(np.array([-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5]))
M = SABR(0.035, 0.5, -0.3, 0.4)


def test_matches_the_zoo_implementation() -> None:
    zoo = SABRProcess(alpha=0.035, beta=0.5, nu=0.4, rho=-0.3, f0=F)
    for T in (0.25, 2.0, 10.0):
        assert M.lognormal_vol(F, K, T) == pytest.approx(zoo.implied_volatility(K, T), rel=1e-12)


def test_degenerate_cases_are_exact() -> None:
    assert sabr_lognormal_vol(F, K, 5.0, 0.2, 1.0, 0.3, 0.0) == pytest.approx(
        np.full(7, 0.2), rel=1e-15
    )
    assert sabr_normal_vol(F, K, 5.0, 0.01, 0.0, 0.3, 0.0) == pytest.approx(
        np.full(7, 0.01), rel=1e-15
    )
    with pytest.raises(ValueError):
        SABR(0.02, 1.2, 0.0, 0.3)
    with pytest.raises(ValueError):
        M.lognormal_vol(F, -0.01, 1.0)


def test_z_over_chi() -> None:
    for rho in (-0.7, 0.0, 0.5):
        for z in (-0.8, -1e-2, 3e-3, 0.4, 2.0):
            chi = quad(
                lambda u: (1 - 2 * rho * u + u * u) ** -0.5, 0.0, z, epsabs=0.0, epsrel=1e-12
            )[0]
            assert z_over_chi(z, rho) == pytest.approx(z / chi, rel=1e-12)
        # the series and the closed form meet at |z| = 1e-5
        for z in (0.99999e-5, 1.00001e-5, -1.00001e-5):
            chi = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
            assert z_over_chi(z, rho) == pytest.approx(z / chi, rel=1e-10)
    assert z_over_chi(0.0, 0.3) == 1.0


def test_normal_and_lognormal_formulas_agree() -> None:
    # both are first-order expansions of the same model; at the money they differ by O(T^2)
    otm = np.where(K < F, "put", "call")
    gaps = []
    for T in (1.0, 0.5, 0.25):
        sn = implied_normal_volatility(M.price(F, K, T, option=otm), F, K, T, option=otm)
        gap = np.abs(sn / M.normal_vol(F, K, T) - 1)
        assert np.max(gap) < 2e-3
        gaps.append(gap[3])
    assert gaps[0] / gaps[1] == pytest.approx(4.0, rel=0.02)
    assert gaps[1] / gaps[2] == pytest.approx(4.0, rel=0.02)


def test_shift_translates_the_smile() -> None:
    sh = SABR(0.035, 0.5, -0.3, 0.4, shift=0.02)
    Fn, Kn = -0.003, np.array([-0.01, -0.003, 0.0, 0.01])
    assert sh.lognormal_vol(Fn, Kn, 2.0) == pytest.approx(
        M.lognormal_vol(Fn + 0.02, Kn + 0.02, 2.0), rel=1e-15
    )
    assert sh.normal_vol(Fn, Kn, 2.0) == pytest.approx(
        M.normal_vol(Fn + 0.02, Kn + 0.02, 2.0), rel=1e-15
    )
    for vol in ("lognormal", "normal"):
        c = sh.price(Fn, Kn, 2.0, 0.95, "call", vol)
        p = sh.price(Fn, Kn, 2.0, 0.95, "put", vol)
        assert c - p == pytest.approx(0.95 * (Fn - Kn), abs=1e-15)
    with pytest.raises(ValueError):
        sh.price(Fn, Kn, 2.0, vol="cev")


def test_obloj_short_expiry_limit() -> None:
    Kw = F * np.exp(np.linspace(-1.0, 1.0, 9))
    # nu = 0 (CEV): the T -> 0 implied vol is the harmonic mean of the local vol alpha u^(beta - 1)
    for beta in (0.0, 0.3, 0.7):
        ob = sabr_lognormal_vol(F, Kw, 0.0, 0.035, beta, -0.3, 0.0, method="obloj")
        for k, v in zip(Kw, ob):
            dist = quad(lambda u: u**-beta / 0.035, k, F, epsabs=0.0, epsrel=1e-13)[0]
            expect = 0.035 * F ** (beta - 1) if k == F else np.log(F / k) / dist
            assert v == pytest.approx(expect, rel=1e-11)
    # beta = 1: z = nu ell / alpha
    ell = np.log(F / Kw)
    z = 0.4 * ell / 0.2
    corr = -0.3 * 0.4 * 0.2 / 4 + (2 - 3 * 0.09) / 24 * 0.16
    expect = 0.2 * z_over_chi(z, -0.3) * (1 + corr * 1.0)
    assert sabr_lognormal_vol(F, Kw, 1.0, 0.2, 1.0, -0.3, 0.4, method="obloj") == pytest.approx(
        expect, rel=1e-13
    )


def test_obloj_against_hagan() -> None:
    for T in (0.0, 1.0, 10.0):
        assert M.lognormal_vol(F, F, T, "obloj") == pytest.approx(
            M.lognormal_vol(F, F, T), rel=1e-14
        )
        near = F * np.exp(np.array([-0.05, 0.05]))
        assert M.lognormal_vol(F, near, T, "obloj") == pytest.approx(
            M.lognormal_vol(F, near, T), rel=1e-5
        )
    far = F * np.exp(np.array([-1.0, 1.0]))
    gap = np.abs(M.lognormal_vol(F, far, 0.0, "obloj") - M.lognormal_vol(F, far, 0.0))
    assert np.all(gap > 1e-4)
    sh = SABR(0.035, 0.5, -0.3, 0.4, shift=0.01)
    assert sh.lognormal_vol(-0.005, 0.0, 2.0, "obloj") == pytest.approx(
        M.lognormal_vol(0.005, 0.01, 2.0, "obloj"), rel=1e-14
    )
    with pytest.raises(ValueError):
        M.lognormal_vol(F, F, 1.0, "west")
