"""Gaussian HJM."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.capfloor import cap_floor
from spxa.rates.g2pp import G2PP
from spxa.rates.hjm import GaussianHJM, exponential_vol, hump_vol
from spxa.rates.hull_white import HullWhite
from spxa.rates.indices import EURIBOR6M
from tests.unit.rates.test_capfloor import DISC, PROJ, REF
from tests.unit.rates.test_hull_white import CURVE


def test_exponential_vol_is_hull_white() -> None:
    a, s = 0.07, 0.011
    hjm = GaussianHJM(CURVE, [exponential_vol(s, a)])
    hw = HullWhite(CURVE, a, s)
    tau = np.array([0.5, 3.0, 12.0])
    assert hjm.Sigma(1.0, 1.0 + tau)[0] == pytest.approx(s * (1 - np.exp(-a * tau)) / a, rel=1e-14)
    assert hjm.drift(1.0, 1.0 + tau) == pytest.approx(
        s**2 * np.exp(-a * tau) * (1 - np.exp(-a * tau)) / a, rel=1e-14
    )
    for T, S in ((1.0, 3.0), (5.0, 15.0)):
        for K in (0.8, 0.95):
            assert hjm.zcb_option(T, S, K) == pytest.approx(hw.zcb_option(T, S, K), rel=1e-11)
            assert hjm.zcb_option(T, S, K, "put") == pytest.approx(
                hw.zcb_option(T, S, K, "put"), rel=1e-11
            )
    hjm_d = GaussianHJM(DISC, [exponential_vol(s, a)])
    cap = cap_floor(REF, "5Y", 0.035, EURIBOR6M, notional=1e6)
    assert hjm_d.cap(cap, PROJ) == pytest.approx(HullWhite(DISC, a, s).cap(cap, PROJ), rel=1e-11)


def test_two_factor_exponential_is_g2pp() -> None:
    a, s, b, eta, rho = 0.5, 0.012, 0.05, 0.009, -0.7

    def v1(t: float, T: Any) -> Any:
        tau = np.asarray(T) - t
        return s * np.exp(-a * tau) + eta * rho * np.exp(-b * tau)

    def v2(t: float, T: Any) -> Any:
        return eta * math.sqrt(1 - rho**2) * np.exp(-b * (np.asarray(T) - t))

    hjm = GaussianHJM(CURVE, [v1, v2])
    g = G2PP(CURVE, a, s, b, eta, rho)
    for T, S, K in ((1.0, 3.0, 0.93), (4.0, 14.0, 0.7)):
        assert hjm.zcb_option(T, S, K) == pytest.approx(g.zcb_option(T, S, K), rel=1e-10)
    cap = cap_floor(REF, "4Y", 0.034, EURIBOR6M, notional=1e6)
    assert GaussianHJM(DISC, [v1, v2]).cap(cap, PROJ) == pytest.approx(
        G2PP(DISC, a, s, b, eta, rho).cap(cap, PROJ), rel=1e-10
    )


def test_hump_variance_against_quadrature_and_parity() -> None:
    v = hump_vol(0.004, 0.006, 0.4)
    hjm = GaussianHJM(CURVE, [v])
    T, S = 3.0, 8.0

    def Sig(u: float, M: float) -> float:
        return quad(lambda x: float(v(u, x)), u, M, epsabs=0, epsrel=1e-13)[0]

    ref = quad(lambda u: (Sig(u, S) - Sig(u, T)) ** 2, 0, T, epsabs=0, epsrel=1e-12)[0]
    assert hjm.forward_bond_variance(T, T, S) == pytest.approx(ref, rel=1e-11)
    K = float(CURVE.discount(S) / CURVE.discount(T))
    c, p = hjm.zcb_option(T, S, K), hjm.zcb_option(T, S, K, "put")
    assert c - p == pytest.approx(CURVE.discount(S) - K * CURVE.discount(T), abs=1e-15)
    # the hump peaks at T - t = 1/gamma - alpha/beta
    taus = np.linspace(0, 10, 10001)
    assert taus[np.argmax(v(0.0, taus))] == pytest.approx(1 / 0.4 - 0.004 / 0.006, abs=1e-3)


def test_piecewise_hull_white_through_breakpoints() -> None:
    from tests.unit.rates.test_hull_white import HW_PW

    hjm = GaussianHJM.from_hull_white(HW_PW)
    for T, S in ((0.5, 2.0), (2.5, 6.0), (6.0, 16.0)):
        K = float(CURVE.discount(S) / CURVE.discount(T))
        assert hjm.zcb_option(T, S, K) == pytest.approx(HW_PW.zcb_option(T, S, K), rel=1e-11)
    # without the breakpoints the quadrature straddles the jumps and loses accuracy
    blind = GaussianHJM(CURVE, hjm.vols)
    K = float(CURVE.discount(6.0) / CURVE.discount(2.5))
    assert abs(blind.zcb_option(2.5, 6.0, K) / HW_PW.zcb_option(2.5, 6.0, K) - 1) > 1e-6


def test_validation() -> None:
    with pytest.raises(ValueError):
        GaussianHJM(CURVE, [])
    with pytest.raises(ValueError):
        GaussianHJM(CURVE, [exponential_vol(0.01, 0.1)]).zcb_option(2.0, 1.0, 0.9)
