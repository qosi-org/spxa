"""G2++ two-factor Gaussian model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.g2pp import G2PP
from spxa.rates.hull_white import HullWhite
from tests.unit.rates.test_hull_white import CURVE

M = G2PP(CURVE, 0.5, 0.012, 0.05, 0.009, -0.7)


def _expect2(model: G2PP, t: float, T: float, f, kink=None) -> float:  # type: ignore[no-untyped-def]
    """E^T[f(x_t, y_t)] by nested quadrature over standard normals.

    ``kink(x)`` gives the ``y`` at which ``f(x, .)`` has a kink, if any.
    """
    m = model.forward_mean(t, T)
    L = np.linalg.cholesky(model.covariance(t))
    pdf = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)  # noqa: E731

    def inner(z1: float) -> float:
        x = m[0] + L[0, 0] * z1
        g = lambda z2: f(x, m[1] + L[1, 0] * z1 + L[1, 1] * z2) * pdf(z2)  # noqa: E731
        pts = [-9.0, 9.0]
        if kink is not None:
            zk = (kink(x) - m[1] - L[1, 0] * z1) / L[1, 1]
            if -9 < zk < 9:
                pts = [-9.0, zk, 9.0]
        return sum(
            quad(g, lo, hi, epsabs=1e-15, epsrel=1e-12)[0] for lo, hi in zip(pts[:-1], pts[1:])
        )

    return quad(lambda z1: inner(z1) * pdf(z1), -9, 9, epsabs=1e-15, epsrel=1e-12)[0]


def test_fits_the_curve_and_variance_function() -> None:
    T = np.array([0.5, 2.0, 10.0, 30.0])
    assert M.discount_bond(0.0, T, 0.0, 0.0) == pytest.approx(CURVE.discount(T), rel=1e-14)
    for t, T2 in ((0.0, 3.0), (1.0, 7.0)):
        a, b, s, e, r = M.a, M.b, M.sigma, M.eta, M.rho
        Ba = lambda u: (1 - np.exp(-a * (T2 - u))) / a  # noqa: E731
        Bb = lambda u: (1 - np.exp(-b * (T2 - u))) / b  # noqa: E731
        num = quad(
            lambda u: s * s * Ba(u) ** 2 + e * e * Bb(u) ** 2 + 2 * r * s * e * Ba(u) * Bb(u), t, T2
        )[0]
        assert M.V(t, T2) == pytest.approx(num, rel=1e-12)
    # phi is f(0, t) plus half the derivative of V(0, t)
    h = 1e-5
    for t in (0.5, 4.0):
        dV = (M.V(0.0, t + h) - M.V(0.0, t - h)) / (2 * h)
        assert M.phi(t) == pytest.approx(CURVE.instantaneous_forward(t) + 0.5 * dV, abs=1e-9)


def _gauss_hermite2(model: G2PP, t: float, T: float, f, n: int = 60) -> float:  # type: ignore[no-untyped-def]
    """E^T[f(x_t, y_t)] for smooth vectorised ``f`` by a tensor Gauss-Hermite rule."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / np.sqrt(2 * np.pi)
    m = model.forward_mean(t, T)
    L = np.linalg.cholesky(model.covariance(t))
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    x = m[0] + L[0, 0] * z1
    y = m[1] + L[1, 0] * z1 + L[1, 1] * z2
    return float(np.sum(np.outer(w, w) * f(x, y)))


def test_martingale_property_under_forward_measures() -> None:
    for t, T, S in ((1.0, 1.0, 4.0), (2.0, 3.0, 8.0)):
        got = _gauss_hermite2(
            M, t, T, lambda x, y: M.discount_bond(t, S, x, y) / M.discount_bond(t, T, x, y)
        )
        assert got == pytest.approx(CURVE.discount(S) / CURVE.discount(T), rel=1e-13)


def test_bond_options_and_caplets() -> None:
    T, S = 2.0, 7.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    Ba, Bb = (1 - np.exp(-M.a * (S - T))) / M.a, (1 - np.exp(-M.b * (S - T))) / M.b
    c = np.log(M.discount_bond(T, S, 0.0, 0.0) / K)  # P = K on the line Ba x + Bb y = c
    num = CURVE.discount(T) * _expect2(
        M,
        T,
        T,
        lambda x, y: max(M.discount_bond(T, S, x, y) - K, 0.0),
        kink=lambda x: (c - Ba * x) / Bb,
    )
    assert M.zcb_option(T, S, K) == pytest.approx(num, rel=1e-8)
    assert M.zcb_option(T, S, K) - M.zcb_option(T, S, K, "put") == pytest.approx(
        CURVE.discount(S) - K * CURVE.discount(T), abs=1e-15
    )
    # one factor switched (almost) off: Hull-White
    hw = HullWhite(CURVE, 0.5, 0.012)
    g = G2PP(CURVE, 0.5, 0.012, 0.05, 1e-9, 0.0)
    assert g.zcb_option(T, S, K) == pytest.approx(hw.zcb_option(T, S, K), rel=1e-9)
    from spxa.rates.capfloor import cap_floor
    from spxa.rates.indices import EURIBOR6M
    from tests.unit.rates.test_capfloor import PROJ

    cap = cap_floor("2024-03-15", "3Y", 0.035, EURIBOR6M)
    assert g.cap(cap, PROJ) == pytest.approx(hw.cap(cap, PROJ), rel=1e-9)


def test_validation() -> None:
    with pytest.raises(ValueError):
        G2PP(CURVE, 0.1, 0.01, 0.1, 0.01, 0.0)
    with pytest.raises(ValueError):
        G2PP(CURVE, 0.1, 0.01, 0.2, 0.01, 1.0)


class TestSwaptions:
    def test_one_cashflow_is_a_bond_option(self) -> None:
        T, S = 2.0, 7.0
        for K in (0.75, 0.8, 0.85):
            got = M.option_on_cashflows(T, [S], [1.0], const=-K)
            assert got == pytest.approx(M.zcb_option(T, S, K), rel=1e-9)
            put = M.option_on_cashflows(T, [S], [-1.0], const=K)
            assert put == pytest.approx(M.zcb_option(T, S, K, "put"), rel=1e-9)

    def test_parity_and_hull_white_limit(self) -> None:
        from spxa.rates.swaptions import swaption
        from tests.unit.rates.test_capfloor import PROJ

        pay = swaption("2024-03-15", "2Y", "5Y", 0.035, notional=1e6)
        rec = swaption("2024-03-15", "2Y", "5Y", 0.035, notional=1e6, payer=False)
        assert M.swaption(pay, PROJ) - M.swaption(rec, PROJ) == pytest.approx(
            pay.swap.pv(CURVE, PROJ), abs=1e-6
        )
        hw = HullWhite(CURVE, 0.5, 0.012)
        g = G2PP(CURVE, 0.5, 0.012, 0.05, 1e-9, 0.0)
        assert g.swaption(pay, PROJ) == pytest.approx(hw.swaption(pay, PROJ), rel=1e-8)
        # receiver: amounts change sign more than once (basis on the float leg) and the
        # conditional payoff is kinked in the limit; the split integral must still agree
        rec10 = swaption("2024-03-15", "5Y", "10Y", 0.036, notional=1e4, payer=False)
        g2 = G2PP(CURVE, 0.05, 0.01, 0.6, 1e-9, 0.0)
        assert g2.swaption(rec10, PROJ) == pytest.approx(
            HullWhite(CURVE, 0.05, 0.01).swaption(rec10, PROJ), rel=1e-10
        )

    def test_correlation_moves_short_and_long_tenors_differently(self) -> None:
        # higher rho adds variance; negative rho cancels the fast factor in short rates
        # while long rates follow the slow factor, so long/short value ratios rise as rho falls
        from spxa.rates.swaptions import swaption
        from tests.unit.rates.test_capfloor import PROJ

        short = swaption("2024-03-15", "5Y", "1Y", 0.034)
        long = swaption("2024-03-15", "5Y", "20Y", 0.034)
        vals = {rho: G2PP(CURVE, 0.5, 0.012, 0.05, 0.009, rho) for rho in (-0.7, 0.0, 0.7)}
        for sw in (short, long):
            p = [vals[r].swaption(sw, PROJ) for r in (-0.7, 0.0, 0.7)]
            assert p[0] < p[1] < p[2]
        ratio = [vals[r].swaption(long, PROJ) / vals[r].swaption(short, PROJ) for r in (-0.7, 0.7)]
        assert ratio[0] > ratio[1]
