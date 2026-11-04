"""CIR++ closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2

from spxa.pde.reference import cir_bond
from spxa.rates.capfloor import cap_floor
from spxa.rates.cir_pp import CIRPlusPlus
from spxa.rates.indices import EURIBOR6M
from tests.unit.rates.test_capfloor import DISC, PROJ, REF
from tests.unit.rates.test_hull_white import CURVE

M = CIRPlusPlus(CURVE, 0.3, 0.025, 0.08, 0.02)


def test_fits_the_curve_and_the_shift_integrates() -> None:
    for T in (0.5, 2.0, 10.0, 25.0):
        assert M.discount_bond(0.0, T, M.x0) == pytest.approx(CURVE.discount(T), rel=1e-14)
        integral = quad(lambda t: float(M.phi(t)), 0, T, epsabs=0, epsrel=1e-11, limit=200)[0]
        lhs = math.exp(-integral) * cir_bond(M.x0, T, M.kappa, M.theta, M.sigma)
        assert lhs == pytest.approx(CURVE.discount(T), rel=1e-7)


def test_cir_forward_is_minus_log_derivative() -> None:
    h = 1e-5
    for t in (0.3, 3.0, 15.0):
        lp = [math.log(cir_bond(M.x0, s, M.kappa, M.theta, M.sigma)) for s in (t - h, t + h)]
        assert M.cir_forward(t) == pytest.approx(-(lp[1] - lp[0]) / (2 * h), rel=1e-8)


def _mc_option(
    T: float, S: float, K: float, n: int = 200_000, steps: int = 80
) -> tuple[float, float]:
    """Exact non-central chi-square stepping of x with trapezoidal integral."""
    rng = np.random.default_rng(5)
    k, th, s = M.kappa, M.theta, M.sigma
    dt = T / steps
    c = s**2 * -math.expm1(-k * dt) / (4 * k)
    d = 4 * k * th / s**2
    x = np.full(n, M.x0)
    integral = np.zeros(n)
    for _ in range(steps):
        xn = c * ncx2.rvs(d, x * math.exp(-k * dt) / c, random_state=rng)
        integral += 0.5 * (x + xn) * dt
        x = xn
    shift = float(CURVE.discount(T)) / cir_bond(M.x0, T, k, th, s)  # exp(-int phi)
    v = shift * np.exp(-integral) * np.maximum(M.discount_bond(T, S, x) - K, 0.0)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(n))


def test_bond_option_matches_simulation_and_parity() -> None:
    T, S = 2.0, 7.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    c = M.zcb_option(T, S, K)
    p = M.zcb_option(T, S, K, "put")
    assert c - p == pytest.approx(CURVE.discount(S) - K * CURVE.discount(T), abs=1e-14)
    mc, se = _mc_option(T, S, K)
    assert abs(c - mc) < 4 * se + 2e-6


def test_caplets_reduce_to_forward_intrinsic_at_low_vol() -> None:
    m = CIRPlusPlus(DISC, 0.3, 0.025, 1e-5, 0.02)
    cap = cap_floor(REF, "3Y", 0.02, EURIBOR6M, notional=1e4)
    intrinsic = sum(
        c.coupon.notional
        * c.coupon.accrual
        * max(c.forward(PROJ) - c.strike, 0.0)
        * DISC.discount(c.coupon.payment_date)
        for c in cap.caplets
    )
    assert m.cap(cap, PROJ) == pytest.approx(intrinsic, rel=1e-8)
    # caps increase with vol
    assert CIRPlusPlus(DISC, 0.3, 0.025, 0.1, 0.02).cap(cap, PROJ) > m.cap(cap, PROJ)


def test_swaptions_by_jamshidian_match_integration_and_parity() -> None:
    from spxa.rates.swaptions import swaption

    m = CIRPlusPlus(DISC, 0.3, 0.025, 0.08, 0.02)
    for K in (0.03, 0.034, 0.04):
        pay = swaption(REF, "2Y", "5Y", K, notional=1e6)
        rec = swaption(REF, "2Y", "5Y", K, notional=1e6, payer=False)
        vp = m.swaption(pay, PROJ)
        assert vp == pytest.approx(m.swaption(pay, PROJ, method="integral"), rel=1e-8)
        assert vp - m.swaption(rec, PROJ) == pytest.approx(pay.swap.pv(DISC, PROJ), abs=1e-6)


def test_forward_measure_law_prices_bonds() -> None:
    # E^T[P(T, S)] = P(0, S) / P(0, T)
    T, S = 3.0, 8.0
    scale, df, nc = M.forward_measure_law(T)
    val = quad(
        lambda z: float(M.discount_bond(T, S, scale * z)) * float(ncx2.pdf(z, df, nc)),
        0,
        np.inf,
        epsabs=0,
        epsrel=1e-12,
    )[0]
    assert val == pytest.approx(CURVE.discount(S) / CURVE.discount(T), rel=1e-11)


def test_validation() -> None:
    with pytest.raises(ValueError):
        CIRPlusPlus(CURVE, 0.3, 0.025, 0.0, 0.02)
    with pytest.raises(ValueError):
        M.zcb_option(3.0, 2.0, 0.9)
    from spxa.rates.swaptions import swaption

    with pytest.raises(ValueError):
        CIRPlusPlus(DISC, 0.3, 0.025, 0.08, 0.02).swaption(
            swaption(REF, "2Y", "5Y", 0.03), PROJ, method="x"
        )
    assert M.feller and not CIRPlusPlus(CURVE, 0.1, 0.01, 0.2, 0.01).feller
