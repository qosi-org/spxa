"""Hull-White one-factor model on a discount curve."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.market.curves import YieldCurve
from spxa.rates.hull_white import HullWhite


class SmoothCurve(YieldCurve):
    """An upward-sloping curve with an analytic discount function."""

    def __init__(self) -> None:
        super().__init__(reference_date="2024-03-15")

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-(0.025 * t + 0.002 * t * t / (1 + 0.1 * t)))


CURVE = SmoothCurve()
HW = HullWhite(CURVE, 0.05, 0.01)
HW_PW = HullWhite(CURVE, 0.08, [0.012, 0.009, 0.007], [1.0, 3.0])


def _expect(model: HullWhite, t: float, T: float, f) -> float:  # type: ignore[no-untyped-def]
    """E^T[f(x_t)] by quadrature."""
    m, s = model.forward_mean(t, T), np.sqrt(model.y(t))
    g = lambda z: f(m + s * z) * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)  # noqa: E731
    return quad(g, -12, 12, epsabs=1e-15, epsrel=1e-13, limit=200)[0]


def test_fits_the_curve_and_constant_vol_variance() -> None:
    T = np.array([0.5, 1.0, 5.0, 20.0])
    assert HW.discount_bond(0.0, T, 0.0) == pytest.approx(CURVE.discount(T), rel=1e-15)
    a, s = 0.05, 0.01
    for t in (0.5, 2.0, 10.0):
        assert HW.y(t) == pytest.approx(s * s * (1 - np.exp(-2 * a * t)) / (2 * a), rel=1e-13)
    # r_t = f(0, t) + x_t
    assert HW.short_rate(1.0, 0.002) == pytest.approx(
        0.025 + 0.004 * (1 + 0.05) / 1.21 + 0.002, abs=1e-9
    )


@pytest.mark.parametrize("model", [HW, HW_PW])
def test_piecewise_variance_and_martingale_property(model: HullWhite) -> None:
    for t in (0.7, 2.5, 6.0):
        num = quad(
            lambda u: (
                np.exp(-2 * model.a * (t - u))
                * model.sigma[np.searchsorted(model.sigma_times, u, "right")] ** 2
            ),
            0,
            t,
            points=list(model.sigma_times[model.sigma_times < t]),
        )[0]
        assert model.y(t) == pytest.approx(num, rel=1e-12)
        # P(t, S)/P(t, T) is a martingale under the T-forward measure
        for T, S in ((t, t + 1.0), (t + 0.5, t + 4.0)):
            got = _expect(
                model, t, T, lambda x: model.discount_bond(t, S, x) / model.discount_bond(t, T, x)
            )
            assert got == pytest.approx(CURVE.discount(S) / CURVE.discount(T), rel=1e-12)


def test_short_rate_shift_reproduces_the_forward_curve() -> None:
    # E^t[r_t] = f(0, t) under the t-forward measure
    for t in (0.5, 3.0, 8.0):
        mean_r = HW_PW.phi(t) + HW_PW.forward_mean(t, t)
        assert mean_r == pytest.approx(CURVE.instantaneous_forward(t), abs=1e-14)


@pytest.mark.parametrize("model", [HW, HW_PW])
def test_bond_options_match_integration(model: HullWhite) -> None:
    T, S = 2.0, 7.0
    fwd = CURVE.discount(S) / CURVE.discount(T)
    for K in (0.9 * fwd, fwd, 1.1 * fwd):
        pT = CURVE.discount(T)
        call = pT * _expect(model, T, T, lambda x: max(model.discount_bond(T, S, x) - K, 0.0))
        assert model.zcb_option(T, S, K) == pytest.approx(call, rel=1e-9)
        parity = model.zcb_option(T, S, K) - model.zcb_option(T, S, K, "put")
        assert parity == pytest.approx(CURVE.discount(S) - K * pT, abs=1e-15)


def test_validation() -> None:
    with pytest.raises(ValueError):
        HullWhite(CURVE, 0.0, 0.01)
    with pytest.raises(ValueError):
        HullWhite(CURVE, 0.05, [0.01, 0.02])
    with pytest.raises(ValueError):
        HW.zcb_option(2.0, 1.0, 0.9)
    assert HW_PW.with_sigma([0.02, 0.02, 0.02]).y(2.0) == pytest.approx(
        HullWhite(CURVE, 0.08, 0.02).y(2.0), rel=1e-14
    )


class TestCapsAndSwaptions:
    def test_caplets_match_integration(self) -> None:
        from spxa.rates.capfloor import cap_floor
        from spxa.rates.indices import EURIBOR6M
        from tests.unit.rates.test_capfloor import PROJ

        model = HullWhite(CURVE, 0.05, 0.012)
        cap = cap_floor("2024-03-15", "5Y", 0.04, EURIBOR6M, notional=1e6)
        floor = cap_floor("2024-03-15", "5Y", 0.04, EURIBOR6M, notional=1e6, kind="floor")
        for cl in cap.caplets[::3]:
            c = cl.coupon
            tf, s, e = (float(CURVE.time(d)) for d in (cl.coupon.fixing, c.start, c.end))
            Q0 = CURVE.discount(s) / CURVE.discount(e)
            Kstar = 1 + c.accrual * (cl.strike - (cl.forward(PROJ) - (Q0 - 1) / c.accrual))

            def payoff(x: float, tf=tf, s=s, e=e, Kstar=Kstar) -> float:  # type: ignore[no-untyped-def]
                Q = model.discount_bond(tf, s, x) / model.discount_bond(tf, e, x)
                return max(Q - Kstar, 0.0)

            num = _expect(model, tf, e, payoff) * c.notional * CURVE.discount(c.payment_date)
            assert model.caplet(cl, PROJ) == pytest.approx(num, rel=1e-8)
        # cap - floor = sum of forward-starting FRAs
        fras = sum(c.weight(CURVE) * (c.forward(PROJ) - 0.04) for c in cap.caplets)
        assert model.cap(cap, PROJ) - model.cap(floor, PROJ) == pytest.approx(fras, rel=1e-12)

    def test_jamshidian_matches_integration_and_parity(self) -> None:
        from spxa.rates.swaptions import swaption
        from tests.unit.rates.test_capfloor import PROJ

        for model in (HW, HW_PW):
            for K in (0.03, 0.035, 0.045):
                pay = swaption("2024-03-15", "2Y", "5Y", K, notional=1e6)
                rec = swaption("2024-03-15", "2Y", "5Y", K, notional=1e6, payer=False)
                j = model.swaption(pay, PROJ)
                assert j == pytest.approx(model.swaption(pay, PROJ, "integral"), rel=1e-9)
                r = model.swaption(rec, PROJ)
                assert r == pytest.approx(model.swaption(rec, PROJ, "integral"), rel=1e-9)
                assert j - r == pytest.approx(pay.swap.pv(CURVE, PROJ), abs=1e-7)

    def test_small_vol_limit_is_intrinsic(self) -> None:
        from spxa.rates.swaptions import swaption
        from tests.unit.rates.test_capfloor import PROJ

        tiny = HullWhite(CURVE, 0.05, 1e-7)
        pay = swaption("2024-03-15", "1Y", "5Y", 0.03, notional=1e6)
        assert tiny.swaption(pay, PROJ) == pytest.approx(
            max(pay.swap.pv(CURVE, PROJ), 0.0), rel=1e-5
        )
        with pytest.raises(ValueError):
            HW.swaption(pay, PROJ, method="tree")
