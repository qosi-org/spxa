"""Variance, gamma and volatility swaps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

import spxa.vol as v
from spxa.market import EquityForward
from spxa.vol.variance_swaps import (
    ForwardVarianceCurve,
    fair_variance,
    gamma_swap_strike,
    heston_integrated_variance_laplace,
    heston_volatility_swap,
    realised_variance,
    variance_swap_mtm,
    variance_swap_payoff,
    variance_swap_curve,
    variance_swap_strike,
    vix_index,
)


class Flat:
    def __init__(self, vol: float) -> None:
        self.vol = vol

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.full(np.shape(k), self.vol**2 * T)


SSVI_SURFACE = v.VolSurface(
    v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, 0.03, dividend_yield=0.01),
)


def test_flat_smile_gives_sigma_squared() -> None:
    for vol, T in [(0.2, 1.0), (0.05, 0.1), (0.6, 5.0)]:
        assert variance_swap_strike(Flat(vol), T) == pytest.approx(vol * vol, rel=1e-12)
        assert gamma_swap_strike(Flat(vol), T) == pytest.approx(vol * vol, rel=1e-12)


def test_replication_recovers_model_expected_variance() -> None:
    # Heston through the H = 1/2 rough Heston pricer (vectorised over strikes)
    h = v.RoughHeston(0.04, 0.06, 1.5, 0.5, -0.7, 0.5)

    def otm(k: np.ndarray) -> np.ndarray:
        return np.where(
            k >= 1, h.price(k, 1.0, n_steps=400), h.price(k, 1.0, option="put", n_steps=400)
        )

    expected = v.Heston(0.04, 1.5, 0.06, 0.5, -0.7).expected_total_variance(1.0)
    assert fair_variance(otm, 1.0) == pytest.approx(expected, abs=1e-6)
    r = v.RoughHeston(0.02, 0.05, 0.3, 0.3, -0.7, 0.1)
    for T in (0.25, 1.0):
        rough = lambda k, T=T: np.where(k >= 1, r.price(k, T), r.price(k, T, option="put"))  # noqa: E731
        ev = quad(lambda t: float(r.expected_variance(t)), 0, T, epsabs=1e-13)[0] / T
        assert fair_variance(rough, T) == pytest.approx(ev, abs=2e-5)


def test_skew_orders_variance_gamma_and_atm() -> None:
    # negative skew: the 1/k^2 weight loads the expensive puts, 1/k less so
    kvar, kgam = variance_swap_strike(SSVI_SURFACE, 1.0), gamma_swap_strike(SSVI_SURFACE, 1.0)
    atm = float(SSVI_SURFACE.total_variance(np.zeros(1), 1.0)[0])
    assert kvar > kgam > atm
    assert (round(kvar, 4), round(kgam, 4), round(atm, 4)) == (0.0569, 0.0494, 0.043)


def test_mark_to_market() -> None:
    # halfway, realised 0.05 vs remaining fair 0.03: expected 0.04 = strike, so worth zero
    assert variance_swap_mtm(0.05, 0.5, 1.0, 0.04, 0.03, 0.98) == pytest.approx(0.0, abs=1e-17)
    assert variance_swap_mtm(0.06, 0.25, 1.0, 0.04, 0.04, 0.98, notional=100.0) == pytest.approx(
        100.0 * 0.98 * 0.005, rel=1e-14
    )
    with pytest.raises(ValueError):
        variance_swap_mtm(0.05, 1.0, 1.0, 0.04, 0.03, 0.98)
    with pytest.raises(ValueError):
        fair_variance(lambda k: k, 0.0)


def test_heston_volatility_swap() -> None:
    h = v.Heston(0.04, 1.5, 0.06, 0.5, -0.7)
    ev = h.expected_total_variance(1.0)
    d = 1e-6
    assert (1 - heston_integrated_variance_laplace(h, 1.0, d)) / d == pytest.approx(ev, rel=1e-6)
    vs = heston_volatility_swap(h, 1.0)
    assert vs < math.sqrt(
        ev
    )  # Jensen: the vol swap sits below the square root of the variance swap
    # Monte Carlo: full-truncation Euler for v and trapezoidal integrated variance
    rng = np.random.default_rng(0)
    n, steps = 100_000, 400
    dt = 1.0 / steps
    var, integral = np.full(n, h.v0), np.zeros(n)
    for _ in range(steps):
        vp = np.maximum(var, 0.0)
        var = var + h.kappa * (h.theta - vp) * dt + h.xi * np.sqrt(vp * dt) * rng.standard_normal(n)
        integral += 0.5 * (vp + np.maximum(var, 0.0)) * dt
    mc = np.sqrt(integral)
    assert abs(vs - mc.mean()) < 3 * mc.std() / math.sqrt(n)
    # vanishing vol of variance: deterministic variance, no convexity
    for T in (0.1, 1.0, 5.0):
        m = v.Heston(0.04, 1.5, 0.06, 1e-4, -0.7)
        assert heston_volatility_swap(m, T) == pytest.approx(
            math.sqrt(m.expected_total_variance(T) / T), abs=1e-8
        )
    assert round(vs, 4) == 0.2090 and round(math.sqrt(ev), 4) == 0.2228


def test_term_structure_and_forward_variance_models() -> None:
    window = 30.0 / 365.0
    expiries = np.array([window, 0.25, 0.5, 1.0, 2.0])
    strikes = variance_swap_curve(SSVI_SURFACE, expiries)
    curve = ForwardVarianceCurve(expiries, strikes)
    assert np.all(np.diff(strikes * expiries) > 0) and np.all(curve.levels > 0)
    np.testing.assert_allclose(curve.variance_swap(expiries), strikes, rtol=1e-14)
    assert curve.forward_variance_swap(1.0, 2.0) == pytest.approx(curve.levels[-1], rel=1e-14)
    assert curve.forward_variance_swap(0.0, 0.5) == pytest.approx(strikes[2], rel=1e-14)
    assert vix_index(SSVI_SURFACE) == pytest.approx(100 * math.sqrt(strikes[0]), rel=1e-14)
    assert round(vix_index(SSVI_SURFACE), 2) == 21.61
    # the curve fed to rough Bergomi: E[VIX_0^2] is the 30-day strike and E[int v] the 1y strike
    assert v.RBergomiVIX(0.1, 1.9, curve).expected_vix_squared(0.0) == pytest.approx(
        strikes[0], rel=1e-14
    )
    paths = v.rbergomi_paths(0.1, 1.9, -0.9, curve, 1.0, 438, 20_000, seed=1)
    se = paths.int_v.std() / math.sqrt(paths.int_v.size)
    assert abs(paths.int_v.mean() - strikes[3]) < 3 * se
    with pytest.raises(ValueError):
        ForwardVarianceCurve([1.0, 2.0], [0.04, 0.01])  # total variance falls
    with pytest.raises(ValueError):
        ForwardVarianceCurve([2.0, 1.0], [0.04, 0.04])


def test_realised_variance_and_discrete_monitoring() -> None:
    # exact GBM: E[sum r_i^2]/T = sigma^2 + sigma^4 dt/4, the discrete-monitoring bias
    sigma, steps, n = 0.4, 52, 200_000
    dt = 1.0 / steps
    rng = np.random.default_rng(0)
    r = -0.5 * sigma**2 * dt + sigma * math.sqrt(dt) * rng.standard_normal((n, steps))
    x, t = np.c_[np.zeros(n), np.cumsum(r, axis=1)], np.linspace(0.0, 1.0, steps + 1)
    rv, rvd = realised_variance(x, t), realised_variance(x, t, demean=True)
    assert abs(rv.mean() - (sigma**2 + sigma**4 * dt / 4)) < 3 * rv.std() / math.sqrt(n)
    assert abs(rvd.mean() - sigma**2) < 3 * rvd.std() / math.sqrt(n)  # demeaned: unbiased
    # daily Heston paths: realised variance prices the variance swap at E[V]/T
    h = v.Heston(0.04, 1.5, 0.06, 0.5, -0.7)
    days = np.linspace(0.0, 1.0, 253)
    paths = v.slv_paths(
        h, lambda y, s: np.ones_like(y), 1.0, 252, 20_000, record=tuple(days[1:]), seed=5
    )
    logs = np.column_stack([np.zeros(20_000)] + [paths.at(d)[0] for d in days[1:]])
    hv = realised_variance(logs, days)
    kvar = h.expected_total_variance(1.0)
    assert abs(hv.mean() - kvar) < 3 * hv.std() / math.sqrt(hv.size)
    capped, plain = variance_swap_payoff(hv, kvar, cap=2.5), variance_swap_payoff(hv, kvar)
    assert np.all(capped <= plain) and np.any(capped < plain)
    assert np.all(capped <= (2.5**2 - 1) * kvar)
    with pytest.raises(ValueError):
        realised_variance(x[:, :3], t[:4])
