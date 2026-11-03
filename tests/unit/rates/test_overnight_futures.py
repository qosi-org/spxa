"""SOFR-style overnight futures and their exact Hull–White convexity."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.market import FlatCurve
from spxa.rates import (
    SOFR,
    FixingHistory,
    OvernightFuture,
    hull_white_compounded_convexity,
    hull_white_compounding_covariance,
    hull_white_discrete_compounded_convexity,
)
from spxa.rates.overnight import observation_schedule

D = dt.date
CURVE = FlatCurve(0.04, reference_date="2024-01-15", day_count="ACT/365F")


def B(a: float, t: float) -> float:
    return t if a == 0 else (1 - math.exp(-a * t)) / a


def J(a: float, tau: float) -> float:
    """Conditional variance of int x over tau, divided by sigma^2 (Brigo–Mercurio V)."""
    if a == 0:
        return tau**3 / 3
    return (tau + 2 / a * math.exp(-a * tau) - math.exp(-2 * a * tau) / (2 * a) - 1.5 / a) / a**2


class TestCovariance:
    @pytest.mark.parametrize("a", [0.0, 1e-7, 0.03, 0.4, 2.0])
    @pytest.mark.parametrize(("t", "T"), [(0.5, 0.5), (0.7, 1.9), (3.0, 10.0)])
    def test_against_stochastic_integral(self, a: float, t: float, T: float) -> None:
        # Cov(x(t), int_0^T x) = sigma^2 int_0^t e^{-a(t-u)} B(T-u) du (Ito isometry)
        sigma = 0.013
        ref = sigma**2 * quad(lambda u: math.exp(-a * (t - u)) * B(a, T - u), 0, t)[0]
        got = hull_white_compounding_covariance(t, T, a, sigma)
        assert got == pytest.approx(ref, rel=1e-11)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            hull_white_compounding_covariance(2.0, 1.0, 0.1, 0.01)
        with pytest.raises(ValueError):
            hull_white_compounding_covariance(0.5, 1.0, -0.1, 0.01)


class TestContinuousConvexity:
    @pytest.mark.parametrize("a", [0.0, 1e-6, 0.05, 0.39, 0.41, 1.5])
    @pytest.mark.parametrize(("ts", "te"), [(0.0, 0.25), (1.0, 1.25), (4.75, 5.0), (2.0, 3.0)])
    def test_is_integral_of_covariance(self, a: float, ts: float, te: float) -> None:
        # Gamma_c = int_{Ts}^{Te} c(t; Te) dt = Var(I) + Cov(int_0^Ts r, I)
        sigma = 0.01
        ref = quad(lambda t: hull_white_compounding_covariance(t, te, a, sigma), ts, te)[0]
        got = hull_white_compounded_convexity(ts, te, a, sigma)
        assert got == pytest.approx(ref, rel=1e-11)

    def test_variance_plus_covariance_closed_form(self) -> None:
        # independent route: Var(I) + Cov(A, I) from the Ito integrands
        a, sigma, ts, te = 0.2, 0.012, 1.5, 2.0
        bse = B(a, te - ts)
        var_i = sigma**2 * (bse**2 * (1 - math.exp(-2 * a * ts)) / (2 * a) + J(a, te - ts))
        cov = sigma**2 * bse * quad(lambda u: B(a, ts - u) * math.exp(-a * (ts - u)), 0, ts)[0]
        assert hull_white_compounded_convexity(ts, te, a, sigma) == pytest.approx(
            var_i + cov, rel=1e-12
        )

    def test_ho_lee_closed_form(self) -> None:
        sigma, ts, tau = 0.01, 1.0, 0.25
        ref = sigma**2 * (tau**2 * ts + tau**3 / 3 + tau * ts**2 / 2)
        assert hull_white_compounded_convexity(ts, ts + tau, 0.0, sigma) == pytest.approx(
            ref, rel=1e-15
        )
        # a spot-starting contract still carries sigma^2 tau^3 / 3
        assert hull_white_compounded_convexity(0.0, tau, 0.0, sigma) == pytest.approx(
            sigma**2 * tau**3 / 3, rel=1e-15
        )

    def test_continuous_in_mean_reversion(self) -> None:
        # the series branch (a tau < 0.1) and the closed form agree at the switch
        lo = hull_white_compounded_convexity(1.0, 1.25, 0.4 - 1e-12, 0.01)
        hi = hull_white_compounded_convexity(1.0, 1.25, 0.4 + 1e-12, 0.01)
        assert lo == pytest.approx(hi, rel=1e-13)
        assert hull_white_compounded_convexity(1.0, 1.25, 1e-12, 0.01) == pytest.approx(
            hull_white_compounded_convexity(1.0, 1.25, 0.0, 0.01), rel=1e-11
        )

    def test_discrete_converges_to_continuous(self) -> None:
        a, sigma, ts, te = 0.1, 0.01, 1.0, 1.25
        ref = hull_white_compounded_convexity(ts, te, a, sigma)
        errs = []
        for n in (500, 1000, 2000):
            g = hull_white_discrete_compounded_convexity(np.linspace(ts, te, n + 1), a, sigma)
            errs.append(abs(g - ref))
        assert errs[-1] < 2e-4 * ref
        assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.02)  # first order in the step
        assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.02)

    def test_fixed_subperiods_do_not_contribute(self) -> None:
        t = [-0.02, -0.01, 0.0, 0.01, 0.02]
        g = hull_white_discrete_compounded_convexity(t, 0.1, 0.01)
        ref = sum(
            B(0.1, 0.01) * hull_white_compounding_covariance(x, 0.02, 0.1, 0.01) for x in (0.01,)
        )
        assert g == pytest.approx(ref, rel=1e-15)
        with pytest.raises(ValueError):
            hull_white_discrete_compounded_convexity([0.1, 0.1], 0.1, 0.01)


def _model_times(fut: OvernightFuture, curve: FlatCurve) -> np.ndarray:
    acc, _, _, _ = observation_schedule(fut.index, fut.start, fut.end)
    return np.array([float(curve.time(d)) for d in [*acc, fut.end]])


def _g1pp_inverse_bonds(t: np.ndarray, a: float, sigma: float, n: int, seed: int) -> np.ndarray:
    """Exact simulation of 1/P(t_i, t_{i+1}) in G1++ fitted to CURVE (antithetic pairs).

    P(t, T) = P(0,T)/P(0,t) exp(1/2 [V(t,T) - V(0,T) + V(0,t)] - B(T-t) x(t)),
    V(t, T) = sigma^2 J(T - t)  (Brigo & Mercurio 2006, eq. 3.39).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, t.size - 1))
    z = np.vstack([z, -z])
    x = np.zeros(2 * n)
    prev = 0.0
    out = np.empty((2 * n, t.size - 1))
    for i in range(t.size - 1):
        dt_ = t[i] - prev
        e = math.exp(-a * dt_)
        x = x * e + sigma * math.sqrt((1 - e * e) / (2 * a)) * z[:, i]
        prev = t[i]
        t0, t1 = t[i], t[i + 1]
        p0 = math.exp(-0.04 * t1) / math.exp(-0.04 * t0)
        v = 0.5 * sigma**2 * (J(a, t1 - t0) - J(a, t1) + J(a, t0))
        out[:, i] = np.exp(-math.log(p0) - v + B(a, t1 - t0) * x)
    return out


def _pair_se(v: np.ndarray) -> float:
    """Standard error of the mean of antithetic pairs (first half vs mirrored second half)."""
    n = v.size // 2
    pairs = 0.5 * (v[:n] + v[n:])
    return float(pairs.std() / math.sqrt(n))


class TestHullWhiteMonteCarlo:
    a, sigma = 0.08, 0.015

    def test_compounded_quarter(self) -> None:
        fut = OvernightFuture.from_imm("2025-03-19", 96.0)
        t = _model_times(fut, CURVE)
        inv = _g1pp_inverse_bonds(t, self.a, self.sigma, 100_000, seed=11)
        growth = inv.prod(axis=1)  # 1 + tau R on each path
        fwd = math.exp(-0.04 * t[0]) / math.exp(-0.04 * t[-1])
        mc_gamma = math.log(growth.mean() / fwd)
        se = _pair_se(growth) / growth.mean()
        gamma = hull_white_discrete_compounded_convexity(t, self.a, self.sigma)
        assert gamma > 50 * se  # the adjustment is resolved by the simulation
        assert abs(mc_gamma - gamma) < 4 * se
        # and the contract applies it on top of the projected forward
        F = fut.forward_rate(CURVE)
        assert 1 + fut.accrual * F == pytest.approx(fwd, rel=1e-13)
        adj = fut.convexity_adjustment(CURVE, self.a, self.sigma)
        assert adj == pytest.approx(fwd * math.expm1(gamma) / fut.accrual, rel=1e-13)

    def test_averaged_month(self) -> None:
        # July 2025 starts on a business day, so every sub-period runs from a fixing
        # date to the next one (Friday 4 July is a holiday: 3 July covers four days)
        fut = OvernightFuture.monthly(2025, 7, 96.0)
        assert (fut.start, fut.end, fut.method) == (D(2025, 7, 1), D(2025, 8, 1), "averaged")
        _, obs, w, _ = observation_schedule(SOFR, fut.start, fut.end)
        assert obs[0] == D(2025, 7, 1) and w[obs.index(D(2025, 7, 3))] == 4 / 360
        t = _model_times(fut, CURVE)
        inv = _g1pp_inverse_bonds(t, self.a, self.sigma, 100_000, seed=12)
        avg = (inv - 1.0).sum(axis=1) / fut.accrual  # sum r_i tau_i / tau
        se = _pair_se(avg)
        model = fut.implied_rate(CURVE, self.a, self.sigma)
        assert fut.convexity_adjustment(CURVE, self.a, self.sigma) > 20 * se
        assert abs(avg.mean() - model) < 4 * se


class TestContract:
    def test_imm_quarter_and_validation(self) -> None:
        fut = OvernightFuture.from_imm("2024-03-20", 94.75)
        assert (fut.start, fut.end) == (D(2024, 3, 20), D(2024, 6, 19))
        assert fut.rate == pytest.approx(0.0525, abs=1e-15)
        assert fut.accrual == pytest.approx(91 / 360, abs=1e-15)
        assert OvernightFuture.monthly(2024, 12, 95.0).end == D(2025, 1, 1)
        with pytest.raises(ValueError, match="IMM"):
            OvernightFuture.from_imm("2024-03-21", 95.0)
        with pytest.raises(ValueError):
            OvernightFuture(D(2024, 3, 20), D(2024, 6, 19), 95.0, method="median")
        with pytest.raises(ValueError):
            OvernightFuture(D(2024, 6, 19), D(2024, 3, 20), 95.0)

    def test_no_volatility_means_forward(self) -> None:
        fut = OvernightFuture.from_imm("2024-06-19", 95.0)
        assert fut.convexity_adjustment(CURVE) == 0.0
        F = fut.forward_rate(CURVE)
        assert fut.implied_price(CURVE) == pytest.approx(100 * (1 - F), abs=1e-12)
        fixed = OvernightFuture.from_imm("2024-06-19", 95.0, convexity=2e-4)
        assert fixed.implied_rate(CURVE, 0.1, 0.01) == pytest.approx(F + 2e-4, abs=1e-16)

    def test_ho_lee_quarter_close_to_continuous(self) -> None:
        # daily compounding over a quarter differs from the continuous limit by O(1 day)
        fut = OvernightFuture.from_imm("2025-03-19", 96.0)
        t = _model_times(fut, CURVE)
        g_d = hull_white_discrete_compounded_convexity(t, 0.0, 0.01)
        g_c = hull_white_compounded_convexity(t[0], t[-1], 0.0, 0.01)
        assert g_d == pytest.approx(g_c, rel=0.02)

    def test_seasoned_contract_uses_fixings(self) -> None:
        # valued mid-quarter: past days carry their fixings and no convexity
        fut = OvernightFuture.from_imm("2024-03-20", 94.7)
        curve = FlatCurve(0.05, reference_date="2024-05-01", day_count="ACT/365F")
        acc, obs, w, _ = observation_schedule(SOFR, fut.start, fut.end)
        past = [o for o in obs if o < D(2024, 5, 1)]
        fx = FixingHistory.from_pairs("SOFR", [(o, 0.0531) for o in past])
        F = fut.forward_rate(curve, fx)
        growth = np.prod([1 + 0.0531 * wi for o, wi in zip(obs, w) if o < D(2024, 5, 1)])
        growth *= float(curve.discount(D(2024, 5, 1))) / float(curve.discount(fut.end))
        assert 1 + fut.accrual * F == pytest.approx(growth, rel=1e-14)
        times = [float(curve.time(d)) for d in [*acc, fut.end]]
        future = [i for i, o in enumerate(obs) if o >= D(2024, 5, 1)]
        g = sum(
            B(0.05, times[i + 1] - times[i])
            * hull_white_compounding_covariance(times[i], times[-1], 0.05, 0.01)
            for i in future
            if times[i] > 0
        )
        adj = fut.convexity_adjustment(curve, 0.05, 0.01, fx)
        assert adj == pytest.approx(growth * math.expm1(g) / fut.accrual, rel=1e-13)
        assert (
            0
            < adj
            < OvernightFuture.from_imm("2024-06-19", 95.0).convexity_adjustment(curve, 0.05, 0.01)
        )
