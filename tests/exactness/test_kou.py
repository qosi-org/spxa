r"""
Mathematical exactness and consistency tests for the Kou (2002) Double Exponential Jump-Diffusion process.

Verifies:
1. Exactness level (EXACT) and canonical aliases (KouProcess, Kou72, Kou02).
2. Exact Lévy triplet (b, σ², ν) with analytical truncated double-exponential drift integral.
3. Property lattice invariants (stationary & independent increments, martingale condition).
4. Characteristic exponent and time additivity.
5. Analytical cumulants of all orders matching raw moments of asymmetric double exponential jumps.
6. Algebraic composition with Brownian motion under triplet arithmetic.
7. Kou (2002) option pricing via Fourier inversion and exact Put-Call parity.
8. Black–Scholes limit when jump intensity λ -> 0.
9. Implied volatility smile and asymmetric jump skew.
10. Exact Monte Carlo simulation convergence with zero discretization error.

References
----------
Kou, S. G. (2002). A jump-diffusion model for option pricing.
    *Management Science*, 48(8), 1086–1101.
Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Chapman & Hall/CRC.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

import spxa
from spxa import BrownianMotion, Kou02, Kou72, KouJumpDiffusion, KouProcess
from spxa.core.exactness import ExactnessLevel


class TestKouExactness:
    """Exactness test suite for KouJumpDiffusion."""

    def test_kou_exactness_level_and_aliases(self) -> None:
        """Kou process is EXACT and all aliases point to the same class."""
        kou = KouJumpDiffusion(drift=0.05, sigma=0.15, rate=1.2, p=0.4, eta1=10.0, eta2=5.0)
        assert kou.exactness == ExactnessLevel.EXACT
        assert KouProcess is KouJumpDiffusion
        assert Kou72 is KouJumpDiffusion
        assert Kou02 is KouJumpDiffusion
        assert "KouJumpDiffusion" in repr(kou)
        assert "sigma=0.15" in repr(kou)
        assert "p=0.4" in repr(kou)

    def test_kou_invalid_parameters_raise(self) -> None:
        """Invalid model parameters raise ValueError."""
        # sigma <= 0
        with pytest.raises(ValueError, match="sigma"):
            KouJumpDiffusion(sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            KouJumpDiffusion(sigma=-0.2)

        # rate <= 0
        with pytest.raises(ValueError, match="rate"):
            KouJumpDiffusion(rate=0.0)
        with pytest.raises(ValueError, match="rate"):
            KouJumpDiffusion(rate=-1.0)

        # p not in [0, 1]
        with pytest.raises(ValueError, match="p"):
            KouJumpDiffusion(p=-0.1)
        with pytest.raises(ValueError, match="p"):
            KouJumpDiffusion(p=1.1)

        # eta1 <= 1.0 (required for finite expectation E[e^Y] < ∞)
        with pytest.raises(ValueError, match="eta1"):
            KouJumpDiffusion(eta1=1.0)
        with pytest.raises(ValueError, match="eta1"):
            KouJumpDiffusion(eta1=0.5)

        # eta2 <= 0
        with pytest.raises(ValueError, match="eta2"):
            KouJumpDiffusion(eta2=0.0)
        with pytest.raises(ValueError, match="eta2"):
            KouJumpDiffusion(eta2=-2.0)

        # Unexpected kwargs
        with pytest.raises(TypeError, match="Unexpected keyword"):
            KouJumpDiffusion(invalid_arg=123)

        # Pricing invalid inputs
        kou = KouJumpDiffusion(drift=0.05, sigma=0.15, rate=1.0, p=0.4, eta1=10.0, eta2=5.0)
        with pytest.raises(ValueError, match="Maturity"):
            kou.call_price(100.0, T=-1.0)
        with pytest.raises(ValueError, match="S0"):
            kou.call_price(100.0, S0=-10.0)
        with pytest.raises(ValueError, match="strikes"):
            kou.call_price(strikes=[-5.0, 100.0])

    def test_kou_triplet_exact(self) -> None:
        """Analytical truncated drift matches numerical integration to machine precision."""
        drift = 0.03
        sigma = 0.18
        rate = 1.5
        p = 0.35
        eta1 = 12.0
        eta2 = 6.0

        kou = KouJumpDiffusion(drift=drift, sigma=sigma, rate=rate, p=p, eta1=eta1, eta2=eta2)
        trip = kou.triplet

        assert trip.sigma_sq == pytest.approx(sigma**2)
        assert trip.nu.total_mass() == pytest.approx(rate)
        assert trip.nu.is_finite_activity() is True

        # Truncated drift integral verification: ∫_{-1}^1 y f_Y(y) dy
        def fy(y: float) -> float:
            if y >= 0:
                return p * eta1 * math.exp(-eta1 * y)
            else:
                return (1.0 - p) * eta2 * math.exp(eta2 * y)

        num_int, _ = quad(lambda y: y * fy(y), -1.0, 1.0)
        expected_b = drift + rate * num_int
        assert trip.b == pytest.approx(expected_b, abs=1e-12)

    def test_kou_properties(self) -> None:
        """Property lattice invariants for Kou process."""
        p, eta1, eta2 = 0.4, 10.0, 5.0
        ey = p / eta1 - (1.0 - p) / eta2
        rate = 1.5

        # Drift set to offset jump drift for martingale
        kou_mart = KouJumpDiffusion(
            drift=-rate * ey, sigma=0.2, rate=rate, p=p, eta1=eta1, eta2=eta2
        )
        props_m = kou_mart.properties
        assert props_m.has_stationary_increments is True
        assert props_m.has_independent_increments is True
        assert props_m.is_martingale is True
        assert props_m.has_finite_variance is True
        assert props_m.has_finite_mean is True
        assert props_m.is_subordinator is False
        assert any("Kou" in note for note in props_m.notes)

        # Non-martingale
        kou_drift = KouJumpDiffusion(drift=0.05, sigma=0.2, rate=rate, p=p, eta1=eta1, eta2=eta2)
        assert kou_drift.properties.is_martingale is False

    def test_kou_characteristic_exponent_and_additivity(self) -> None:
        """Characteristic function satisfies probability invariants and time additivity."""
        kou = KouJumpDiffusion(drift=0.02, sigma=0.15, rate=1.0, p=0.45, eta1=9.0, eta2=4.0)

        # phi(0) = 1
        assert kou.characteristic_function(0.0) == pytest.approx(1.0 + 0j)

        # |phi(u)| <= 1
        for u in [0.5, 1.0, 5.0, 10.0]:
            cf = kou.characteristic_function(u, t=1.0)
            assert abs(cf) <= 1.0 + 1e-12

        # Time additivity: phi(u, t1 + t2) = phi(u, t1) * phi(u, t2)
        t1, t2 = 0.35, 0.65
        for u in [1.5, -2.0, 3.5]:
            cf_sum = kou.characteristic_function(u, t=t1 + t2)
            cf_prod = kou.characteristic_function(u, t=t1) * kou.characteristic_function(u, t=t2)
            assert cf_sum.real == pytest.approx(cf_prod.real, rel=1e-10)
            assert cf_sum.imag == pytest.approx(cf_prod.imag, rel=1e-10)

    def test_kou_cumulants_analytical(self) -> None:
        """Analytical cumulants match theoretical raw moments and time scaling."""
        drift = 0.04
        sigma = 0.16
        rate = 1.8
        p = 0.4
        q = 0.6
        eta1 = 11.0
        eta2 = 5.5

        kou = KouJumpDiffusion(drift=drift, sigma=sigma, rate=rate, p=p, eta1=eta1, eta2=eta2)

        # Moments of jump distribution Y
        ey = p / eta1 - q / eta2
        ey2 = 2.0 * p / (eta1**2) + 2.0 * q / (eta2**2)
        ey3 = 6.0 * (p / (eta1**3) - q / (eta2**3))
        ey4 = 24.0 * (p / (eta1**4) + q / (eta2**4))

        k1 = kou.cumulants(order=4, t=1.0)
        assert k1[1] == pytest.approx(drift + rate * ey)
        assert k1[2] == pytest.approx(sigma**2 + rate * ey2)
        assert k1[3] == pytest.approx(rate * ey3)
        assert k1[4] == pytest.approx(rate * ey4)

        # Linear scaling with time t
        t_val = 2.5
        kt = kou.cumulants(order=4, t=t_val)
        for n in range(1, 5):
            assert kt[n] == pytest.approx(t_val * k1[n], rel=1e-10)

    def test_kou_composition_with_brownian_motion(self) -> None:
        """Algebraic composition Kou + c * BM preserves EXACT exactness and triplet rules."""
        kou = KouJumpDiffusion(drift=0.03, sigma=0.15, rate=1.2, p=0.4, eta1=10.0, eta2=5.0)
        bm = BrownianMotion(mu=0.01, sigma=0.10)

        # Addition
        composed = kou + 2.0 * bm
        assert composed.exactness == ExactnessLevel.EXACT
        assert composed.triplet.sigma_sq == pytest.approx(
            kou.sigma**2 + 4.0 * bm.sigma**2, rel=1e-10
        )
        assert composed.triplet.nu.total_mass() == pytest.approx(kou.rate)

    def test_kou_option_pricing_and_put_call_parity(self) -> None:
        """Kou Fourier option pricer satisfies exact Put-Call parity across strikes."""
        S0 = 100.0
        r = 0.05
        q = 0.015
        T = 0.75

        kou = KouJumpDiffusion(drift=0.0, sigma=0.15, rate=1.5, p=0.35, eta1=12.0, eta2=5.0)
        strikes = [75.0, 90.0, 100.0, 110.0, 125.0]

        calls = kou.call_price(strikes, T=T, S0=S0, r=r, q=q)
        puts = kou.put_price(strikes, T=T, S0=S0, r=r, q=q)

        # Monotonicity: call prices strictly decrease with strike
        assert np.all(np.diff(calls) < 0.0)
        # Put prices strictly increase with strike
        assert np.all(np.diff(puts) > 0.0)

        # Put-Call parity: C - P = S_0 e^{-q T} - K e^{-r T}
        for K, c, p in zip(strikes, calls, puts):
            parity_expected = S0 * math.exp(-q * T) - K * math.exp(-r * T)
            assert (c - p) == pytest.approx(parity_expected, abs=1e-8)

    def test_kou_black_scholes_limit(self) -> None:
        """As jump intensity λ -> 0, Kou option price converges to Black–Scholes."""
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.04
        q = 0.01
        sigma = 0.20

        # Very low jump arrival intensity
        kou_bs = KouJumpDiffusion(drift=0.0, sigma=sigma, rate=1e-5, p=0.4, eta1=10.0, eta2=5.0)
        kou_call = kou_bs.call_price(K, T=T, S0=S0, r=r, q=q)

        # Analytical Black-Scholes price
        d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        bs_call = S0 * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(
            -r * T
        ) * stats.norm.cdf(d2)

        assert kou_call == pytest.approx(bs_call, abs=1e-4)

    def test_kou_call_pricing_vs_monte_carlo(self) -> None:
        """Fourier inversion call price matches high-sample Monte Carlo simulation."""
        S0 = 100.0
        K = 100.0
        T = 0.5
        r = 0.05
        q = 0.0
        sigma = 0.15
        rate = 1.5
        p = 0.4
        eta1 = 10.0
        eta2 = 5.0

        kou = KouJumpDiffusion(drift=0.0, sigma=sigma, rate=rate, p=p, eta1=eta1, eta2=eta2)
        fourier_call = kou.call_price(K, T=T, S0=S0, r=r, q=q)

        # Monte Carlo simulation of risk-neutral price paths
        rng = np.random.default_rng(42)
        kappa = kou.expected_relative_jump
        mu_rn = r - q - 0.5 * (sigma**2) - rate * kappa

        paths = spxa.sim.exact.sample_kou(
            drift=mu_rn,
            sigma=sigma,
            rate=rate,
            p=p,
            eta1=eta1,
            eta2=eta2,
            dt=T / 50,
            n_steps=50,
            n_paths=250_000,
            rng=rng,
        )
        s_T = S0 * np.exp(paths[:, -1])
        mc_call = math.exp(-r * T) * np.mean(np.maximum(s_T - K, 0.0))

        assert fourier_call == pytest.approx(mc_call, abs=0.05)

    def test_kou_implied_vol_smile(self) -> None:
        """Kou jump-diffusion generates an implied volatility smile/skew."""
        S0 = 100.0
        T = 0.5
        r = 0.05
        q = 0.0
        kou = KouJumpDiffusion(drift=0.0, sigma=0.15, rate=1.5, p=0.3, eta1=10.0, eta2=5.0)

        strikes = [85.0, 95.0, 100.0, 105.0, 115.0]
        vols = kou.implied_vol_smile(strikes, T=T, S0=S0, r=r, q=q)

        assert len(vols) == len(strikes)
        assert np.all(np.isfinite(vols))
        assert np.all(vols > 0.10)
        # Downward jumps dominate (p=0.3 < 0.5, eta1 > eta2), generating a negative skew: OTM put vol > OTM call vol
        assert vols[0] > vols[-1]

    def test_kou_simulation_paths(self) -> None:
        """Simulate method produces valid shape and initial conditions."""
        kou = KouJumpDiffusion(drift=0.05, sigma=0.2, rate=2.0, p=0.4, eta1=10.0, eta2=5.0)
        rng = np.random.default_rng(123)
        paths = kou.simulate(n_steps=25, time_horizon=1.0, n_paths=100, rng=rng)

        assert paths.shape == (100, 26)
        assert np.all(paths[:, 0] == 0.0)


class TestKouClosedFormCharFunc:
    """char_func is evaluated in closed form, including at complex arguments."""

    kou = KouJumpDiffusion(drift=0.03, sigma=0.16, rate=1.0, p=0.4, eta1=10.0, eta2=5.0)

    def test_matches_characteristic_function(self) -> None:
        for u in (0.0, 0.7, -2.5, 13.0, 0.3 - 0.8j):
            for t in (0.25, 1.0, 3.0):
                assert self.kou.char_func(u, t=t) == pytest.approx(
                    self.kou.characteristic_function(u, t), rel=1e-13, abs=1e-15
                )

    def test_vectorised(self) -> None:
        u = np.linspace(-20.0, 20.0, 41)
        phi = self.kou.char_func(u, t=0.5)
        assert phi.shape == u.shape
        expected = [self.kou.characteristic_function(x, 0.5) for x in u]
        assert np.allclose(phi, expected, rtol=1e-13, atol=1e-15)
        assert np.allclose(self.kou.char_func(-u, t=0.5), np.conj(phi), rtol=1e-13, atol=1e-15)

    def test_exponential_moment(self) -> None:
        """E[exp(X_t)] = exp(t (mu + sigma^2/2 + lam (p eta1/(eta1-1) + q eta2/(eta2+1) - 1)))."""
        k = self.kou
        psi = (
            k.drift
            + 0.5 * k.sigma**2
            + k.rate * (k.p * k.eta1 / (k.eta1 - 1) + k.q * k.eta2 / (k.eta2 + 1) - 1)
        )
        value = k.char_func(-1j, t=2.0)
        assert abs(value.imag) < 1e-14
        assert value.real == pytest.approx(math.exp(2.0 * psi), rel=1e-13)

    def test_cumulants_from_char_func(self) -> None:
        """Numerical derivatives of log phi reproduce the exact first two cumulants."""
        h = 1e-4
        k = self.kou
        lp = [np.log(k.char_func(x, t=1.0)) for x in (-h, 0.0, h)]
        k1 = ((lp[2] - lp[0]) / (2j * h)).real
        k2 = (-(lp[2] - 2 * lp[1] + lp[0]) / h**2).real
        cum = k.cumulants(order=2)
        assert k1 == pytest.approx(cum[1], rel=1e-7)
        assert k2 == pytest.approx(cum[2], rel=1e-6)
