"""
Exactness tests for NIG (Normal Inverse Gaussian) process.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy import NIG


@pytest.mark.exactness
def test_nig_invalid_params_raises() -> None:
    """
    NIG requires |β| < α. Violated parameters must raise ValueError.
    Barndorff-Nielsen (1997): the parameter space is α > |β|.
    """
    with pytest.raises(ValueError):
        NIG(alpha=1.0, beta=1.5, delta=1.0)
    with pytest.raises(ValueError):
        NIG(alpha=1.0, beta=-1.0, delta=1.0)


@pytest.mark.exactness
def test_nig_no_gaussian_component() -> None:
    """
    NIG has σ²=0 — it is a pure-jump process.
    Barndorff-Nielsen (1997): NIG is defined as BM subordinated by IG;
    the resulting Lévy process has no independent Brownian component.
    """
    nig = NIG(alpha=2.0, beta=0.0, delta=1.0)
    assert nig.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)


@pytest.mark.exactness
def test_nig_char_func_at_zero() -> None:
    """
    φ_NIG(0; t) = 1. Characteristic functions satisfy φ(0) = E[1] = 1.
    Barndorff-Nielsen (1997), equation (2.2).
    """
    nig = NIG(alpha=2.0, beta=-0.5, delta=1.0)
    cf = nig.char_func_exact(u=0.0, t=1.0)
    assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_nig_char_func_modulus_leq_1() -> None:
    """
    |φ(u)| ≤ 1 for all u ∈ ℝ. Characteristic functions are bounded by 1.
    """
    nig = NIG(alpha=3.0, beta=-0.5, delta=1.5)
    u_vals = np.linspace(-10.0, 10.0, 100)
    cf_vals = nig.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)


@pytest.mark.exactness
def test_nig_char_func_time_additivity() -> None:
    """
    φ(u; s+t) = φ(u; s) · φ(u; t) for NIG (Lévy process property).
    Equivalent to ψ(u) being linear in t. Sato (1999), Definition 1.6.
    """
    nig = NIG(alpha=2.0, beta=-0.3, delta=1.0)
    u, s, t = 1.0, 0.3, 0.8
    cf_s = complex(nig.char_func_exact(u=u, t=s))
    cf_t = complex(nig.char_func_exact(u=u, t=t))
    cf_st = complex(nig.char_func_exact(u=u, t=s + t))
    assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_nig_symmetric_when_beta_zero() -> None:
    """
    NIG with β=0 is symmetric about μ: characteristic function is real for real u.
    Barndorff-Nielsen (1997): β=0 gives the symmetric NIG distribution.
    """
    nig = NIG(alpha=2.0, beta=0.0, delta=1.0, mu=0.0)
    u_vals = np.array([0.5, 1.0, 2.0, 3.0])
    cf_vals = nig.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals.imag) < 1e-12)


@pytest.mark.exactness
def test_nig_simulate_mean() -> None:
    """
    Empirical mean of NIG_1 converges to μ + δβ/γ.
    Barndorff-Nielsen (1997): E[X_1] = μ + δβ/γ where γ = sqrt(α²-β²).
    """
    rng = np.random.default_rng(20)
    alpha, beta, delta, mu = 3.0, -0.5, 1.0, 0.0
    nig = NIG(alpha=alpha, beta=beta, delta=delta, mu=mu)
    gamma = np.sqrt(alpha**2 - beta**2)
    expected_mean = mu + delta * beta / gamma
    paths = nig.simulate(n_steps=50, n_paths=30_000, T=1.0, rng=rng)
    empirical_mean = paths[:, -1].mean()
    assert empirical_mean == pytest.approx(expected_mean, abs=0.02)


@pytest.mark.exactness
def test_nig_simulate_variance() -> None:
    """
    Empirical variance of NIG_1 converges to δα²/γ³.
    Barndorff-Nielsen (1997): Var(X_1) = δα²/γ³.
    """
    rng = np.random.default_rng(21)
    alpha, beta, delta = 3.0, -0.5, 1.0
    nig = NIG(alpha=alpha, beta=beta, delta=delta, mu=0.0)
    gamma = np.sqrt(alpha**2 - beta**2)
    expected_var = delta * alpha**2 / gamma**3
    paths = nig.simulate(n_steps=50, n_paths=30_000, T=1.0, rng=rng)
    empirical_var = paths[:, -1].var()
    assert empirical_var == pytest.approx(expected_var, rel=0.05)


@pytest.mark.exactness
def test_nig_levy_density_reproduces_cumulants() -> None:
    """
    The Lévy density k(x) = αδ/π · K_1(α|x|)/|x| · e^{βx} must reproduce the
    cumulants of L_1: κ_2 = ∫x² k(x) dx = δα²/γ³ and
    log E[e^{L_1}] - μ = ∫(e^x - 1) k(x) dx = δ(γ - √(α² - (β+1)²)).
    Barndorff-Nielsen (1997), equations (2.4) and (2.2).
    """
    from scipy.integrate import quad

    alpha, beta, delta = 15.0, -5.0, 0.5
    nu = NIG(alpha=alpha, beta=beta, delta=delta).triplet.nu

    def k(x: float) -> float:
        return float(nu.density(np.array([x]))[0])

    gamma = np.sqrt(alpha**2 - beta**2)
    m2 = (
        quad(lambda x: x * x * k(x), -np.inf, 0.0)[0] + quad(lambda x: x * x * k(x), 0.0, np.inf)[0]
    )
    assert m2 == pytest.approx(delta * alpha**2 / gamma**3, rel=1e-8)
    expm = quad(lambda x: np.expm1(x) * k(x), -np.inf, 0.0, limit=200)[0]
    expm += quad(lambda x: np.expm1(x) * k(x), 0.0, np.inf, limit=200)[0]
    assert expm == pytest.approx(delta * (gamma - np.sqrt(alpha**2 - (beta + 1) ** 2)), rel=1e-7)


@pytest.mark.exactness
def test_nig_triplet_matches_char_func() -> None:
    """
    The triplet (b, 0, ν) must reproduce the NIG characteristic exponent
    iμu + δ(γ - √(α² - (β + iu)²)) (Barndorff-Nielsen, 1997) through the
    Lévy–Khintchine formula with truncation 𝟙_{|x|≤1}.
    """
    from scipy.integrate import quad

    nig = NIG(alpha=15.0, beta=-5.0, delta=0.5, mu=0.02)
    tr = nig.triplet

    def k(x: float) -> float:
        return float(tr.nu.density(np.array([x]))[0])

    for u in (1.0, 5.0):
        re = im = 0.0
        for a, b in ((-np.inf, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, np.inf)):
            small = np.isfinite(a) and np.isfinite(b)
            re += quad(lambda x: (np.cos(u * x) - 1.0) * k(x), a, b, limit=400)[0]
            im += quad(
                lambda x: (np.sin(u * x) - (u * x if small else 0.0)) * k(x), a, b, limit=400
            )[0]
        expected = complex(np.log(complex(nig.char_func(u, t=1.0))))
        assert re == pytest.approx(expected.real, abs=1e-8)
        assert u * tr.b + im == pytest.approx(expected.imag, abs=1e-8)


@pytest.mark.exactness
def test_nig_tail_is_integral_of_density() -> None:
    """ν((x, ∞)) must equal the integral of the Lévy density over (x, ∞)."""
    from scipy.integrate import quad

    nu = NIG(alpha=15.0, beta=-5.0, delta=0.5).triplet.nu
    for x in (0.05, 0.3):
        expected = quad(lambda t: float(nu.density(np.array([t]))[0]), x, np.inf)[0]
        assert nu.tail(x) == pytest.approx(expected, rel=1e-10)
    # α|x| K_1(α|x|) → 1 as x → 0, so ν((x, ∞)) ~ δ/(π x)
    assert nu.tail(1e-4) * 1e-4 == pytest.approx(0.5 / np.pi, rel=1e-2)
