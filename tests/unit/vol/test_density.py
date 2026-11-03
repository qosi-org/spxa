"""Risk-neutral distribution implied by a total-variance slice."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from spxa.vol import (
    SSVI,
    ImpliedDistribution,
    PowerLawPhi,
    SVIRaw,
    normalised_call_from_total_variance,
)

GOOD = SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2)
VOGT = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


class Flat:
    """Constant total variance: X ~ N(-w/2, w)."""

    def __init__(self, w: float) -> None:
        self.w = w

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(k, float), self.w)

    def dw(self, k: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(k, float))

    def d2w(self, k: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(k, float))


def test_flat_slice_is_gaussian() -> None:
    w = 0.09
    d = ImpliedDistribution(Flat(w))
    k = np.linspace(-1.5, 1.5, 31)
    ref = norm(loc=-w / 2, scale=np.sqrt(w))
    assert np.allclose(d.pdf(k), ref.pdf(k), rtol=1e-13, atol=0)
    assert np.allclose(d.cdf(k), ref.cdf(k), rtol=0, atol=2e-16)
    assert d.moment(1) == pytest.approx(-w / 2, abs=1e-12)
    assert d.moment(2) == pytest.approx(w + w * w / 4, abs=1e-12)
    assert d.quantile(0.5) == pytest.approx(-w / 2, abs=1e-12)
    assert d.quantile(np.array([0.1, 0.9])) == pytest.approx(ref.ppf([0.1, 0.9]), abs=1e-11)


def test_survival_is_minus_strike_derivative_of_call() -> None:
    # Q(S_T > K) = -dC/dK with F = 1, K = e^k
    d = ImpliedDistribution(GOOD)
    k = np.linspace(-1.0, 1.0, 21)
    K = np.exp(k)
    h = 1e-5

    def c(KK: np.ndarray) -> np.ndarray:
        x = np.log(KK)
        return normalised_call_from_total_variance(x, GOOD(x))

    dc = (c(K * (1 + h)) - c(K * (1 - h))) / (2 * K * h)
    assert np.allclose(d.digital_call(k), -dc, rtol=0, atol=1e-9)
    assert np.allclose(d.digital_put(k) + d.digital_call(k), 1.0, rtol=0, atol=1e-15)


def test_cdf_is_the_integral_of_pdf() -> None:
    d = ImpliedDistribution(GOOD)
    for k in (-0.8, -0.1, 0.0, 0.3, 1.2):
        mass, _ = quad(d.pdf, d.k_min, k, epsabs=1e-14, epsrel=1e-13, limit=200)
        assert mass == pytest.approx(d.cdf(k), abs=1e-11)


def test_martingale_and_repricing() -> None:
    d = ImpliedDistribution(GOOD)
    assert d.butterfly_free
    # the default truncation leaves less than 1e-17 of (share-weighted) tail mass
    assert d.cdf(d.k_min) < 1e-17 and np.exp(d.k_max) * d.survival(d.k_max) < 1e-17
    assert d.total_mass() == pytest.approx(1.0, abs=1e-11)
    assert d.forward_ratio() == pytest.approx(1.0, abs=1e-11)
    for k in (-0.5, 0.0, 0.4):
        ref = normalised_call_from_total_variance(k, GOOD(k))
        assert d.call(k) == pytest.approx(ref, abs=1e-11)


def test_ssvi_slice_distribution() -> None:
    # an arbitrage-free SSVI slice (power law with eta (1+|rho|) <= 2)
    s = SSVI(rho=-0.6, phi=PowerLawPhi(eta=1.0, gamma=0.4), theta=lambda t: 0.04 * t).slice(1.0)
    d = ImpliedDistribution(s)
    assert d.butterfly_free
    assert d.forward_ratio() == pytest.approx(1.0, abs=1e-10)
    u = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert np.allclose(d.cdf(d.quantile(u)), u, rtol=0, atol=1e-13)


def test_butterfly_arbitrage_shows_as_decreasing_cdf() -> None:
    d = ImpliedDistribution(VOGT)
    assert not d.butterfly_free
    k = np.linspace(0.5, 1.3, 801)
    assert np.min(np.diff(d.cdf(k))) < 0
    with pytest.raises(ValueError, match="butterfly"):
        d.quantile(0.5)
    with pytest.raises(ValueError, match="butterfly"):
        d.sample(10)


def test_sampling_reprices_forward_and_calls() -> None:
    d = ImpliedDistribution(GOOD)
    x = d.sample(400_000, rng=7)
    s = np.exp(x)
    se = s.std() / np.sqrt(s.size)
    assert abs(s.mean() - 1.0) < 4 * se
    pay = np.maximum(s - 1.0, 0.0)
    ref = normalised_call_from_total_variance(0.0, GOOD(0.0))
    assert abs(pay.mean() - ref) < 4 * pay.std() / np.sqrt(s.size)


def test_validation() -> None:
    with pytest.raises(ValueError):
        ImpliedDistribution(GOOD, k_min=1.0, k_max=-1.0)
    with pytest.raises(ValueError):
        ImpliedDistribution(lambda k: np.zeros_like(np.asarray(k, float)))
    with pytest.raises(ValueError, match="probabilities"):
        ImpliedDistribution(GOOD).quantile(1.0)
