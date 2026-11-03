"""Black deltas and strike-from-delta conversion."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from spxa.market.black import black_greeks, black_price
from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.delta import atm_dns_strike, black_delta, strike_from_delta
from spxa.vol.surface import VolSurface

F, T, SIG = 100.0, 0.5, 0.25


def test_delta_conventions() -> None:
    K = np.array([80.0, 100.0, 125.0])
    d1 = (np.log(F / K) + 0.5 * SIG**2 * T) / (SIG * math.sqrt(T))
    d2 = d1 - SIG * math.sqrt(T)
    assert black_delta(F, K, T, SIG) == pytest.approx(norm.cdf(d1), rel=1e-14)
    assert black_delta(F, K, T, SIG, "put") == pytest.approx(-norm.cdf(-d1), rel=1e-14)
    # forward delta is dV/dF of the undiscounted Black price
    g = black_greeks(F, K, T, SIG, 1.0, "call")
    assert black_delta(F, K, T, SIG) == pytest.approx(g["delta"], rel=1e-12)
    # premium-adjusted = delta minus premium in units of the forward
    pa = black_delta(F, K, T, SIG, premium_adjusted=True)
    assert pa == pytest.approx(K / F * norm.cdf(d2), rel=1e-14)
    assert pa == pytest.approx(norm.cdf(d1) - black_price(F, K, T, SIG) / F, rel=1e-12)
    assert black_delta(F, K, T, SIG, kind="spot", carry=0.98) == pytest.approx(0.98 * norm.cdf(d1))
    with pytest.raises(ValueError):
        black_delta(F, K, T, SIG, "straddle")
    with pytest.raises(ValueError):
        black_delta(F, K, T, SIG, kind="cash")


@pytest.mark.parametrize(
    "option,delta", [("call", 0.25), ("call", 0.5), ("put", -0.25), ("put", -0.1)]
)
@pytest.mark.parametrize("pa", [False, True])
def test_round_trip_flat(option: str, delta: float, pa: bool) -> None:
    K = strike_from_delta(delta, F, T, SIG, option, premium_adjusted=pa)
    assert black_delta(F, K, T, SIG, option, premium_adjusted=pa) == pytest.approx(delta, abs=1e-13)
    Ks = strike_from_delta(0.97 * delta, F, T, SIG, option, "spot", pa, carry=0.97)
    assert Ks == pytest.approx(K, rel=1e-12)


def test_closed_form_and_pa_branch() -> None:
    sw = SIG * math.sqrt(T)
    K = strike_from_delta(0.25, F, T, SIG)
    assert K == pytest.approx(F * math.exp(-norm.ppf(0.25) * sw + sw * sw / 2), rel=1e-14)
    # PA call delta is not monotone: the returned strike is on the upper branch
    Kpa = strike_from_delta(0.25, F, T, SIG, premium_adjusted=True)
    ks = np.linspace(-1.5, 1.5, 3001)
    vals = black_delta(F, F * np.exp(ks), T, SIG, premium_adjusted=True)
    k_peak = ks[np.argmax(vals)]
    assert math.log(Kpa / F) > k_peak
    assert Kpa < K  # premium adjustment lowers call delta, so the 25-delta strike moves in
    with pytest.raises(ValueError):
        strike_from_delta(1.2, F, T, SIG)
    with pytest.raises(ValueError):
        strike_from_delta(0.25, F, T, SIG, "put")
    with pytest.raises(ValueError):
        strike_from_delta(0.99, F, T, SIG, premium_adjusted=True)  # above the PA maximum


def test_atm_dns() -> None:
    w = SIG**2 * T
    K = atm_dns_strike(F, T, SIG)
    assert K == pytest.approx(F * math.exp(w / 2), rel=1e-15)
    straddle = black_delta(F, K, T, SIG) + black_delta(F, K, T, SIG, "put")
    assert straddle == pytest.approx(0.0, abs=1e-15)
    Kpa = atm_dns_strike(F, T, SIG, premium_adjusted=True)
    assert Kpa == pytest.approx(F * math.exp(-w / 2), rel=1e-15)
    pa = black_delta(F, Kpa, T, SIG, premium_adjusted=True) + black_delta(
        F, Kpa, T, SIG, "put", premium_adjusted=True
    )
    assert pa == pytest.approx(0.0, abs=1e-15)


def test_smile_and_surface() -> None:
    ss = SSVI(-0.6, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)
    fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
    surf = VolSurface(ss, fwd)
    Ft = fwd.forward(T)
    for option, delta in (("call", 0.25), ("put", -0.25), ("call", 0.1)):
        K = surf.strike_from_delta(delta, T, option)
        assert surf.delta(K, T, option) == pytest.approx(delta, abs=1e-13)
        assert black_delta(Ft, K, T, surf.implied_vol(K, T), option) == pytest.approx(
            delta, abs=1e-13
        )
    # spot delta: carry Q = F D / S = exp(-q T)
    Kc = surf.strike_from_delta(0.25 * math.exp(-0.01 * T), T, kind="spot")
    assert Kc == pytest.approx(surf.strike_from_delta(0.25, T), rel=1e-12)
    smile = lambda k: float(ss.implied_vol(k, T))  # noqa: E731
    Kd = atm_dns_strike(Ft, T, smile)
    assert math.log(Kd / Ft) == pytest.approx(0.5 * smile(math.log(Kd / Ft)) ** 2 * T, abs=1e-15)
    with pytest.raises(ValueError):
        VolSurface(ss, _NoSpot()).delta(100.0, T, kind="spot")


class _NoSpot:
    def forward(self, t: float) -> float:
        return 100.0

    def discount(self, t: float) -> float:
        return 1.0
