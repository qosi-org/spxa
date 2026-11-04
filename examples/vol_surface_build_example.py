"""
From option quotes to a risk-ready volatility surface with spxa.vol.

Generates bid/ask chains from a skewed SSVI surface, then:
1. builds an arbitrage-free SVI surface from them, with parity-implied forwards
   and a quality report;
2. reads strikes off it by delta;
3. computes deltas under two smile dynamics, and bucketed vega;
4. saves the surface to JSON and checks it by Monte Carlo.

Run with: python examples/vol_surface_build_example.py
"""

import math

import numpy as np

from spxa.market.black import black_price
from spxa.market.quotes import OptionChain, OptionSurfaceQuotes
from spxa.vol import (
    SSVI,
    PowerLawPhi,
    bucketed_vega,
    build_vol_surface,
    dumps,
    expiry_buckets,
    loads,
    parallel_bump,
)

S, R, Q = 100.0, 0.03, 0.015
TRUE = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
EXPIRIES = [0.1, 0.25, 0.5, 1.0, 2.0]
rng = np.random.default_rng(7)


def chain(T: float) -> OptionChain:
    F, D = S * math.exp((R - Q) * T), math.exp(-R * T)
    K = np.arange(60.0, 145.0, 2.5)
    k = np.log(K / F)
    K = K[np.abs(k) < 2.5 * math.sqrt(0.04 * T + 0.003 * T * T) + 0.05]
    vol = TRUE.implied_vol(np.log(K / F), T) * (1 + 0.002 * rng.standard_normal(K.size))
    c, p = black_price(F, K, T, vol, D, "call"), black_price(F, K, T, vol, D, "put")
    hc, hp = 0.01 * c + np.minimum(0.02, c / 2), 0.01 * p + np.minimum(0.02, p / 2)
    return OptionChain.from_arrays(T, K, c - hc, c + hc, p - hp, p + hp)


print("=" * 72)
print("spxa.vol — building a surface from option quotes")
print("=" * 72)

# 1. Build
quotes = OptionSurfaceQuotes([chain(T) for T in EXPIRIES], spot=S)
build = build_vol_surface(quotes, method="svi", weights="spread")
surf = build.surface
true_F = S * np.exp((R - Q) * np.array(EXPIRIES))
print("\n1. SVI surface from 5 noisy bid/ask chains")
print(
    f"   parity forwards within 1 bp of the truth: {bool(np.all(np.abs(build.forwards / true_F - 1) < 1e-4))}"
)
quality = build.quality()
print("   " + quality.table().replace("\n", "\n   "))
print(f"   arbitrage-free: {quality.arbitrage_free}")
print(f"   worst implied-vol error below 1 vol point: {quality.worst_vol_error < 0.01}")

# 2. Strikes by delta at 1Y
print("\n2. Strikes by forward delta at 1Y")
for d in (0.25, 0.5):
    kc = surf.strike_from_delta(d, 1.0)
    kp = surf.strike_from_delta(-d, 1.0, option="put")
    print(
        f"   {d:.0%} call {kc:7.2f} (vol {surf.implied_vol(kc, 1.0):.2%})   put {kp:7.2f} (vol {surf.implied_vol(kp, 1.0):.2%})"
    )

# 3. Risk
print("\n3. Delta under two smile dynamics (1Y calls)")
K = np.array([90.0, 100.0, 110.0])
ss = surf.greeks(K, 1.0, dynamics="sticky_strike")["delta"]
sm = surf.greeks(K, 1.0, dynamics="sticky_moneyness")["delta"]
for Ki, a, b in zip(K, ss, sm):
    print(f"   K = {Ki:5.1f}: sticky strike {a:.4f}   sticky moneyness {b:.4f}")
print(f"   sticky moneyness larger under negative skew: {bool(np.all(sm > ss))}")


def book(s) -> float:
    return float(s.price(100.0, 0.75) + 2 * s.price(110.0, 1.5))


size = 1e-4
buckets = bucketed_vega(book, surf, expiry_buckets(EXPIRIES, size), size)
parallel = bucketed_vega(book, surf, [parallel_bump(size)], size)[0]
print("   bucketed vega of a 0.75y + 1.5y book (per vol point):")
print("   " + "  ".join(f"{T:g}y {v / 100:7.4f}" for T, v in zip(EXPIRIES, buckets)))
print(
    f"   buckets sum to the parallel vega: {abs(buckets.sum() - parallel) < 1e-6 * abs(parallel)}"
)

# 4. Persistence and a Monte Carlo check
print("\n4. JSON round trip and Monte Carlo")
back = loads(dumps(build.surface))
grid = np.linspace(70.0, 140.0, 15)
print(
    f"   reloaded surface identical: {bool(np.array_equal(back.implied_vol(grid, 1.0), surf.implied_vol(grid, 1.0)))}"
)
ST = surf.sample_terminal(1.0, 200_000, rng=np.random.default_rng(11))
D, F = surf.discount(1.0), surf.forward(1.0)
payoff = D * np.maximum(ST - 100.0, 0.0)
se = payoff.std() / math.sqrt(payoff.size)
se_f = ST.std() / math.sqrt(ST.size) / F
print(f"   martingale: E[S_T]/F - 1 within 4 standard errors: {abs(ST.mean() / F - 1) < 4 * se_f}")
print(
    f"   100-strike call: surface {surf.price(100.0, 1.0):.4f}, MC {payoff.mean():.4f} +- {se:.4f}"
)
print(f"   within 4 standard errors: {abs(payoff.mean() - surf.price(100.0, 1.0)) < 4 * se}")
