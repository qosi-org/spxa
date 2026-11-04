"""
spxa.vol and spxa.rates working together.

1. An ESTR curve bootstrapped from OIS quotes, and key-rate sensitivities of
   the 10Y par rate from one reverse pass through a graph node.
2. The implied-volatility surface of a Merton jump diffusion from the zoo.
3. A book of index puts on that surface and payer swaps on the curve, run
   through the historical stress scenarios of spxa.risk.
4. Delta and vega of a put from a surface-price graph node.

Run with: python examples/vol_rates_risk_example.py
"""

import numpy as np

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.graph.rates import par_swap_rate_node
from spxa.graph.vol import surface_price_node
from spxa.rates import (
    ESTR,
    MonotoneConvex,
    SwapHelper,
    bootstrap_curve,
    curve_price,
    curve_pricing_fn,
    ois_swap,
)
from spxa.risk.instruments import EuropeanOption, InterestRateSwap
from spxa.risk.scenarios import StressTestingEngine
from spxa.vol import surface_price, surface_pricing_fn
from spxa.zoo.levy.merton import MertonJumpDiffusion

REF = "2024-01-15"
OIS = {"3M": 3.905, "6M": 3.83, "1Y": 3.56, "2Y": 3.02, "3Y": 2.78, "5Y": 2.60,
       "7Y": 2.56, "10Y": 2.58, "15Y": 2.62, "20Y": 2.58, "30Y": 2.42}  # fmt: skip

print("=" * 72)
print("spxa.vol + spxa.rates: curves, surfaces, graphs and stress tests")
print("=" * 72)

# 1. Curve and key rates
print("\n1. ESTR curve and key-rate sensitivities of the 10Y par rate")
curve = bootstrap_curve(
    [SwapHelper(ois_swap(REF, t, 0.0, ESTR), q / 100) for t, q in OIS.items()],
    REF,
    MonotoneConvex(),
).curve
pillars = [float(t) for t in curve.node_times]
zeros = [ParameterNode(float(z), f"z{i}") for i, z in enumerate(curve.node_zero_rates)]
par = par_swap_rate_node(zeros, pillars, 10.0, frequency=1.0)
g = ComputationalGraph("par10")
g.add_output(par, name="rate")
rate10 = g.evaluate()["rate"]
grads = GradientTape(g).gradient("rate")
print(f"   10Y annual par rate {rate10:.4%}")
for label, i in (("7Y", 6), ("10Y", 7), ("15Y", 8)):
    print(f"   d rate / d {label:>3s} zero {grads[zeros[i]]:+.4f}")
print(f"   sum of key rates {sum(grads[z] for z in zeros):.4f}")

# 2. Surface of a jump diffusion
print("\n2. Implied surface of a Merton jump diffusion (sigma 15%, 0.5 jumps/yr of -10%)")
merton = MertonJumpDiffusion(sigma=0.15, rate=0.5, jump_mean=-0.1, jump_std=0.15)
r = float(curve.zero_rate(1.0))
fit = merton.to_vol_surface(4800.0, r, [0.25, 0.5, 1.0, 2.0], dividend_yield=0.013)
surface = fit.surface
for T in (0.25, 1.0, 2.0):
    print(
        f"   T={T:<4} ATM {float(surface.atm_vol(T)):.2%}  90% {float(surface.implied_vol(4320.0, T)):.2%}"
        f"  110% {float(surface.implied_vol(5280.0, T)):.2%}"
    )
print(f"   largest SVI misfit {fit.max_vol_error * 1e4:.1f} bp")

# 3. Stress test of a mixed book
print("\n3. Stress scenarios on 100 1Y 90% puts and a 10Y payer on 10m")
put = EuropeanOption("SPX_P90", "SPX", 4320.0, 1.0, is_call=False, notional=100.0)
swap = InterestRateSwap("IRS10", "ESTR", round(rate10, 4), 10.0, 1.0, notional=1e7)
base = {"SPX": 4800.0, "SPX_SURFACE": surface, "SPX_VOL": 0.2, "ESTR_CURVE": curve, "r": r}
vol_fn = surface_pricing_fn([(put, 1.0)], base, dynamics="sticky_moneyness")
rates_fn = curve_pricing_fn([(swap, 1.0)], base)
print(f"   put {surface_price(put, base).price:,.0f}   swap {curve_price(swap, base).price:,.0f}"
      f"   swap DV01 {curve_price(swap, base).dv01:,.0f}")  # fmt: skip
legs = {"puts": vol_fn, "swap": rates_fn, "book": lambda m: vol_fn(m) + rates_fn(m)}
runs = {k: StressTestingEngine(f).run_historical_suite(base) for k, f in legs.items()}
print(f"   {'scenario':22s} {'puts':>12s} {'swap':>12s} {'book':>12s}")
for name in runs["book"]:
    pnl = [runs[k][name]["pnl_impact"] for k in legs]
    print(f"   {name:22s} {pnl[0]:>12,.0f} {pnl[1]:>12,.0f} {pnl[2]:>12,.0f}")

# 4. Greeks from a graph node
print("\n4. A 1Y 90% put as a graph node")
s, dv = ParameterNode(4800.0, "spot"), ParameterNode(0.0, "dvol")
node = surface_price_node(s, dv, surface=surface, strike=4320.0, maturity=1.0, option="put")
g2 = ComputationalGraph("put")
g2.add_output(node, name="price")
greeks = GradientTape(g2).gradient("price")
quote = surface_price(EuropeanOption("P", "SPX", 4320.0, 1.0, is_call=False), base)
print(f"   price {g2.evaluate()['price']:.4f}  delta {greeks[s]:.5f}  vega {greeks[dv]:.2f}")
print(f"   matches the risk adapter: {np.isclose(greeks[s], quote.delta, rtol=1e-12)}")
