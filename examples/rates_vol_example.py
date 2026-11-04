"""
Interest-rate volatility with spxa.rates.

Strips caplet vols from flat cap quotes and checks the caps reprice. Then
calibrates SABR to a swaption smile and compares Hagan's formula with the
arbitrage-free density PDE at a long, low-rate expiry. Finally builds a
small SABR cube and prices a CMS coupon by replication.

Run with: python examples/rates_vol_example.py
"""

import warnings

import numpy as np

from spxa.market import FlatCurve
from spxa.market.black import black_price
from spxa.rates import (
    EURIBOR6M,
    SABR,
    ArbitrageFreeSABR,
    CMSReplication,
    SwaptionCube,
    calibrate_sabr,
    cap_floor,
    strip_caplet_vols,
    swaption,
)

REF = "2024-03-15"
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)

# 1. caplet stripping (illustrative quotes, not market data)
tenors = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
flat = [0.30, 0.28, 0.26, 0.24, 0.23, 0.22]
caps = [cap_floor(REF, t, 0.035, EURIBOR6M, notional=1e6) for t in tenors]
curve = strip_caplet_vols(caps, flat, DISC, PROJ)
print("Caplet vols by expiry:")
for T, v in zip(curve.expiries, curve.vols):
    print(f"  up to {T:5.2f}y: {v:.4f}")
err = max(
    abs(c.price(DISC, PROJ, curve) / c.price(DISC, PROJ, fv) - 1) for c, fv in zip(caps, flat)
)
print(f"caps reprice from stripped vols (< 1e-10): {err < 1e-10}")

# 2. SABR calibration to a 5Y x 10Y smile
sw = swaption(REF, "5Y", "10Y", 0.03)
F = sw.forward_rate(DISC, PROJ)
T = sw.time_to_expiry(DISC)
true = SABR(0.06, 0.5, -0.25, 0.35)
K = F + np.array([-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02])
fit = calibrate_sabr(F, K, true.lognormal_vol(F, K, T), T, beta=0.5)
print(f"\n5Y x 10Y forward {F:.4%}, fitted rho {fit.model.rho:+.4f}, nu {fit.model.nu:.4f}")
print(f"SABR smile recovered (rmse < 1e-10): {fit.rmse < 1e-10}")

# 3. Hagan versus the arbitrage-free PDE at F = 1%, T = 10
low = SABR(0.02, 0.5, -0.3, 0.5)
k = np.linspace(1e-4, 0.01, 400)
neg = np.min(np.diff(low.price(0.01, k, 10.0), 2)) < 0
dens = ArbitrageFreeSABR(low, 0.01, 10.0).solve()
print(f"\nHagan density negative near zero: {neg}")
print(f"PDE mass {dens.mass():.12f}, absorbed at zero {dens.P_L:.2%}")

# 4. a SABR cube and a CMS coupon
exp, ten = [1.0, 5.0, 10.0], [2.0, 10.0]
fwd = [[0.03 + 0.001 * e for _ in ten] for e in exp]
strikes = [[f * np.exp(np.linspace(-0.5, 0.5, 9)) for f in row] for row in fwd]
vols = [[SABR(0.035, 0.5, -0.2, 0.4 / np.sqrt(e)).lognormal_vol(fwd[i][j], strikes[i][j], e)
         for j in range(2)] for i, e in enumerate(exp)]  # fmt: skip
cube = SwaptionCube.calibrate(exp, ten, fwd, strikes, vols, beta=0.5)
m = cube.sabr(5.0, 10.0)
print(f"\ncube at 5Y x 10Y: alpha {m.alpha:.4f}, rho {m.rho:+.4f}, nu {m.nu:.4f}")
A0, P = sw.annuity(DISC), float(DISC.discount(T))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cms_sabr = CMSReplication(
        F, T, A0, P, lambda x, o: float(m.price(F, max(x, 1e-8), T, 1.0, o)), n=10
    )
    atm = float(m.lognormal_vol(F, F, T))
    cms_flat = CMSReplication(
        F, T, A0, P, lambda x, o: float(black_price(F, max(x, 1e-12), T, atm, 1.0, o)), n=10
    )
    a_s, a_f = cms_sabr.convexity_adjustment(), cms_flat.convexity_adjustment()
print(f"CMS convexity: flat ATM {a_f * 1e4:.1f} bp, SABR smile {a_s * 1e4:.1f} bp")
# beta = 0.5 and rho < 0 lower the vols at high strikes, where the CMS payoff
# is most convex, so here the smile reduces the adjustment
print(f"smile lowers the adjustment (skewed right wing): {a_s < a_f}")
