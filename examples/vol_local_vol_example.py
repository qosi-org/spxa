"""
Local volatility with spxa.vol.

Takes a Heston-like SSVI surface, computes its Dupire local volatility and
checks it three ways: the forward Dupire PDE reprices the surface, Monte
Carlo on a tabulated grid reprices it too, and the Fokker-Planck density
is a proper density with the right forward. Then prices an American put, a
barrier pair and forward-start calls under the same model.

Run with: python examples/vol_local_vol_example.py
"""

import numpy as np

from spxa.market.curves import EquityForward
from spxa.vol import (
    SSVI,
    HestonLikePhi,
    LocalVolGrid,
    LocalVolSurface,
    VolSurface,
    local_vol_diagnostics,
    local_vol_paths,
)

S0, R, Q = 100.0, 0.03, 0.01
FWD = EquityForward(S0, R, dividend_yield=Q)
SURF = VolSurface(SSVI(-0.6, HestonLikePhi(2.0), lambda t: 0.04 * t + 0.003 * t * t), FWD)
LV = LocalVolSurface(SURF)

# 1. the local volatility surface
print("Local vol (rows: T, columns: K/F = 0.8, 0.9, 1.0, 1.1, 1.2)")
for T in (0.25, 0.5, 1.0):
    F = float(FWD.forward(T))
    row = LV.local_vol(F * np.array([0.8, 0.9, 1.0, 1.1, 1.2]), T)
    print(f"  T = {T:4.2f}: " + "  ".join(f"{s:.4f}" for s in row))
print(f"downside local vol above upside: {bool(row[0] > row[-1])}")

# 2. repricing with the forward Dupire PDE
K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
diag = local_vol_diagnostics(LV, S0, R, Q, K, [0.25, 0.5, 1.0], n_space=801, n_time=400)
T_w, K_w, e_w = diag.worst()
print(
    f"\nDupire PDE reprices the surface: worst error {abs(e_w) * 1e4:.2f} bp of vol at T={T_w}, K={K_w:.0f}"
)
print(f"local vol defined everywhere: {diag.undefined_fraction == 0.0}")

# 3. Monte Carlo on a tabulated grid
grid = LocalVolGrid.from_surface(LV, np.linspace(-2.0, 1.2, 161), np.linspace(0.01, 1.0, 100))
paths = local_vol_paths(grid, S0, R, Q, 1.0, 100, 200_000, seed=3, record=[0.5, 1.0])
ok = []
for k in (90.0, 100.0, 110.0):
    m, se = paths.price(lambda s, k=k: np.maximum(s - k, 0.0), 1.0)
    ok.append(abs(m - float(SURF.price(k, 1.0))) < 4 * se)
print(f"\nMonte Carlo calls within 4 standard errors of the surface: {all(ok)}")

# 4. the Fokker-Planck density at six months
y, pdf = LV.density(0.5, S0, R, Q)
h = y[1] - y[0]
print(f"density mass {pdf.sum() * h:.8f}, E[S/F] {np.sum(np.exp(y) * pdf) * h:.6f}")

# 5. exotics under local vol
am = LV.american(100.0, 1.0, S0, R, Q, "put").price
eu = float(SURF.price(100.0, 1.0, "put"))
print(f"\nATM 1y put: American {am:.4f}, European {eu:.4f}, early-exercise premium {am - eu:.4f}")
do = LV.barrier(100.0, 1.0, S0, 85.0, R, Q, "down-and-out").price
di = LV.barrier(100.0, 1.0, S0, 85.0, R, Q, "down-and-in").price
van = float(SURF.price(100.0, 1.0))
print(
    f"down-and-out {do:.4f} + down-and-in {di:.4f} = vanilla {van:.4f}: {abs(do + di - van) < 1e-4 * van}"
)
c, se = paths.forward_start(0.5, 1.0, np.array([0.9, 1.0, 1.1]))
print("forward-start calls (T1 = 0.5, T2 = 1): " + ", ".join(f"{x:.4f}" for x in c))
