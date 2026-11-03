"""
Arbitrage-free implied-volatility surfaces with spxa.vol.

Starts from Axel Vogt's butterfly-arbitrage slice and repairs it, then fits noisy quotes
generated from an SSVI surface three ways (raw SVI per slice, calendar-free SVI and eSSVI).
It interpolates the surface in time without arbitrage, then checks the implied
distribution: its mass, the martingale property, Lee's wings and Monte Carlo repricing.

Run with: python examples/vol_svi_surface_example.py
"""

import numpy as np

from spxa.vol import (
    SSVI,
    ImpliedDistribution,
    PowerLawPhi,
    PriceInterpolatedSurface,
    SVIRaw,
    calibrate_essvi,
    calibrate_svi,
    calibrate_svi_butterfly_free,
    calibrate_svi_surface,
    check_butterfly,
    check_calendar,
    normalised_call_from_total_variance,
    wing_asymptotics,
)

print("=" * 72)
print("spxa.vol — SVI, SSVI and eSSVI surfaces")
print("=" * 72)

# 1. Butterfly arbitrage in a raw SVI slice and two ways to remove it
vogt = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
chk = check_butterfly(vogt)
print("\n1. Vogt's slice (Gatheral & Jacquier 2014, Example 3.1)")
print(f"   min g = {chk.min_g:.6f} at k = {chk.argmin:.3f}  -> butterfly-free: {bool(chk)}")
k = np.linspace(-1.5, 1.5, 61)
fit = calibrate_svi_butterfly_free(k, vogt(k))
print(
    f"   constrained refit: rmse {fit.rmse:.5f}, butterfly-free: {bool(check_butterfly(fit.params))}"
)
proj = vogt.to_jump_wings(1.0).butterfly_free_projection().to_ssvi_natural()
print(f"   jump-wings projection (keeps ATM level and skew): {bool(check_butterfly(proj))}")

# 2. Quotes from an SSVI surface with noise
true = SSVI(rho=-0.7, phi=PowerLawPhi(eta=1.1, gamma=0.4), theta=lambda t: 0.04 * t + 0.003 * t * t)
expiries = [0.1, 0.25, 0.5, 1.0, 2.0]
rng = np.random.default_rng(2024)
ks, ws = [], []
for t in expiries:
    kk = np.linspace(-0.8, 0.4, 25) * np.sqrt(t / 0.5)
    vol = np.sqrt(true.total_variance(kk, t) / t) + rng.normal(0.0, 0.001, kk.size)
    ks.append(kk)
    ws.append(vol * vol * t)

print("\n2. Five noisy slices (vol noise 10bp) from SSVI(rho=-0.7, eta=1.1, gamma=0.4)")


def vol_rmse(slices) -> float:
    """Largest root-mean-square implied-vol error over the slices."""
    errs = [
        np.sqrt(np.mean((np.sqrt(sl(kk) / t) - np.sqrt(ww / t)) ** 2))
        for sl, kk, ww, t in zip(slices, ks, ws, expiries)
    ]
    return float(max(errs))


free = [calibrate_svi(kk, ww).params for kk, ww in zip(ks, ws)]
grid = np.linspace(-4, 4, 4001)
print(
    f"   raw SVI per slice:       calendar-free {bool(check_calendar(free, grid))}, "
    f"max vol rmse {vol_rmse(free):.5f}"
)
surf = calibrate_svi_surface(expiries, ks, ws)
print(
    f"   calendar-free SVI:       calendar-free {bool(check_calendar(surf.slices, grid))}, "
    f"max vol rmse {vol_rmse(surf.slices):.5f}"
)
ess = calibrate_essvi(expiries, ks, ws)
ess_slices = [ess.slice(t) for t in expiries]
print(
    f"   eSSVI (Corbetta et al.): calendar-free {bool(check_calendar(ess_slices, grid))}, "
    f"max vol rmse {vol_rmse(ess_slices):.5f}, "
    f"butterfly-free {all(bool(check_butterfly(s, grid)) for s in ess_slices)}"
)

# 3. Interpolation in time between the calibrated slices
pis = PriceInterpolatedSurface(expiries, surf.slices)
ts = np.linspace(0.05, 2.0, 40)
prices = np.array([pis.call_price(grid[1500:2600], t) for t in ts])
print("\n3. Gatheral-Jacquier price interpolation")
print(
    f"   prices non-decreasing in t on 40 expiries: {bool(np.all(np.diff(prices, axis=0) >= -1e-15))}"
)
print(
    f"   ATM vol at t=0.75: {pis.implied_vol(0.0, 0.75):.4f}  (true {np.sqrt(true.total_variance(0.0, 0.75) / 0.75):.4f})"
)

# 4. The implied distribution of the 1Y slice
s1 = surf.slices[3]
dist = ImpliedDistribution(s1)
wings = wing_asymptotics(s1)
print("\n4. Implied distribution of the 1Y slice")
print(f"   mass {dist.total_mass():.10f}   E[S/F] {dist.forward_ratio():.10f}")
print(
    f"   wing slopes L/R {wings.left_slope:.4f}/{wings.right_slope:.4f}  "
    f"critical moments q~ {wings.left_moment:.3f}, p~ {wings.right_moment:.3f}"
)
x = dist.sample(400_000, rng=7)
st = np.exp(x)
ok = True
for kk in (-0.3, 0.0, 0.2):
    pay = np.maximum(st - np.exp(kk), 0.0)
    ref = normalised_call_from_total_variance(kk, s1(kk))
    se = pay.std() / np.sqrt(pay.size)
    ok &= abs(pay.mean() - ref) < 4 * se
    print(f"   call k={kk:+.1f}: MC {pay.mean():.6f} +- {se:.6f}   Black {ref:.6f}")
print(f"   Monte Carlo within 4 standard errors: {ok}")
