"""
Market models with spxa.rates.

Calibrates a LIBOR market model to a humped caplet curve and a grid of
swaption volatilities, then checks Rebonato's swaption formula against
simulation. Next it compares backward-looking (compounded in arrears) and
forward-looking caplets in the forward market model, and prices a caplet
smile in the SABR-LMM by Monte Carlo against Hagan's formula.

Run with: python examples/rates_lmm_example.py
"""

import numpy as np

from spxa.market import FlatCurve, implied_black_volatility
from spxa.rates import (
    AbcdVol,
    ForwardMarketModel,
    LiborMarketModel,
    SABRLMM,
    calibrate_lmm,
    fmm_paths,
    lmm_paths,
    rebonato_correlation,
    sabr_lmm_paths,
)

CURVE = FlatCurve(0.03)
TIMES = np.arange(1.0, 11.01, 0.5)  # 20 semi-annual forwards fixing from 1Y to 10.5Y
FIX = TIMES[:-1]


def lmm() -> LiborMarketModel:
    print("1. LIBOR market model calibration")
    # synthetic market: humped caplet vols, and swaption vols from a Rebonato
    # correlation with a few bp of quote noise
    market = LiborMarketModel(
        CURVE,
        TIMES,
        AbcdVol(0.05, 0.09, 0.44, 0.11),
        correlation=rebonato_correlation(FIX, 0.12, 0.45),
    )
    caplet_vols = [market.caplet_vol(i) for i in range(market.n)]
    grid = [(2, 6), (2, 12), (2, 20), (6, 12), (6, 20), (10, 20)]
    noise = np.array([3.0, -2.0, 1.0, -3.0, 2.0, -1.0]) * 1e-4
    quotes = np.array([market.swaption_vol(a, b) for a, b in grid]) + noise
    cal = calibrate_lmm(CURVE, TIMES, caplet_vols, grid, quotes)
    m = cal.model
    exact = max(abs(m.caplet_vol(i) - v) for i, v in enumerate(caplet_vols))
    print(
        f"   abcd {m.vol.a:.4f} {m.vol.b:.4f} {m.vol.c:.4f} {m.vol.d:.4f}, k in [{m.k.min():.3f}, {m.k.max():.3f}]"
    )
    print(f"   caplets exact (< 1e-12): {exact < 1e-12}")
    print(f"   correlation beta {cal.beta:.4f}, rho_inf {cal.rho_inf:.4f}")
    print(f"   swaption vols: max error {cal.max_swaption_error * 1e4:.1f} bp")
    paths = lmm_paths(m, 40_000, 2, seed=1)
    a, b = 2, 12
    bonds = np.cumprod(1.0 / (1.0 + m.tau[a:b] * paths.forwards[a][:, a:b]), axis=1)
    annuity = bonds @ m.tau[a:b]
    swap = (1.0 - bonds[:, -1]) / annuity
    K = m.swap_weights(a, b)[1]
    pv, se = paths.price(annuity * np.maximum(swap - K, 0.0), a)
    formula = m.swaption(a, b, K)
    print(
        f"   2Y x 5Y ATM payer: Rebonato {formula * 1e4:.2f} bp, MC {pv * 1e4:.2f} +- {se * 1e4:.2f} bp"
    )
    print(f"   within 3 SE: {abs(pv - formula) < 3 * se}")
    return m


def fmm() -> None:
    print("2. Backward-looking caplets (forward market model)")
    vol = AbcdVol(0.05, 0.09, 0.44, 0.11)
    rho = rebonato_correlation(FIX, 0.1, 0.4)
    f = ForwardMarketModel(CURVE, TIMES, vol, correlation=rho)
    g = LiborMarketModel(CURVE, TIMES, vol, correlation=rho)
    for i in (0, 6, 18):
        print(
            f"   {TIMES[i]:4.1f}Y x 6M ATM: forward-looking {g.caplet(i, g.forwards[i]) * 1e4:6.2f} bp,"
            f" in arrears {f.caplet(i, f.forwards[i]) * 1e4:6.2f} bp"
        )
    paths = fmm_paths(f, 20_000, 2, seed=2)
    i = 6
    pv, se = paths.price(f.tau[i] * np.maximum(paths.fixings[:, i] - f.forwards[i], 0.0), i + 1)
    print(f"   4Y x 6M in arrears by simulation: {pv * 1e4:.2f} +- {se * 1e4:.2f} bp")
    print(f"   within 3 SE: {abs(pv - f.caplet(i, f.forwards[i])) < 3 * se}")


def sabr_lmm() -> None:
    print("3. SABR-LMM caplet smile on the 3Y forward")
    rho = rebonato_correlation(FIX, 0.1, 0.4)
    m = SABRLMM(CURVE, TIMES, np.full(20, 0.035), beta=0.5, rho=-0.3, nu=0.4, correlation=rho)
    paths = sabr_lmm_paths(m, 40_000, 8, seed=3)
    i = 4
    F, T, df = m.forwards[i], TIMES[i], m.discounts[i + 1] * m.tau[i]
    ok = True
    for K in (0.02, 0.025, F, 0.04, 0.045):
        pv, se = paths.price(m.tau[i] * np.maximum(paths.fixings[:, i] - K, 0.0), i + 1)
        mc = implied_black_volatility(pv, F, K, T, df)
        hagan = implied_black_volatility(m.caplet(i, K), F, K, T, df)
        ok &= abs(pv - m.caplet(i, K)) < 3 * se + 0.01 * pv
        print(f"   K {K * 100:.2f}%: Hagan {hagan * 100:.2f}%, MC {mc * 100:.2f}%")
    print(f"   prices within 3 SE + 1%: {ok}")


if __name__ == "__main__":
    lmm()
    fmm()
    sabr_lmm()
