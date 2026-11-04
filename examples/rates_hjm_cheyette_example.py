"""
HJM and Cheyette models with spxa.rates.

Simulates a two-factor Gaussian HJM model with a humped volatility and
checks the simulated curve and a bond option against closed forms. Then
calibrates a Cheyette model to a swaption smile quoted in normal vols,
reprices the smile on the two-dimensional PDE, and values an American
bond put.

Run with: python examples/rates_hjm_cheyette_example.py
"""

import numpy as np

from spxa.market import FlatCurve
from spxa.market.black import black_price
from spxa.rates import (
    CheyettePDE,
    GaussianHJM,
    calibrate_cheyette,
    exponential_vol,
    hjm_paths,
    hump_vol,
    market_prices,
    swaption,
)

REF = "2024-03-15"
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)


def hjm() -> None:
    print("1. Two-factor HJM with a humped volatility")
    model = GaussianHJM(DISC, [hump_vol(0.004, 0.006, 0.4), exponential_vol(0.006, 0.03)])
    grid = np.linspace(0.0, 10.0, 81)
    paths = hjm_paths(model, grid, 100_000, record=[2.0, 5.0], keep=[2.0, 5.0, 7.0, 10.0], seed=7)
    worst = 0.0
    for t in (2.0, 5.0):
        for T in (t, 7.0, 10.0):
            z = paths.deflated(t, T)
            worst = max(worst, abs(z.mean() - DISC.discount(T)) / (z.std(ddof=1) / np.sqrt(z.size)))
    print(f"   deflated bonds reprice the curve: worst error {worst:.2f} standard errors")
    T, S = 2.0, 7.0
    K = float(DISC.discount(S) / DISC.discount(T))
    pv, se = paths.price(np.maximum(paths.zero_bond(T, S) - K, 0.0), T)
    left = sum(
        float(np.sum((model.Sigma(t, S) - model.Sigma(t, T)) ** 2)) * 0.125 for t in grid[grid < T]
    )
    pT = float(DISC.discount(T))
    disc_ref = black_price(float(DISC.discount(S)) / pT, K, 1.0, np.sqrt(left), pT, "call")
    print(
        f"   2Y call on the 7Y bond: MC {pv:.6f} +/- {se:.6f}, closed form {model.zcb_option(T, S, K):.6f}"
    )
    print(f"   MC agrees with Black at the scheme's variance: {abs(pv - disc_ref) < 4 * se}")


def cheyette() -> None:
    print("2. Cheyette calibrated to a 3x3 swaption smile")
    pairs = (("1Y", "9Y"), ("3Y", "7Y"), ("5Y", "5Y"))
    strikes = (0.025, 0.034, 0.045)
    insts = [
        swaption(REF, e, t, k, notional=1e4, payer=k >= 0.034) for e, t in pairs for k in strikes
    ]
    vols = np.array([76.0, 89.0, 102.0, 72.0, 84.0, 96.0, 68.0, 79.0, 90.0]) * 1e-4
    tgt = market_prices(insts, vols, DISC, PROJ, "normal")
    cal = calibrate_cheyette(DISC, insts, tgt, PROJ, kappa=0.03)
    m = cal.model
    print("   sigma levels:", " ".join(f"{s:.5f}" for s in m.sigma), f" beta {m.beta:.4f}")
    approx = np.array([s.implied_vol(p, DISC, PROJ, "normal") for s, p in zip(insts, cal.prices)])
    print(f"   formula fit: max vol error {np.abs(approx - vols).max() * 1e4:.2f} bp")
    pde = CheyettePDE(m, 161, 41, 100)
    on_pde = np.array(
        [s.implied_vol(pde.swaption(s, PROJ), DISC, PROJ, "normal") for s in insts[:3]]
    )
    print("   1Y x 9Y smile on the PDE:", " ".join(f"{v * 1e4:.1f}" for v in on_pde), "bp")
    print(f"   PDE within 1.5 bp of the quotes: {np.abs(on_pde - vols[:3]).max() < 1.5e-4}")
    T, S = 2.0, 6.0
    K = float(DISC.discount(S))

    def put(t):
        return lambda X, Y: np.maximum(K - m.discount_bond(t, S, X, Y), 0.0)

    pde2 = CheyettePDE(m, 161, 41, 200)
    eu = pde2.solve(put(T), T)
    am = pde2.solve(put(T), T, exercise={t: put(t) for t in np.linspace(0, T, 201)[1:-1]})
    print(f"   2Y put on the 6Y bond struck at P(0,6): European {eu:.6f}, American {am:.6f}")


if __name__ == "__main__":
    hjm()
    cheyette()
