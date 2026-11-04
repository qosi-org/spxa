"""
Short-rate models with spxa.rates.

Calibrates Hull-White to co-terminal swaptions and checks the tree and PDE
on an American bond option. Then fits G2++ to a swaption grid quoted in
normal vols, compares the smiles of Hull-White and CIR++, and prices a
bond option on a Black-Karasinski tree and by simulation on its drift.

Run with: python examples/rates_short_rate_example.py
"""

import numpy as np

from spxa.market import FlatCurve
from spxa.rates import (
    BlackKarasinski,
    CIRPlusPlus,
    HullWhite,
    HullWhitePDE,
    HullWhiteTree,
    black_karasinski_paths,
    calibrate_g2pp,
    calibrate_hull_white,
    market_prices,
    model_implied_vols,
    swaption,
)

REF = "2024-03-15"
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)


def hull_white() -> None:
    print("1. Hull-White on co-terminal swaptions (10Y final maturity)")
    expiries = ["1Y", "2Y", "3Y", "5Y", "7Y"]
    tenors = ["9Y", "8Y", "7Y", "5Y", "3Y"]
    insts = [swaption(REF, e, t, 0.034, notional=1e4) for e, t in zip(expiries, tenors)]
    vols = [0.0095, 0.0092, 0.0089, 0.0085, 0.0081]  # normal vols
    tgt = market_prices(insts, vols, DISC, PROJ, "normal")
    cal = calibrate_hull_white(DISC, insts, tgt, PROJ, a=0.03, sigma_times=[1, 2, 3, 5])
    print("   sigma levels:", " ".join(f"{s:.5f}" for s in cal.model.sigma))
    print(f"   swaptions reprice (< 1e-10): {cal.max_relative_error < 1e-10}")
    hw = HullWhite(DISC, 0.05, 0.01)
    K = float(DISC.discount(6.0))
    tree = HullWhiteTree(hw, 6.0, 240).zcb_option(2.0, 6.0, K, "put", american=True)
    pde = HullWhitePDE(hw, 301, 400).zcb_option(2.0, 6.0, K, "put", american=True)
    print(f"   American 2Y put on the 6Y bond: tree {tree:.6f}, PDE {pde:.6f}")
    print(f"   tree (240 steps) and PDE agree to 1%: {abs(tree / pde - 1) < 1e-2}")


def g2pp() -> None:
    print("2. G2++ on a 3x3 swaption grid")
    grid = [(e, t) for e in ("1Y", "3Y", "7Y") for t in ("2Y", "5Y", "15Y")]
    # normal vols in bp, rounded to 0.1 bp
    vols = np.array([62.7, 60.5, 55.5, 62.2, 61.2, 55.2, 61.9, 60.1, 52.8]) * 1e-4
    insts = [swaption(REF, e, t, 0.034, notional=1e4) for e, t in grid]
    tgt = market_prices(insts, vols, DISC, PROJ, "normal")
    cal = calibrate_g2pp(DISC, insts, tgt, PROJ)
    m = cal.model
    print(f"   a {m.a:.4f}, sigma {m.sigma:.5f}, b {m.b:.4f}, eta {m.eta:.5f}, rho {m.rho:.3f}")
    err = (model_implied_vols(m, insts, PROJ, "normal") - vols) * 1e4
    print(f"   max vol error {np.abs(err).max():.3f} bp (quotes rounded to 0.1 bp)")


def smiles() -> None:
    print("3. Smiles: 2Y x 5Y normal vols at ATM -100/0/+100 bp")
    atm = swaption(REF, "2Y", "5Y", 0.03).forward_rate(DISC, PROJ)
    insts = [swaption(REF, "2Y", "5Y", atm + d, notional=1e4) for d in (-0.01, 0.0, 0.01)]
    for name, model in (
        ("Hull-White", HullWhite(DISC, 0.05, 0.01)),
        ("CIR++", CIRPlusPlus(DISC, 0.3, 0.025, 0.08, 0.02)),
    ):
        v = model_implied_vols(model, insts, PROJ, "normal") * 1e4
        print(f"   {name:10s}" + "".join(f" {x:6.2f}" for x in v))


def black_karasinski() -> None:
    print("4. Black-Karasinski: tree against simulation")
    bk = BlackKarasinski(DISC, 0.1, 0.25)
    tree = bk.tree(6.0, 120)
    paths = black_karasinski_paths(tree, 100_000, seed=3)
    F = float(DISC.discount(6.0) / DISC.discount(2.0))
    pv, se = paths.price(np.maximum(paths.zero_bond(40, 6.0) - 0.97 * F, 0.0), 40)
    tv = tree.zcb_option(2.0, 6.0, 0.97 * F)
    print(f"   2Y call on the 6Y bond: tree {tv:.6f}, MC {pv:.6f} +/- {se:.6f}")
    print(f"   within 4 standard errors: {abs(pv - tv) < 4 * se}")
    print(f"   all tree rates positive: {all(np.all(tree.rates(m) > 0) for m in range(120))}")


if __name__ == "__main__":
    hull_white()
    g2pp()
    smiles()
    black_karasinski()
