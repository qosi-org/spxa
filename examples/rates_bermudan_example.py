"""
Bermudan swaptions with spxa.rates.

Calibrates Hull-White to the co-terminal Europeans of a 7Y-non-call-2
payer Bermudan, prices it on the PDE, and brackets it by least-squares
Monte Carlo and the Andersen-Broadie dual bound. Then compares the
trinomial tree with the PDE, and the effect of a rates skew in Cheyette.

Run with: python examples/rates_bermudan_example.py
"""

import numpy as np

from spxa.market import FlatCurve
from spxa.rates import (
    Cheyette,
    CheyettePDE,
    HullWhitePDE,
    HullWhiteTree,
    andersen_broadie,
    bermudan_from_times,
    bermudan_pde,
    bermudan_swaption,
    bermudan_tree,
    calibrate_hull_white,
    cheyette_exercise_problem,
    coterminal_swaptions,
    hull_white_exercise_problem,
    lsm,
    market_prices,
    model_price,
    swaption,
)

REF = "2024-03-15"
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)
SWAP = swaption(REF, "1Y", "6Y", 0.035, notional=1e4).swap


def hull_white() -> None:
    print("1. Hull-White calibrated to the co-terminal Europeans")
    ber = bermudan_swaption(SWAP, DISC, PROJ, lockout=1)
    insts = coterminal_swaptions(SWAP, DISC, lockout=1)
    vols = np.array([92.0, 89.0, 86.0, 83.0, 80.0]) * 1e-4
    targets = market_prices(insts, vols, DISC, PROJ, "normal")
    cal = calibrate_hull_white(DISC, insts, targets, PROJ, a=0.03)
    hw = cal.model
    err = max(abs(model_price(hw, s, PROJ) - p) for s, p in zip(insts, targets))
    print("   exercise times:", " ".join(f"{t:.3f}" for t in ber.exercise_times))
    print("   sigma levels:  ", " ".join(f"{s:.5f}" for s in hw.sigma))
    print(f"   Europeans reprice (< 1e-8): {err < 1e-8}")
    pde = bermudan_pde(HullWhitePDE(hw, 401, 400), ber)
    print(
        f"   Bermudan on the PDE {pde:.2f}; largest European {targets.max():.2f}, sum {targets.sum():.2f}"
    )
    res = lsm(hull_white_exercise_problem(hw, ber), 200_000, seed=1, n_train=50_000)
    dual = andersen_broadie(res, 1000, 500, seed=2)
    print(
        f"   LSM lower bound {res.price:.2f} +- {res.stderr:.2f}, dual upper bound {dual.upper:.2f}"
    )
    print(f"   duality gap {dual.gap:.3f} ({dual.gap / pde * 1e4:.1f} bp of the price)")
    print(
        f"   PDE inside the bounds (3 SE): {dual.lower - 3 * res.stderr < pde < dual.upper + 3 * dual.stderr}"
    )


def tree() -> None:
    print("2. Tree against PDE (annual receiver on year fractions)")
    from spxa.rates import HullWhite

    hw = HullWhite(DISC, 0.03, 0.009)
    ber = bermudan_from_times([2.0, 3.0, 4.0, 5.0, 6.0], 7.0, 0.03, notional=1e4, payer=False)
    pde = bermudan_pde(HullWhitePDE(hw, 401, 400), ber)
    for n in (175, 350, 700, 1400):
        print(f"   tree with {n:4d} steps: {bermudan_tree(HullWhiteTree(hw, 7.0, n), ber):.3f}")
    print(f"   PDE: {pde:.3f}")


def cheyette() -> None:
    print("3. Rates skew: Cheyette with beta = 0 and 0.3")
    ber = bermudan_swaption(SWAP, DISC, PROJ, lockout=1)
    for beta in (0.0, 0.3):
        m = Cheyette(DISC, 0.03, 0.009, beta=beta)
        pde = bermudan_pde(CheyettePDE(m, 161, 41, 200), ber)
        eu = max(bermudan_pde(CheyettePDE(m, 161, 41, 200), ber.restrict([k])) for k in range(5))
        res = lsm(cheyette_exercise_problem(m, ber), 200_000, seed=3, n_train=50_000)
        dual = andersen_broadie(res, 500, 300, seed=4)
        print(
            f"   beta {beta}: PDE {pde:.2f} (switch value {pde - eu:.2f}),"
            f" Monte Carlo [{res.price:.2f}, {dual.upper:.2f}] +- {dual.stderr:.2f}"
        )


if __name__ == "__main__":
    hull_white()
    tree()
    cheyette()
