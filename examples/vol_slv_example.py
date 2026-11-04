"""
Stochastic local volatility with spxa.vol.

Calibrates an SLV model (a Heston variance with a leverage function) to an
SSVI surface for three mixing fractions, checks that each reprices the
smile, and shows how barrier prices move with the mixing fraction while
vanillas stay fixed. At eta = 0.5 the barrier price is cross-checked
between the backward PDE and Monte Carlo, and the particle calibration is
compared with the Fokker-Planck one.

Run with: python examples/vol_slv_example.py
"""

import numpy as np

from spxa.market import EquityForward
from spxa.pde import BlackScholesPDE
from spxa.vol import (
    SSVI,
    Heston,
    PowerLawPhi,
    VolSurface,
    as_local_vol,
    calibrate_leverage_particles,
    calibrate_slv,
)

R, Q = 0.03, 0.01
SURFACE = VolSurface(
    SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, R, dividend_yield=Q),
)
HESTON = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)


def main() -> None:
    lv = as_local_vol(SURFACE, 1.0)
    k = np.array([-0.3, -0.15, 0.0, 0.1, 0.2])
    target = np.asarray(SURFACE.implied_vol_k(k, 1.0), float)
    lv_do = float(
        BlackScholesPDE(R, lv, Q).barrier(100.0, 1.0, 100.0, 85.0, "down-and-out", "call")
    )
    print("1. One-year barriers (spot = strike = 100), local vol DO(85):", f"{lv_do:.4f}")
    print("   eta   smile err (bp)   ATM leverage   DO(85)    UO(130)")
    models = {}
    for eta in (0.0, 0.5, 1.0):
        slv = calibrate_slv(lv, HESTON, eta, 1.0, n_time=50, record=(1.0,))
        models[eta] = slv
        err = np.abs(slv.smile(k, 1.0) - target).max() * 1e4
        pde = slv.pde(R, Q, n_x=81, n_v=31, n_time=40)
        do = pde.barrier(100.0, 1.0, "call", 100.0, lower=85.0)
        uo = pde.barrier(100.0, 1.0, "call", 100.0, upper=130.0)
        lev = slv.leverage(0.0, 0.99)
        print(f"   {eta:.1f}   {err:14.2f}   {lev:12.4f}   {do:7.4f}   {uo:7.4f}")

    print("2. Cross-checks at eta = 0.5")
    slv = models[0.5]
    pde_uo = slv.pde(R, Q, n_x=121, n_v=41, n_time=60).barrier(
        100.0, 1.0, "call", 100.0, upper=130.0
    )

    def barrier(t: float) -> float:
        return float(np.log(130.0 / (100.0 * np.exp((R - Q) * t))))

    paths = slv.paths(1.0, 200, 200_000, seed=11, upper=barrier)
    F, df = 100.0 * np.exp(R - Q), np.exp(-R)
    pv, se = paths.price(lambda S: np.maximum(F * S - 100.0, 0.0), 1.0, knock_out=True)
    print(
        f"   UO(130): PDE {pde_uo:.4f} (121x41x60), Monte Carlo {df * pv:.4f} +/- {df * se:.4f} (200 steps)"
    )
    print(f"   PDE within 3 standard errors: {abs(pde_uo - df * pv) < 3 * df * se}")
    pc = calibrate_leverage_particles(lv, slv.model, 1.0, 100, 50_000, seed=5)
    ys = np.linspace(-0.3, 0.2, 6)
    gap = np.abs(pc.leverage(ys, 0.99) / slv.leverage(ys, 0.99) - 1).max()
    print(f"   particle vs Fokker-Planck leverage on y in [-0.3, 0.2]: max gap {gap:.2%}")


if __name__ == "__main__":
    main()
