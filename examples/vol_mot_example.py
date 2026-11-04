"""
Model-free bounds by martingale optimal transport with spxa.vol.

1. Marginals at 0.5y and 1y from an SSVI surface; forward-start call bounds
   in money for three strikes.
2. The superhedge as trades: call strips on both dates plus a forward
   position entered at 0.5y.
3. Entropic (Sinkhorn) transport converging to the LP bound.
4. A pathwise backtest of the superhedge on Heston paths.
5. A three-date Asian call.

Run with: python examples/vol_mot_example.py
"""

import numpy as np

from spxa.market import EquityForward, black_price
from spxa.vol import (
    SSVI,
    Heston,
    PowerLawPhi,
    DiscreteMarginal,
    VolSurface,
    forward_start_call,
    forward_start_straddle,
    marginal_from_calls,
    mot_bounds,
    mot_multi,
    mot_sinkhorn,
    mot_solve,
    robust_bounds,
    slv_paths,
    static_replication,
    superhedge_pnl,
    surface_marginals,
)

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SURFACE = VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), FWD)
GRID = np.linspace(0.02, 3.02, 101)


def main() -> None:
    T1, T2 = 0.5, 1.0
    F1, F2, df2 = FWD.forward(T1), FWD.forward(T2), FWD.discount(T2)
    print("1. Forward-start calls (S_T2 - k S_T1)^+ on the SSVI surface, spot 100")
    for k in (0.9, 1.0, 1.1):
        lo, hi, _, _ = robust_bounds(
            SURFACE, T1, T2, forward_start_call(k, F1 / F2), GRID, scale=F2 * df2
        )
        print(f"  k={k}: [{lo:.4f}, {hi:.4f}]")

    mu, nu = surface_marginals(SURFACE, [T1, T2], GRID)
    sup = mot_solve(mu, nu, forward_start_straddle(), "max")
    phi, psi = static_replication(mu.points, sup.phi), static_replication(nu.points, sup.psi)
    print("\n2. Superhedge of the forward-start straddle |Y - X|")
    print(f"  upper bound {sup.value:.6f}; vanillas cost {phi.price(mu) + psi.price(nu):.6f}")
    print(
        f"  {np.count_nonzero(np.abs(phi.calls) > 1e-9)} calls on T1, {np.count_nonzero(np.abs(psi.calls) > 1e-9)} on T2"
    )
    print(
        f"  forward position h(X) at X = 0.8, 1.0, 1.2: {np.interp([0.8, 1.0, 1.2], mu.points, sup.h).round(3)}"
    )

    print("\n3. Sinkhorn, flat 20% smiles at 0.5y and 1y")
    k = np.linspace(0.2, 3.0, 57)
    c1, c2 = black_price(1.0, k, 0.5, 0.2), black_price(1.0, k, 1.0, 0.2)
    m1 = marginal_from_calls(k, c1)
    m2 = marginal_from_calls(k, c2, floor=c1)
    lo, hi = mot_bounds(m1, m2, forward_start_straddle())
    print(f"  LP bounds [{lo:.4f}, {hi:.4f}]")
    for eps in (1e-2, 3e-3):
        r = mot_sinkhorn(m1, m2, forward_start_straddle(), eps=eps)
        print(
            f"  eps={eps:g}: {r.value:.6f} (gap {hi - r.value:.1e}, martingale error {r.martingale_error:.0e})"
        )

    print("\n4. Backtest on Heston paths (forward-start ATM call)")
    heston, grid = Heston(0.04, 1.5, 0.04, 0.5, -0.7), np.linspace(0.05, 3.05, 121)
    hm1 = marginal_from_calls(grid, np.asarray(heston.price(grid, 0.5), float))
    hm2 = marginal_from_calls(
        grid, np.asarray(heston.price(grid, 1.0), float), floor=hm1.call(grid)
    )
    pay = forward_start_call(1.0)
    hs = mot_solve(hm1, hm2, pay, "max")
    paths = slv_paths(
        heston, lambda y, t: np.ones_like(y), 1.0, 200, 50_000, record=(0.5, 1.0), seed=3
    )
    X, Y = np.exp(paths.at(0.5)[0]), np.exp(paths.at(1.0)[0])
    pnl = superhedge_pnl(hs, pay, X, Y)
    print(f"  upper bound {hs.value:.4f}, Heston price {pay(X, Y).mean():.4f}")
    print(f"  hedge P&L: mean {pnl.mean():.4f}, worst {pnl.min():.4f}")

    print("\n5. Three-date Asian call, flat 20% smiles at 0.5, 1, 1.5y")
    kk = np.linspace(0.3, 2.3, 21)
    ms: list[DiscreteMarginal] = []
    for T in (0.5, 1.0, 1.5):
        ms.append(
            marginal_from_calls(
                kk, black_price(1.0, kk, T, 0.2), floor=ms[-1].call(kk) if ms else None
            )
        )

    def asian(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.maximum((a + b + c) / 3 - 1.0, 0.0)

    print(f"  [{mot_multi(ms, asian, 'min')[0]:.4f}, {mot_multi(ms, asian, 'max')[0]:.4f}]")


if __name__ == "__main__":
    main()
