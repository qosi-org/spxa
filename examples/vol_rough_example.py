"""
Rough volatility with spxa.vol.

1. Rough Heston smiles. The short-dated ATM skew explodes like T^{H-1/2}.
2. Lifted Heston, a Markovian multi-factor approximation, converging to
   rough Heston.
3. Rough Heston calibration with H free versus classical Heston (H = 1/2).
4. Rough Bergomi by the hybrid scheme, with the conditional estimator.
5. VIX futures and smiles in rough Bergomi and Heston.

Run with: python examples/vol_rough_example.py
"""

import numpy as np

from spxa.vol import (
    Heston,
    HestonVIX,
    LiftedHeston,
    RBergomiVIX,
    RoughHeston,
    calibrate_rough_heston,
    rbergomi_paths,
)

ROUGH = RoughHeston(v0=0.04, theta=0.04, lam=0.3, nu=0.3, rho=-0.7, hurst=0.1)
K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])


def atm_skew(model: RoughHeston, T: float) -> float:
    e = 0.02 * np.sqrt(T)
    lo, hi = model.implied_vol(np.exp([-e, e]), T)
    return float((hi - lo) / (2 * e))


def main() -> None:
    print("1. Rough Heston (H = 0.1)")
    for T in (0.05, 0.25, 1.0):
        vols = ROUGH.implied_vol(K, T, n_steps=400)
        print(f"  T={T:<5} vols " + " ".join(f"{100 * x:6.2f}" for x in vols))
    taus = np.array([0.0025, 0.01, 0.04])
    skews = np.array([atm_skew(ROUGH, t) for t in taus])
    slope = np.polyfit(np.log(taus), np.log(-skews), 1)[0]
    print(f"  ATM skew {skews.round(3)} -> power {slope:.3f} (H - 1/2 = -0.4)")
    heston_skew = [atm_skew(RoughHeston(0.04, 0.04, 0.3, 0.3, -0.7, 0.5), t) for t in taus]
    print(f"  classical Heston skew stays bounded: {np.ptp(heston_skew) < 0.1 * np.ptp(skews)}")

    print("\n2. Lifted Heston, largest 1y gap to rough Heston (vol points)")
    rough = ROUGH.implied_vol(K, 1.0, n_steps=400)
    for n in (5, 20, 50):
        lifted = LiftedHeston(0.04, 0.04, 0.3, 0.3, -0.7, 0.1, n=n).implied_vol(K, 1.0, n_steps=400)
        print(f"  n={n:<3} {100 * np.abs(lifted - rough).max():.2f}")

    print("\n3. Calibration to rough smiles at 0.1, 0.5, 1.5y")
    truth = RoughHeston(0.03, 0.05, 0.3, 0.4, -0.65, 0.08)
    ts = [0.1, 0.5, 1.5]
    ks = [np.exp(np.linspace(-2, 2, 9) * 0.2 * np.sqrt(T)) for T in ts]
    vols = [truth.implied_vol(k, T, n_steps=400) for T, k in zip(ts, ks)]
    rough_fit = calibrate_rough_heston(ts, ks, vols, RoughHeston(0.04, 0.04, 0.3, 0.3, -0.5, 0.2))
    heston_fit = calibrate_rough_heston(
        ts,
        ks,
        vols,
        RoughHeston(0.04, 0.04, 0.3, 0.3, -0.5, 0.5),
        free=("v0", "theta", "nu", "rho"),
    )
    print(f"  H free: H = {rough_fit.model.hurst:.4f}, RMSE {100 * rough_fit.rmse:.4f} vol points")
    print(f"  H = 1/2: RMSE {100 * heston_fit.rmse:.2f} vol points")

    print("\n4. Rough Bergomi (H = 0.1, eta = 1.9, rho = -0.9), hybrid scheme, 100k paths")
    paths = rbergomi_paths(0.1, 1.9, -0.9, 0.04, 1.0, 100, 100_000, seed=2)
    mixed, se = paths.price(K)
    plain, se_plain = paths.price(K, conditional=False)
    print("  vols " + " ".join(f"{100 * x:6.2f}" for x in paths.implied_vol(K)))
    print(
        f"  conditional and plain agree within 3 SE: {bool(np.all(np.abs(mixed - plain) < 3 * se_plain))}"
    )
    print(f"  SE ratio plain/conditional: {(se_plain / se).round(2)}")
    print(
        f"  E[S_T] = {paths.spot.mean():.4f} +- {paths.spot.std() / np.sqrt(paths.spot.size):.4f}"
    )

    print("\n5. VIX at T = 0.5")
    rb = RBergomiVIX(0.1, 1.9, lambda t: 0.04 + 0.01 * np.asarray(t))
    fut, fut_se = rb.futures(0.5, 400_000, seed=1)
    print(
        f"  rough Bergomi: sqrt E[VIX^2] {100 * np.sqrt(rb.expected_vix_squared(0.5)):.2f}, futures {100 * fut:.2f}"
    )
    print(
        "  rough Bergomi VIX smile: "
        + " ".join(f"{100 * x:.1f}" for x in rb.implied_vols(0.5, [0.14, 0.2, 0.36], seed=1))
    )
    hv = HestonVIX(Heston(0.04, 1.5, 0.04, 0.5, -0.7))
    print(f"  Heston: futures {100 * hv.futures(0.5):.2f}")
    print(
        "  Heston VIX smile:        "
        + " ".join(f"{100 * x:.1f}" for x in hv.implied_vols(0.5, [0.12, 0.2, 0.3]))
    )


if __name__ == "__main__":
    main()
