"""
Variance swaps and model risk with spxa.vol.

1. Variance and gamma swap strikes, the forward variance curve and the VIX
   from an SSVI surface.
2. The exact Heston volatility swap and its convexity gap.
3. Three SLV models (mixing fractions 0, 0.5, 1) calibrated to the same
   surface: model risk of vanillas, forward-start calls, variance and
   volatility swaps, set against the model-free MOT bounds.

Run with: python examples/vol_model_risk_example.py
"""

import numpy as np

from spxa.market import EquityForward
from spxa.vol import (
    SSVI,
    ForwardVarianceCurve,
    Heston,
    ModelRisk,
    PowerLawPhi,
    VolSurface,
    as_local_vol,
    calibrate_slv,
    forward_start_call,
    gamma_swap_strike,
    heston_volatility_swap,
    realised_variance,
    robust_bounds,
    smile_rmse,
    variance_swap_curve,
    vix_index,
)

SURFACE = VolSurface(
    SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, 0.03, dividend_yield=0.01),
)
HESTON = Heston(0.04, 1.5, 0.04, 0.5, -0.7)


def main() -> None:
    print("1. Variance swap term structure on the SSVI surface")
    expiries = np.array([30 / 365, 0.25, 0.5, 1.0, 2.0])
    strikes = variance_swap_curve(SURFACE, expiries)
    curve = ForwardVarianceCurve(expiries, strikes)
    print("   T      K_var   vol strike  gamma swap  ATM var   fwd var")
    for T, K, xi in zip(expiries, strikes, curve.levels):
        atm = float(SURFACE.total_variance(np.zeros(1), T)[0]) / T
        print(
            f"   {T:5.3f}  {K:.4f}  {np.sqrt(K):.4f}      {gamma_swap_strike(SURFACE, T):.4f}      {atm:.4f}    {xi:.4f}"
        )
    print(
        f"   VIX index {vix_index(SURFACE):.2f}; 1y-2y forward variance swap {curve.forward_variance_swap(1.0, 2.0):.4f}"
    )

    print(
        "\n2. Heston volatility swap (v0 = 0.04, kappa = 1.5, theta = 0.06, xi = 0.5, rho = -0.7)"
    )
    h = Heston(0.04, 1.5, 0.06, 0.5, -0.7)
    for T in (0.25, 1.0, 5.0):
        vs, sq = heston_volatility_swap(h, T), np.sqrt(h.expected_total_variance(T) / T)
        print(f"   T={T:<4}  vol swap {vs:.4f}  sqrt(K_var) {sq:.4f}  convexity {sq - vs:.4f}")

    print("\n3. Model risk across SLV mixing fractions")
    lv = as_local_vol(SURFACE, 1.0)
    k_smile = np.array([-0.3, -0.15, 0.0, 0.1, 0.2])
    days = np.linspace(0.0, 1.0, 253)
    fwd, rv, errors = {}, {}, {}
    for eta in (0.0, 0.5, 1.0):
        m = calibrate_slv(lv, HESTON, eta, 1.0, n_time=50, record=(0.5, 1.0))
        name = f"eta={eta}"
        model = np.concatenate([m.smile(k_smile, T) for T in (0.5, 1.0)])
        market = np.concatenate([SURFACE.implied_vol_k(k_smile, T) for T in (0.5, 1.0)])
        errors[name] = smile_rmse(model, market)
        p = m.paths(1.0, 252, 20_000, record=tuple(days[1:]), seed=9)
        logs = np.column_stack([np.zeros(20_000)] + [p.at(d)[0] for d in days[1:]])
        fwd[name] = (np.exp(p.at(0.5)[0]), np.exp(p.at(1.0)[0]))
        rv[name] = (realised_variance(logs, days),)
    claims = {
        "1y ATM call": ModelRisk.from_samples(fwd, lambda x, y: np.maximum(y - 1.0, 0.0), errors),
        "fwd-start ATM": ModelRisk.from_samples(fwd, forward_start_call(1.0), errors),
        "variance swap": ModelRisk.from_samples(rv, lambda r: r, errors),
        "vol swap": ModelRisk.from_samples(rv, np.sqrt, errors),
    }
    print("   smile RMSE (bp): " + ", ".join(f"{n} {1e4 * e:.1f}" for n, e in errors.items()))
    print("   claim            lower    upper    width    significant")
    for name, mr in claims.items():
        print(
            f"   {name:15}  {mr.lower():.4f}   {mr.upper():.4f}   {mr.width():.4f}   {bool(mr.significant())}"
        )
    lo, hi, _, _ = robust_bounds(
        SURFACE, 0.5, 1.0, forward_start_call(1.0), np.linspace(0.02, 3.02, 101)
    )
    fs = claims["fwd-start ATM"]
    print(
        f"   MOT bounds for the forward start [{lo:.4f}, {hi:.4f}]: family inside {fs.within(lo, hi)}, share {fs.share(lo, hi):.0%}"
    )


if __name__ == "__main__":
    main()
