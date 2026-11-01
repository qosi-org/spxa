"""
One-factor PDE pricing with spxa.pde.

Covers European, American and barrier options under Black-Scholes, checks each against a
closed form or a published value, and measures the convergence order of Crank-Nicolson.

Run with: python examples/pde_black_scholes_example.py
"""

from spxa.pde import BlackScholesPDE, convergence_study
from spxa.pde.reference import barrier_price, bs_greeks, bs_price

print("=" * 72)
print("spxa.pde — Black-Scholes finite differences")
print("=" * 72)

r, sigma, q = 0.05, 0.25, 0.02
K, T, S = 100.0, 1.0, 100.0
bs = BlackScholesPDE(r=r, sigma=sigma, q=q)

# 1. European call with Greeks taken from the grid
eu = bs.european(K, T, S, option="call", bump_greeks=True)
ref = bs_price(S, K, T, r, sigma, q, option="call")
greeks = bs_greeks(S, K, T, r, sigma, q, option="call")
print("\n1. European call")
print(f"   price  PDE {eu.price:.6f}   closed form {ref:.6f}   error {eu.price - ref:+.1e}")
for name in ("delta", "gamma", "vega"):
    print(f"   {name:<6} PDE {getattr(eu, name):.6f}   closed form {greeks[name]:.6f}")

# 2. American put: the Longstaff-Schwartz test case with four LCP solvers
print("\n2. American put (K=40, r=0.06, sigma=0.2, T=1, S=36; accurate value 4.48644)")
ls = BlackScholesPDE(r=0.06, sigma=0.2)
for method in ("penalty", "psor", "brennan_schwartz", "ikonen_toivanen"):
    res = ls.american(40.0, 1.0, 36.0, option="put", method=method)
    print(f"   {method:<17} {res.price:.6f}")
res = ls.american(40.0, 1.0, 36.0, option="put")
print(f"   early-exercise boundary at t=0: S* = {res.exercise_boundary.at(1.0):.4f}")

# 3. Down-and-out call: domain truncated at the barrier
H = 90.0
dao = bs.barrier(K, T, S, barrier=H, barrier_type="down-and-out", option="call")
ref = barrier_price(S, K, H, T, r, sigma, q, option="call", barrier_type="down-and-out")
print(f"\n3. Down-and-out call H={H}: PDE {dao.price:.6f}  Reiner-Rubinstein {ref:.6f}")


# 4. Convergence order of the European price (space and time refined together)
def solve(n: int) -> float:
    return bs.european(K, T, S, option="call", n_space=n, n_time=n).price


study = convergence_study(solve, [51, 101, 201, 401], reference=bs_price(S, K, T, r, sigma, q))
print("\n4. Convergence study (expected order 2)")
print(study)
