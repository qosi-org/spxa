"""
Two-factor PDEs with ADI time stepping in spxa.pde.

Prices the in 't Hout-Foulon Heston case 1 with each of the four ADI schemes and checks the
results against Lewis's Fourier formula. Also prices an American put under Heston and an exchange
option against Margrabe's closed form.

Run with: python examples/pde_heston_adi_example.py
"""

import time

from spxa.pde import HestonPDE, TwoAssetBlackScholesPDE
from spxa.pde.reference import heston_cf, lewis_call_price, margrabe_price

print("=" * 72)
print("spxa.pde — Heston and two-asset ADI")
print("=" * 72)

# 1. European call, in 't Hout & Foulon (2010) case 1
r, kappa, theta, sigma, rho = 0.025, 1.5, 0.04, 0.3, -0.9
K, T, v0 = 100.0, 1.0, 0.04
heston = HestonPDE(r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
print(f"\nFeller condition 2*kappa*theta >= sigma^2: {heston.feller}")
cf = heston_cf(T, v0, kappa, theta, sigma, rho)
print("\n1. European call at S = 90, 100, 110 (101 x 51 nodes, 50 steps)")
for scheme in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"):
    t0 = time.perf_counter()
    errors = []
    for S in (90.0, 100.0, 110.0):
        res = heston.european(K, T, S, v0, n_s=101, n_v=51, n_time=50, scheme=scheme)
        errors.append(res.price - lewis_call_price(cf, S, K, T, r))
    dt = time.perf_counter() - t0
    print(f"   {scheme:<22} max |error| {max(map(abs, errors)):.2e}   ({dt:.1f}s)")

res = heston.european(K, T, 100.0, v0, n_s=201, n_v=101, n_time=100)
ref = lewis_call_price(cf, 100.0, K, T, r)
print(f"   HV on 201 x 101: {res.price:.6f} vs Lewis {ref:.6f}")
print(f"   delta {res.delta:.4f}  gamma {res.gamma:.5f}  vega {res.vega:.4f}")

# 2. American put (Ikonen-Toivanen splitting inside ADI)
amer = HestonPDE(r=0.1, kappa=5.0, theta=0.16, sigma=0.9, rho=0.1)
print("\n2. American put, K=10, T=0.25, v0=0.0625 (reference values from the literature)")
for S, ref in ((8.0, 2.0), (9.0, 1.10763), (10.0, 0.52004), (11.0, 0.21368), (12.0, 0.08205)):
    p = amer.american(10.0, 0.25, S, 0.0625, n_s=161, n_v=81, n_time=100).price
    print(f"   S = {S:>4}: {p:.5f}   reference {ref:.5f}")

# 3. Exchange option on two correlated assets
two = TwoAssetBlackScholesPDE(r=0.05, sigma1=0.3, sigma2=0.2, rho=0.5, q1=0.02, q2=0.01)
ex = two.exchange(1.0, 100.0, 95.0)
ref = margrabe_price(100.0, 95.0, 1.0, 0.3, 0.2, 0.5, 0.02, 0.01)
print(f"\n3. Exchange option max(S1 - S2, 0): PDE {ex.price:.6f}  Margrabe {ref:.6f}")
