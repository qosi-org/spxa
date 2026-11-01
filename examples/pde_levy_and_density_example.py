"""
Levy PIDEs and Fokker-Planck densities in spxa.pde.

Turns zoo processes into pricing equations with Process.to_pde(). Prices European options under
Merton, Kou, variance gamma and CGMY and checks them against Fourier (Lewis) prices, then prices a
Merton American put. Finally, it propagates an Ornstein-Uhlenbeck density with the
mass-conserving Fokker-Planck solver.

Run with: python examples/pde_levy_and_density_example.py
"""

import numpy as np

from spxa.pde import FokkerPlanck1D, Grid1D
from spxa.pde.reference import levy_cf, lewis_call_price
from spxa.zoo.beyond import OrnsteinUhlenbeckProcess
from spxa.zoo.levy import CGMY, KouJumpDiffusion, MertonJumpDiffusion, VarianceGamma

print("=" * 72)
print("spxa.pde — Levy PIDEs and Fokker-Planck densities")
print("=" * 72)

r, q, K, T, S = 0.05, 0.0, 100.0, 0.5, 100.0
processes = {
    "Merton": MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45),
    "Kou": KouJumpDiffusion(drift=0.0, sigma=0.16, rate=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "VG": VarianceGamma(sigma=0.12, nu=0.2, theta=-0.14),
    "CGMY": CGMY(C=1.0, G=8.8, M=12.0, Y=0.5),
}

# 1. European calls: PIDE (implicit diffusion, explicit FFT jump integral) vs Lewis
print(f"\n1. European call K={K}, T={T}, S={S}")
for name, proc in processes.items():
    pide = proc.to_pde(r=r, q=q)
    pde_price = pide.european(K, T, S, option="call").price
    fourier = lewis_call_price(levy_cf(proc, T), S, K, T, r, q)
    print(
        f"   {name:<7} PIDE {pde_price:.6f}   Lewis {fourier:.6f}   error {pde_price - fourier:+.1e}"
    )

# 2. American put under Merton (reference 3.241207 for these parameters)
merton = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
amer = merton.to_pde(r=0.05).american(100.0, 0.25, 100.0, "put", n_space=1601, n_time=400)
print(f"\n2. Merton American put K=S=100, T=0.25: {amer.price:.6f}  (reference 3.241207)")

# 3. Density of an OU process: the transition density is Gaussian
ou = OrnsteinUhlenbeckProcess(theta=2.0, mu=0.1, sigma=0.3, x0=0.5)
gen = ou.to_pde()
grid = Grid1D.uniform(-1.5, 2.0, 701)
fp = FokkerPlanck1D(gen.drift, gen.volatility, grid)
dens = fp.solve(fp.delta(ou.x0), T=1.0, n_time=400)
m = ou.mu + (ou.x0 - ou.mu) * np.exp(-ou.theta)
v = ou.sigma**2 * (1 - np.exp(-2 * ou.theta)) / (2 * ou.theta)
exact = np.exp(-((grid.nodes - m) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)
l1 = float(np.sum(dens.weights * np.abs(dens.pdf(grid.nodes) - exact)))
print("\n3. OU density at t = 1")
print(f"   mass {dens.mass():.15f}   mean {dens.mean():.6f} (exact {m:.6f})")
print(f"   variance {dens.variance():.6f} (exact {v:.6f})   L1 error {l1:.1e}")
