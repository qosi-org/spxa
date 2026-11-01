# spxa: Stochastic Process Algebra & Computational Graph Engine

<p align="center">
  <a href="https://github.com/Nityahapani/spxa/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-passing-brightgreen.svg?style=flat-square" alt="CI Status"></a>
  <a href="https://pypi.org/project/spxa/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?style=flat-square" alt="Python Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checking-mypy%20strict-blue.svg?style=flat-square" alt="Mypy"></a>
  <a href="#exactness-guarantees--mathematical-integrity"><img src="https://img.shields.io/badge/exactness-algebraic%20lattice-orange.svg?style=flat-square" alt="Exactness"></a>
  <a href="https://github.com/Nityahapani/spxa/blob/main/PROCESS_REGISTRY.md"><img src="https://img.shields.io/badge/zoo-198%2B%20processes-purple.svg?style=flat-square" alt="Process Zoo"></a>
</p>

<p align="center">
  <strong>Stochastic processes as first-class algebraic and computational objects.</strong><br>
  Exact Lévy–Khintchine arithmetic, DAG computational graph compilation, reverse-mode automatic differentiation, statistical information geometry, rough path signatures, fractional Volterra rough volatility, and exact model morphing.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Why spxa?](#why-spxa)
- [Core Architectural Pillars](#core-architectural-pillars)
- [Quickstart Examples](#quickstart-examples)
  - [1. Algebraic Process Composition & Exact Triplet Arithmetic](#1-algebraic-process-composition--exact-triplet-arithmetic)
  - [2. Computational Graph & Reverse-Mode Greeks (`GradientTape`)](#2-computational-graph--reverse-mode-greeks-gradienttape)
  - [3. Statistical Information Geometry & Rao Hyperbolic Distance](#3-statistical-information-geometry--rao-hyperbolic-distance)
  - [4. Rough Paths & Hyperbolic Goursat Signature Kernel](#4-rough-paths--hyperbolic-goursat-signature-kernel)
  - [5. Rough Volatility: rBergomi & Fractional Riccati Solver](#5-rough-volatility-rbergomi--fractional-riccati-solver)
  - [6. High-Frequency Market Making & Inventory Control](#6-high-frequency-market-making--inventory-control)
  - [7. Model Morphing](#7-model-morphing)
  - [8. PDE & PIDE Solvers](#8-pde--pide-solvers)
- [Computational Graph Architecture & Duality Engine](#computational-graph-architecture--duality-engine)
- [Exactness Guarantees & Mathematical Integrity](#exactness-guarantees--mathematical-integrity)
- [Subpackage Ecosystem](#subpackage-ecosystem)
- [Performance & Benchmarks](#performance--benchmarks)
- [Installation](#installation)
- [Development & Testing](#development--testing)
- [Documentation & Resources](#documentation--resources)
- [Citation](#citation)
- [License](#license)

---

## Overview

**spxa** is an open-source Python library for quantitative researchers, financial engineers, and mathematical scientists. It rethinks stochastic modeling by elevating continuous-time stochastic processes into **first-class algebraic entities** and unifying analytical pricing, numerical quadrature, pathwise Monte Carlo simulation, and exact automatic differentiation inside a shared **computational graph**.

In traditional modeling libraries, stochastic processes are merely simulation loops or black-box function wrappers. **spxa** exploits the deep mathematical bijection established by the **Lévy–Khintchine formula**: independent increments and infinite divisibility allow linear combinations ($X + cY$), Bochner subordinations ($X \circ Y$), and scale transformations to be computed **in closed algebraic form** directly on characteristic triplets $(\gamma, \sigma^2, \nu)$, preserving symbolic cumulants and exact characteristic exponents.

When an operation crosses the boundary of the Lévy realm (e.g. state-dependent volatility, mean reversion, or memory kernels), `spxa` provides a **provably sound degradation policy**—transparently moving from exact triplet arithmetic to moment propagation or pathwise simulation without ever silently producing inaccurate claims.

---

## Why spxa?

| Feature | Legacy Quant Frameworks (e.g. QuantLib) | Deep Learning Frameworks (e.g. PyTorch) | **spxa** |
| :--- | :--- | :--- | :--- |
| **Object Model** | Monolithic C++ class hierarchies with pricing engines decoupled from process identity | Generic multidimensional tensor arrays without stochastic semantics | **First-class stochastic processes with algebraic operator overloading (`+`, `*`, `@`)** |
| **Lévy–Khintchine Arithmetic** | ❌ None (processes are not composable objects) | ❌ None | **✅ Exact closed-form triplet arithmetic & symbolic cumulant propagation** |
| **Execution Architecture** | Hardcoded pricing trees and fixed numerical engines | Static / Dynamic computational graphs optimized for neural net layers | **Unified Stochastic DAG: dynamically routes between Analytical, Fourier, Batch, and MC backends** |
| **Sensitivities & Greeks** | Manual bumping with finite-difference truncation error and numerical noise | General reverse-mode autograd on floating-point tensors | **`GradientTape` reverse-mode autograd tailored to stochastic operators (zero-truncation Delta, Gamma, Vega, Rho, Theta)** |
| **Information Geometry** | ❌ None | ❌ Limited to Euclidean gradient updates | **Riemannian parameter manifolds under the Fisher metric, geodesic shooting, and natural gradients** |
| **Rough Paths & Signatures** | ❌ None | ⚠️ Third-party external wrappers | **Native tensor signatures, shuffle algebra, and 2D hyperbolic Goursat PDE kernel solver** |
| **Rough Volatility** | ❌ Markovian models only | ❌ None | **Fractional Brownian motion, Riemann-Liouville kernels, rBergomi, and fractional Riccati ABM solvers** |
| **Model Interpolation** | ❌ None (switch models by hand) | ❌ None | **Exact Lévy–Khintchine homotopies plus generator, Fisher–Rao and Wasserstein morphs, time- and state-dependent regime schedules, and model barycenters** |
| **Mathematical Guarantees** | Silent approximation | Silent truncation | **Strict `ExactnessLevel` lattice (`EXACT` $\to$ `MOMENT` $\to$ `SIMULATION`) with mathematical audit logs (`__story__`)** |

---

## Core Architectural Pillars

1. **Stochastic Process Algebra (`spxa.core`, `spxa.zoo`)**: Write `Z = X + 0.5 * Y` or `W = X @ Y`. Processes compose via operator overloading. Characteristic triplets compose algebraically, cumulants derive symbolically to arbitrary order, and human-readable mathematical proofs are generated on demand via `__story__()`.
2. **Computational Graph Engine (`spxa.graph`)**: A dedicated DAG representation of stochastic pricing pipelines. Compiles expressions, eliminates common subexpressions (CSE), folds constants, and routes execution dynamically to analytical formulas, Fourier quadrature (Carr–Madan, Lewis, COS), or vectorized Monte Carlo.
3. **Reverse-Mode Autodiff Tape (`spxa.graph.autodiff`)**: Machine-precision risk sensitivities and full Hessian matrices calculated in a single reverse pass without finite-difference step-size tuning. Includes differentiable smooth mollifiers for non-smooth payoffs.
4. **Statistical Information Geometry (`spxa.geom`, `spxa.graph.geom`)**: Treats parameter spaces as Riemannian manifolds endowed with the Fisher Information Metric (FIM). Evaluates Levi-Civita connections, Amari $\alpha$-connections, geodesic flows, hyperbolic Rao distances, and executes manifold-constrained natural gradient calibration.
5. **Rough Paths & Tensor Signatures (`spxa.signature`, `spxa.graph.signature`)**: Non-parametric path representation capturing the order of market events. Provides truncated tensor signatures, shuffle product identities, log-signature Lie algebra dimension compression, an infinite-order 2D hyperbolic Goursat PDE kernel solver, and linear regression option pricing.
6. **Rough Volatility & Volterra Calculus (`spxa.rough`, `spxa.graph.rough`)**: Non-Markovian, sub-diffusive stochastic volatility models driven by fractional Brownian motion with Hurst index $H \in (0, 1/2)$. Features Riemann–Liouville singular power kernels, Bennedsen–Lunde–Pakkanen hybrid convolutions, rough Bergomi (rBergomi) with power-law short-time skew blow-up, and fractional Adams–Bashforth–Moulton Riccati solvers.
7. **Model Morphing (`spxa.morph`, `spxa.graph.morph`)**: Continuous interpolation between any two processes along a chosen geometry. The triplet homotopy $\psi_\alpha = (1-\alpha)\psi_0 + \alpha\psi_1$ stays inside the Lévy class and keeps `EXACT` exactness. Generator, Fisher–Rao geodesic and Wasserstein (displacement) morphs cover diffusions and Gaussian marginals. Dynamic schedules (linear, cosine, smoothstep, sigmoidal, state-threshold) drive regime transitions inside a simulation, and `ModelBarycenter`, `ModelEnsemble` and `SurfaceMorphEngine` average models and volatility surfaces.
8. **PDE & PIDE Solvers (`spxa.pde`, `spxa.graph.pde`)**: Verified finite-difference pricing for the generators of the zoo. Sinh-stretched grids aligned to strikes and barriers, Crank–Nicolson with Rannacher start-up, and penalty, PSOR, Brennan–Schwartz and Ikonen–Toivanen solvers for early exercise. Covers Black–Scholes (American, Bermudan, barriers, discrete dividends), Dupire local volatility, Večeř Asian options, short-rate bond options, mass-conserving Fokker–Planck densities, exponential-Lévy PIDEs (Merton, Kou, VG, CGMY, NIG) with moment-matched small jumps, and Douglas, Craig–Sneyd, modified Craig–Sneyd and Hundsdorfer–Verwer ADI for Heston and two-asset options. Least-squares Monte Carlo is included as a cross-check. Each scheme's convergence order is tested against closed forms or published references.
9. **Comprehensive Quantitative Toolchain**:
   - **Stochastic Optimal Control (`spxa.control`)**: Merton portfolio problems, Avellaneda–Stoikov high-frequency market making, Almgren–Chriss optimal liquidation, and finite-difference Hamilton–Jacobi–Bellman (HJB) viscosity solvers.
   - **Econometric Calibration & Filtering (`spxa.calibration`)**: Volatility surface calibrators, maximum likelihood (MLE), GMM, Kalman and Particle state-space filters, Godambe sandwich standard errors, and surface arbitrage diagnostics.
   - **Copulas & Multi-Asset Dependence (`spxa.copula`)**: Archimedean (Clayton, Gumbel, Frank, Joe, AMH), Elliptic (Gaussian, Student-$t$), and Extreme Value copulas with tail dependence analysis.

---

## Quickstart Examples

### 1. Algebraic Process Composition & Exact Triplet Arithmetic

```python
import spxa
from spxa import VarianceGamma, GammaProcess

# Instantiate two independent Lévy processes
X = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
Y = GammaProcess(a=1.0, b=2.0)

# Exact algebraic linear combination and Bochner subordination (time-change)
Z = X + 0.5 * Y          # exact: Lévy–Khintchine triplets add
W = X @ Y                # exact: subordination via Laplace-Bernstein functions

# Inspect properties
print("Exactness Level:", Z.exactness)                  # ExactnessLevel.EXACT
print("Lévy Triplet:", Z.triplet)                       # LevyTriplet(b=..., sigma_sq=..., nu=...)
print("Symbolic Cumulants:", Z.cumulants(order=4))       # {1: ..., 2: ..., 3: ..., 4: ...}
print("Characteristic Function at u=1.0:", Z.char_func(u=1.0))

# Print the complete mathematical derivation trace
print(Z.__story__())
```

### 2. Computational Graph & Reverse-Mode Greeks (`GradientTape`)

Construct an end-to-end pricing DAG and evaluate exact Greeks in a single reverse-mode pass without numerical bump errors:

```python
from spxa.graph import build_black_scholes_pricing_graph, compute_graph_greeks

# Construct a Black-Scholes pricing DAG
graph = build_black_scholes_pricing_graph(
    s0=100.0,
    k=100.0,
    t=1.0,
    r=0.05,
    sigma=0.2,
    is_call=True,
)

# Compute full first-order and second-order Greeks via reverse-mode tape
greeks = compute_graph_greeks(graph)

print(f"Price: {greeks['price']:.4f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega:  {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Rho:   {greeks['rho']:.4f}")
```

### 3. Statistical Information Geometry & Rao Hyperbolic Distance

Analyze parameter manifolds equipped with the Fisher Information Metric and calculate invariant Riemannian distances:

```python
import numpy as np
from spxa.geom import FisherInformationMetric, AffineConnection, GeodesicFlow

# Exact 2x2 Fisher Information Metric for univariate Gaussian manifold
g = FisherInformationMetric.gaussian_fisher(mu=0.0, sigma=1.0)
print("Fisher Information Matrix:\n", g)

# Compute exact Riemannian Rao distance along the hyperbolic half-plane
conn = AffineConnection(
    metric_fn=lambda th: FisherInformationMetric.gaussian_fisher(th[0], th[1]),
    dim=2,
)
flow = GeodesicFlow(conn)
d_rao = flow.gaussian_exact_rao_distance((0.0, 1.0), (2.0, 3.0))  # (mu, sigma) pairs
print(f"Hyperbolic Rao Distance: {d_rao:.4f}")

# Numerical geodesic shooting agrees with the closed form
d_num = flow.rao_distance(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
print(f"Geodesic-shooting Rao Distance: {d_num:.4f}")
```

### 4. Rough Paths & Hyperbolic Goursat Signature Kernel

Compute path signatures and solve the 2D hyperbolic Goursat PDE for infinite-dimensional signature kernels:

```python
import numpy as np
from spxa.signature import PathSignature, SignatureKernel

# Discretize 2D trajectories
t = np.linspace(0, 1, 100)
path_a = np.column_stack([t, np.sin(2 * np.pi * t)])
path_b = np.column_stack([t, np.cos(2 * np.pi * t)])

# Truncated path signature at depth 3
sig_engine = PathSignature(depth=3)
sig = sig_engine.signature(path_a)

# Infinite-order signature kernel solved via 2D Goursat PDE:
# ∂²K/∂s∂t = K(s, t) ⟨X_s', Y_t'⟩, with K(s, 0) = K(0, t) = 1
kernel_value = SignatureKernel.evaluate(path_a, path_b)
print(f"Signature Kernel <S(A), S(B)>: {kernel_value:.4f}")
```

### 5. Rough Volatility: rBergomi & Fractional Riccati Solver

Simulate sub-diffusive rough volatility ($H < 1/2$) and reproduce the steep power-law ATM implied volatility skew observed in real option markets:

```python
from spxa.rough import RoughBergomi

# Initialize Rough Bergomi with Hurst index H = 0.07
rberg = RoughBergomi(
    hurst=0.07,
    eta=1.9,
    rho=-0.7,
    xi0=0.04,
    s0=100.0,
)

# Theoretical leading-order ATM skew blow-up: |∂σ/∂k| ~ τ^{H - 1/2}
tau = 0.05  # short maturity (approx 18 days)
short_skew = rberg.theoretical_short_time_skew(tau=tau)
print(f"Short-Maturity ATM Implied Volatility Skew: {short_skew:.4f}")

# Exact joint simulation of the Volterra driver, variance, and correlated spot
sim = rberg.simulate_paths(t_mat=1.0, n_steps=250, n_paths=1000, seed=42)
s_paths, v_paths = sim["spot_paths"], sim["variance_paths"]
print("E[v_T] ≈ xi0:", v_paths[:, -1].mean())
```

### 6. High-Frequency Market Making & Inventory Control

Compute optimal inventory-dependent bid/ask quotes and reservation prices via the Avellaneda–Stoikov model:

```python
from spxa.control import AvellanedaStoikov

# Instantiate Avellaneda-Stoikov market maker
mm = AvellanedaStoikov(
    t_terminal=1.0,
    sigma=0.2,
    gamma_risk=0.1,
    intensity_a=140.0,
    intensity_k=1.5,
)

# Current state: spot = 100.0, long 2 units of inventory, half-time elapsed
r_price = mm.reservation_price(s=100.0, q=2, t=0.5)
spread = mm.optimal_spread(t=0.5)
bid, ask = mm.optimal_quotes(s=100.0, q=2, t=0.5)

print(f"Reservation Price: {r_price:.4f}")
print(f"Optimal Spread:    {spread:.4f}")
print(f"Optimal Quotes:    Bid {bid:.4f} | Ask {ask:.4f}")
```

### 7. Model Morphing

Interpolate exactly between a calm and a stressed regime, measure how far apart they are, and simulate a smooth regime transition:

```python
import numpy as np
from spxa import VarianceGamma, MertonJumpDiffusion
from spxa.morph import CosineSchedule, DynamicMorphedProcess, gaussian_rao_distance

calm = VarianceGamma(sigma=0.12, nu=0.2, theta=-0.05)
stressed = MertonJumpDiffusion(sigma=0.20, rate=3.0, jump_mean=-0.08, jump_std=0.10)

# 1. Lévy–Khintchine homotopy: psi_alpha = (1 - alpha) psi_0 + alpha psi_1
mid = calm.morph(stressed, alpha=0.3, method="triplet")
print(mid.exactness)                       # ExactnessLevel.EXACT
k2 = mid.cumulants(order=2)
print(k2[2], 0.7 * calm.cumulants(order=2)[2] + 0.3 * stressed.cumulants(order=2)[2])  # 0.03719 0.03719

# 2. Fisher–Rao geodesic distance between Gaussian marginals N(mu, sigma^2)
print(gaussian_rao_distance(0.0, 0.20, 0.05, 0.35))  # 0.8125

# 3. Smooth calm -> stressed regime transition on [0.25, 0.75] with a cosine schedule
regime = DynamicMorphedProcess(calm, stressed, schedule=CosineSchedule(t0=0.25, t1=0.75))
paths, alphas = regime.simulate(n_steps=252, n_paths=1000, T=1.0,
                                rng=np.random.default_rng(0), return_alpha_paths=True)
print(paths.shape, alphas.shape, alphas[0, [0, 63, 126, 189, 251]])  # (1000, 253) (1000, 253) [0. 0. 0.5 1. 1.]

print(mid.__story__())
```

Other geometries are one keyword away: `method` accepts `"triplet"`, `"generator"`, `"geodesic"`, `"wasserstein"` and `"convex_combination"`. See [Theory: Model Morphing](docs/theory/model_morphing.md) for when each one is exact.

### 8. PDE & PIDE Solvers

Price an American put, a Heston call by ADI and a Merton put taken directly from the process zoo:

```python
from spxa.pde import BlackScholesPDE, HestonPDE
from spxa.zoo.levy import MertonJumpDiffusion

# 1. American put (Longstaff-Schwartz case): penalty LCP on Rannacher-smoothed Crank-Nicolson
am = BlackScholesPDE(r=0.06, sigma=0.2).american(K=40.0, T=1.0, spot=36.0, option="put")
print(round(am.price, 4), round(am.delta, 4))                 # 4.4863 -0.6969

# 2. Heston (in 't Hout-Foulon case 1) by Hundsdorfer-Verwer ADI
heston = HestonPDE(r=0.025, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.9)
res = heston.european(K=100.0, T=1.0, spot=100.0, v0=0.04, n_s=201, n_v=101, n_time=100)
print(round(res.price, 4), round(res.delta, 4))               # 8.8941 0.6716  (Lewis: 8.8949)

# 3. Exponential-Levy PIDE straight from a zoo process
merton = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
put = merton.to_pde(r=0.05).european(K=100.0, T=0.25, spot=100.0, option="put")
print(round(put.price, 4))                                    # 3.1489  (series: 3.1490)

print(res.__story__())
```

`spxa.pde.pde_price` values `spxa.risk` instruments for stress testing, and `spxa.graph.pde` puts PDE prices into computational graphs. See [Theory: PDE and PIDE methods](docs/theory/pde.md).

---

## Computational Graph Architecture & Duality Engine

`spxa.graph` maps models, transformations, and contracts to a Directed Acyclic Graph (DAG) optimized by `GraphCompiler`. Subgraphs are automatically routed to the most efficient backend:

```
                            ┌───────────────────────────────────┐
                            │    Universal Process Node         │
                            │    (GBM, Heston, VarianceGamma)   │
                            └─────────────────┬─────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
    │  Fourier Facet   │            │  Simulation Wire │            │ Closed-Form Facet│
    │  (Char-Function) │            │  (Path Sampler)  │            │ (Exact Moments)  │
    └─────────┬────────┘            └─────────┬────────┘            └─────────┬────────┘
              │                               │                               │
              ▼                               ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
    │ NumericalBackend │            │SimulationBackend │            │AnalyticalBackend │
    │ (Carr-Madan/COS) │            │ (Vectorized MC)  │            │ (Black-Scholes)  │
    └─────────┬────────┘            └─────────┬────────┘            └─────────┬────────┘
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │     Terminal Valuation    │
                                └─────────────┬─────────────┘
                                              │
                                ┌─────────────▼─────────────┐
                                │   GradientTape Autodiff   │
                                │   (Exact Greek Vectors)   │
                                └───────────────────────────┘
```

### Four-Facet Representation Duality
Every continuous-time stochastic process possesses up to 4 dual mathematical representations:
1. **SDE Facet**: Itô / Stratonovich drift and diffusion $dX_t = \mu(t, X_t)dt + \sigma(t, X_t)dW_t$.
2. **Infinitesimal Generator Facet**: Differential Dynkin operator $\mathcal{A}f(x) = \mu(x)f'(x) + \frac{1}{2}\sigma^2(x)f''(x) + \int [f(x+y)-f(x)-y f'(x)\mathbf{1}_{|y|<1}]\nu(dy)$.
3. **Fourier Characteristic Exponent Facet**: $\Phi_t(u) = \mathbb{E}[e^{i u X_t}] = e^{t \Psi(u)}$.
4. **Transition Density Facet**: Scale function $s(x)$, speed density $m(x)$, and Green's resolvent.

The `ProcessDualityCore` engine automatically translates and inverts between these facets, allowing seamless transitions from SDE specifications to characteristic functions and Dynkin boundary classifiers.

---

## Exactness Guarantees & Mathematical Integrity

Unlike numerical libraries that silently approximate or introduce uncontrolled truncation, `spxa` enforces a rigorous mathematical contract across all computations:

```
                       ┌─────────────────────────┐
                       │   ExactnessLevel.EXACT  │
                       └────────────┬────────────┘
                                    │ Degradation under non-linear
                                    │ transforms or state-dependence
                                    ▼
                 ┌──────────────────────────────────────┐
                 │ ExactnessLevel.MOMENT_PROPAGATION    │
                 └──────────────────┬───────────────────┘
                                    │ Degradation under intractable
                                    │ densities or path memory
                                    ▼
                 ┌──────────────────────────────────────┐
                 │   ExactnessLevel.SIMULATION_ONLY     │
                 └──────────────────────────────────────┘
```

- **Exact Arithmetic**: Operations between exact Lévy processes compute exact characteristic triplets.
- **Auditable Warnings**: When an operation causes exactness to degrade, `SpxaDegradationWarning` is emitted detailing the exact mathematical reason.
- **Property Lattice**: Automatically tracks and maintains invariants including stationarity of increments, independent increments, martingale status, tail behavior (Gaussian, Exponential, Heavy-tailed, Pareto), and self-similarity index.
- **Mathematical Audit Trail (`__story__`)**: Every process stores an internal composition DAG that outputs a human-readable step-by-step mathematical proof of its derivation and validity.

---

## Subpackage Ecosystem

| Subpackage | Namespace | Description | Key Modules & APIs |
| :--- | :--- | :--- | :--- |
| **Core** | `spxa.core` | Algebraic foundations, triplet types, and exactness lattice | `Process`, `LevyTriplet`, `LevyMeasure`, `ExactnessLevel` |
| **Process Zoo** | `spxa.zoo` | 198+ continuous-time models (Lévy, diffusions, jumps) | `VarianceGamma`, `NIG`, `Heston`, `SABR`, `BatesProcess`, `CoxIngersollRoss`, `CEV` |
| **Computational Graph** | `spxa.graph` | Stochastic DAG engine, compilers, and backends | `ComputationalGraph`, `GraphCompiler`, `GradientTape`, `to_graph` |
| **Information Geometry** | `spxa.geom` | Differential geometry of statistical manifolds | `FisherInformationMetric`, `AffineConnection`, `GeodesicFlow` |
| **Rough Paths** | `spxa.signature` | Path signatures, shuffle algebra, and PDE kernels | `PathSignature`, `SignatureAlgebra`, `SignatureKernel` |
| **Rough Volatility** | `spxa.rough` | Fractional Brownian motion and Volterra calculus | `FractionalKernel`, `RoughBergomi`, `FractionalRiccatiSolver` |
| **Model Morphing** | `spxa.morph` | Exact and geometric interpolation between processes, regime schedules, and barycenters | `MorphedProcess`, `DynamicMorphedProcess`, `CosineSchedule`, `ModelBarycenter`, `gaussian_rao_distance` |
| **PDE & PIDE Solvers** | `spxa.pde` | Finite differences for pricing, densities, Lévy PIDEs and ADI | `BlackScholesPDE`, `HestonPDE`, `LevyPIDE`, `FokkerPlanck1D`, `ADISolver`, `pde_price` |
| **Stochastic Control** | `spxa.control` | Portfolio optimization, market making, and HJB solvers | `MertonPortfolio`, `AvellanedaStoikov`, `AlmgrenChriss`, `HJBSolver` |
| **Model Calibration** | `spxa.calibration` | Volatility surface calibration, MLE, GMM, and filtering | `HestonCalibrator`, `CIRMaximumLikelihood`, `UnscentedKalmanFilter`, `ParticleFilter`, `sandwich_covariance_matrix` |
| **Copula & Dependence** | `spxa.copula` | Multi-asset dependence and extreme value copulas | `ClaytonCopula`, `GumbelCopula`, `GaussianCopula`, `CoupledProcess` |
| **Stochastic Calculus** | `spxa.calculus` | Itô calculus, Dynkin generators, and Malliavin weights | `apply_ito_symbolic`, `InfinitesimalGenerator`, `bismut_elworthy_li_delta`, `compute_scale_and_speed_densities` |
| **Differentiable SDEs** | `spxa.diff` | JAX-differentiable SDE solvers, pathwise Greeks, and neural SDEs | `DifferentiableHeston`, `solve_sde`, `compute_pathwise_greeks`, `DifferentiableNeuralSDE` |
| **Simulation** | `spxa.sim` | Discretisation schemes, exact samplers, QMC, and MLMC | `euler_maruyama`, `milstein`, `stochastic_runge_kutta`, `mlmc_estimate` |
| **Financial Risk** | `spxa.risk` | Portfolio risk engines, stress testing, SIMM/FRTB, and XVA analytics | `Portfolio`, `StressTestingEngine`, `ISDASimmEngine`, `XVACalculator` |
| **Visualization** | `spxa.vis` | Volatility surfaces, smiles, Lévy densities, and composition DAGs | `plot_iv_surface`, `plot_smile_comparison`, `plot_dag`, `inspect_levy_triplet` |

---

## Performance & Benchmarks

All core components are benchmarked continuously under `pytest-benchmark` against high-performance targets. Timings below reflect single-threaded CPU execution on modern x86_64 hardware:

| Benchmark Operation | Target Component | Execution Time (Median) | Operations / Sec |
| :--- | :--- | :---: | :---: |
| **Characteristic Function Evaluation** | `spxa.BrownianMotion` $\Phi(u)$ | **17.5 μs** | ~52,700 ops/s |
| **Copula Bivariate Sampling (N=1,000)** | `spxa.ClaytonCopula` | **52.7 μs** | ~17,500 ops/s |
| **Brownian Motion Path Simulation** | `spxa.BrownianMotion` ($T=1, N=252$) | **165.9 μs** | ~5,750 ops/s |
| **Algebraic Triplet Composition** | `spxa.core` ($X + 0.5Y$) | **454.0 μs** | ~2,160 ops/s |
| **Heston Bivariate Simulation** | `spxa.HestonProcess` ($N=252$ steps) | **1.05 ms** | ~920 ops/s |
| **Goursat Hyperbolic PDE Kernel** | `spxa.SignatureKernel` ($100 \times 100$ grid) | **3.8 ms** | ~260 ops/s |
| **Reverse-Mode Greek Vector (Delta/Gamma/Vega/Theta/Rho)** | `spxa.graph.GradientTape` | **2.1 ms** | ~470 ops/s |

---

## Installation

### Standard Installation

Install `spxa` from PyPI via `pip`:

```bash
pip install spxa
```

Requires **Python ≥ 3.10**.

### Development Installation

To install `spxa` from source with all development, testing, and benchmarking dependencies:

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
```

Optional PyTorch acceleration adapter can be installed alongside:
```bash
pip install torch
```

---

## Development & Testing

`spxa` maintains an extensive test suite verifying mathematical exactness against published literature and ensuring strict code quality:

```bash
# Run unit and exactness test suites
pytest

# Run performance benchmarks
pytest benchmarks/

# Check code formatting and linting
ruff check .
ruff format --check .

# Run static type checking with strict type coverage
mypy spxa
```

---

## Documentation & Resources

- **[Full API Reference Manual](docs/api.md)**: Exhaustive class and function documentation for all modules.
- **[Process Registry](PROCESS_REGISTRY.md)**: Master catalog of 198+ implemented stochastic processes, exactness ratings, and academic citations.
- **[Theory: Lévy–Khintchine Formula](docs/theory/levy_khintchine.md)**: Foundations of infinitely divisible distributions and triplet arithmetic.
- **[Theory: Triplet Arithmetic under Composition](docs/theory/triplet_arithmetic.md)**: Compositional rules for linear combinations, scaling, and subordination.
- **[Theory: Beyond the Lévy Regime](docs/theory/beyond_levy.md)**: Analysis of non-independent increments, memory kernels, and degradation lattices.
- **[Theory: Model Morphing](docs/theory/model_morphing.md)**: Triplet, generator, Fisher–Rao and Wasserstein homotopies, exactness rules, and dynamic schedules.
- **[Theory: Information Geometry](docs/theory/information_geometry.md)**: Fisher metric, α-connections, curvature, Rao distance, and natural-gradient calibration.
- **[Theory: Rough Paths & Signatures](docs/theory/rough_paths_signatures.md)**: Tensor algebra, log-signatures, the Goursat signature kernel, and expected signatures.
- **[Theory: Rough Volatility](docs/theory/rough_volatility.md)**: Volterra fBM, rough Bergomi and its ATM skew, and the rough Heston fractional Riccati equation.
- **[Theory: Copulas & Dependence](docs/theory/copulas.md)**: Sklar's theorem, dependence measures, copula families, estimation, and coupled simulation.
- **[Theory: Stochastic Control](docs/theory/stochastic_control.md)**: HJB, LQG, Merton, Almgren–Chriss, Avellaneda–Stoikov, and finite-difference HJB.
- **[Theory: Estimation & Calibration](docs/theory/calibration.md)**: Exact MLE, GMM/ECF, filtering, surface calibration, and statistical inference.
- **[Computational Graph Deep-Dive](docs/computational_graph.md)**: Architectural guide to DAG compilers, execution backends, and autodiff tapes.
- **[Interactive Tutorials](docs/tutorials/)**: Jupyter notebooks covering introductory compositions, derivative pricing, and physical systems.
- **[Changelog](CHANGELOG.md)**: Detailed version history and architectural evolution.
- **[Contributing Guidelines](CONTRIBUTING.md)**: Development workflows, RFC proposals, and pull request standards.

---

## Citation

If you use `spxa` in your academic research, quantitative modeling, or open-source projects, please cite it. Citation metadata is in [`CITATION.cff`](CITATION.cff), so GitHub's **Cite this repository** button gives APA and BibTeX directly. Equivalently:

```bibtex
@software{hapani2026spxa,
  author       = {Nitya Hapani and Contributors},
  title        = {spxa: Stochastic Process Algebra and Computational Graph Engine},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Nityahapani/spxa}},
  version      = {1.0.1}
}
```

---

## License

`spxa` is open-source software released under the [MIT License](LICENSE).

Participation in the project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report vulnerabilities privately as described in the [Security Policy](SECURITY.md).
