# PDE and PIDE Methods

Every Markov process in spxa has a generator $\mathcal{A}$. By the Feynman–Kac theorem, a discounted expectation

$$
u(x, \tau) = \mathbb{E}\!\left[e^{-\int_0^\tau r\,ds}\, g(X_\tau) \,\middle|\, X_0 = x\right]
$$

solves the backward equation $u_\tau = \mathcal{A}u - r u$ with $u(x, 0) = g(x)$. Here $\tau = T - t$ is time to maturity. `spxa.pde` discretises these equations with finite differences. Every scheme in the package has a test that measures its convergence order against a closed form or a published reference.

| Object | Implementation |
| :--- | :--- |
| Grids, boundary conditions, θ-stepping | `Grid1D`, `TimeGrid`, `Dirichlet`/`Neumann`/`Linear`/`Natural`, `ThetaSolver1D` |
| Early exercise (LCP) | `PenaltyLCP`, `PSORLCP`, `BrennanSchwartzLCP`, `IkonenToivanenLCP` |
| Black–Scholes and local volatility | `BlackScholesPDE`, `DupirePDE`, `DupireSurface` |
| Asian options | `AsianPDE` (Večeř) |
| Short-rate models | `ShortRatePDE` |
| Generators of zoo processes | `to_pde`, `DiffusionPDE`, `Process.to_pde()` |
| Transition densities | `FokkerPlanck1D`, `DensitySolution` |
| Exponential Lévy models | `LevyJumpOperator`, `LevyPIDE` |
| Two-factor problems (ADI) | `DiscreteOperator2D`, `ADISolver`, `HestonPDE`, `TwoAssetBlackScholesPDE` |
| Simulation cross-check | `longstaff_schwartz`, `lsm_american` |
| Bridges | `pde_price` / `pde_pricing_fn` (`spxa.risk`), `spxa.graph.pde` nodes |
| Verification | `convergence_study`, `observed_order`, `richardson_extrapolate`, `spxa.pde.reference` |

---

## 1. One-dimensional problems

### 1.1 Equation and grids

Every one-dimensional problem has the form

$$
u_\tau = a(x,\tau)\,u_{xx} + b(x,\tau)\,u_x + c(x,\tau)\,u + f(x,\tau),
$$

where $a \ge 0$. Each coefficient can be a scalar, an array of nodal values, or a callable `f(x, tau)`.

**Grids.** Grids are either uniform or sinh-stretched:

$$
x(\xi) = c + \alpha \sinh\!\big(c_2\xi + c_1(1-\xi)\big), \qquad \xi \in [0, 1],
$$

with $c_1 = \operatorname{arsinh}((x_{\min}-c)/\alpha)$ and $c_2 = \operatorname{arsinh}((x_{\max}-c)/\alpha)$. This clusters nodes around the strike $c$, and $\alpha$ sets how strongly. Strikes, barriers and spots are placed *exactly* on nodes. The grid is split at each of these points in $\xi$, and the nodes are spread in proportion to the lengths of the pieces. This keeps the spacing ratio at $1 + O(1/n)$. Snapping a node onto the strike after the fact would instead perturb the local spacing by up to half a cell.

**Stencils.** First and second derivatives use three-point stencils that are exact for quadratics on non-uniform grids, so the scheme is second order. Where the cell Péclet number $|b|h/(2a)$ exceeds one, the drift term can switch to an upwind stencil (`convection_scheme='auto'`).

**Boundary conditions.** Four kinds are available:

- `Dirichlet`: $u = g(\tau)$.
- `Neumann`: $u_x = g(\tau)$.
- `Linear`: $u_{xx} = 0$, appropriate for far fields where the solution is asymptotically linear.
- `Natural`: the equation itself, with one-sided derivatives. Use this on degenerate boundaries such as $S = 0$, where $a = b = 0$.

### 1.2 Time stepping

The solver uses the θ-scheme

$$
(I - \theta\Delta\tau A)\,U^{n+1} = (I + (1-\theta)\Delta\tau A)\,U^n + \Delta\tau\,(\theta F^{n+1} + (1-\theta)F^n).
$$

It factorises each matrix once with sparse LU (`splu`) and reuses the factorisation while the coefficients stay constant. Crank–Nicolson ($\theta = \frac12$) is second order. However, the kink of a vanilla payoff excites the high-frequency modes that Crank–Nicolson does not damp. To control this, the solver first takes two fully implicit half-steps (Rannacher, 1984) and projects the payoff onto control-volume averages next to the kink. Together these restore second-order convergence for values and Greeks.

Events are specified in calendar time and converted with $\tau = T - t$. They include:

- discrete dividends, applied as the jump condition $u(S) \leftarrow u(S - D)$ by cubic interpolation;
- Bermudan exercise dates;
- discrete barrier monitoring.

### 1.3 Early exercise

An American option solves the linear complementarity problem (LCP)

$$
u_\tau - \mathcal{L}u \ge 0,\qquad u \ge g,\qquad (u_\tau - \mathcal{L}u)(u - g) = 0.
$$

After discretisation, each time step becomes a finite LCP $Mu \ge q$, $u \ge g$, $(Mu-q)^\top(u-g) = 0$. Four solvers are provided:

| Solver | Method |
| :--- | :--- |
| `PenaltyLCP` | Forsyth–Vetzal penalty iteration $(M + P(u))u = q + P(u)g$, with penalty $1/\mathrm{tol}$ |
| `PSORLCP` | Projected SOR |
| `BrennanSchwartzLCP` | Direct elimination for tridiagonal $M$ with a single exercise region on the given `side` |
| `IkonenToivanenLCP` | Operator splitting: $\tilde u$ from the linear system with multiplier $\lambda$; then $u = \max(\tilde u - \Delta\tau\lambda, g)$ and $\lambda \leftarrow \max(0, \lambda + (g - \tilde u)/\Delta\tau)$ |

`complementarity_residual` measures how well a computed solution satisfies the LCP. The exercise boundary is recorded at every step and returned as an `ExerciseBoundary`.

### 1.4 Black–Scholes and its variants

`BlackScholesPDE(r, sigma, q)` solves the equation in $S$ on $[0, S_{\max}]$ with

$$
S_{\max} = \max(S_0, K)\,e^{6\sigma\sqrt T},
$$

a natural boundary at $S = 0$ and a linear boundary at $S_{\max}$. The volatility $\sigma$ can be a function $\sigma(S, t)$ (local volatility). Supported contracts:

- **European, digital, American, Bermudan.** Vega and rho are available by bump-and-revalue.
- **Barriers.** Continuously monitored barriers end the computational domain, with a Dirichlet rebate there. Knock-ins use in–out parity: vanilla minus knock-out, plus the rebate times the price of a no-touch claim.
- **Double barriers and discretely monitored barriers.** For discrete monitoring, the solution is reset to the rebate outside the barrier on each monitoring date.

**Verification.** The pricers are checked against:

- the Black–Scholes formula, with second-order convergence of price, delta and gamma;
- Reiner–Rubinstein barrier formulas and the sine-series double-barrier price;
- the Broadie–Glasserman–Kou continuity correction and Fusai's discrete barrier values;
- the Leisen–Reimer binomial tree for American options;
- the perpetual put $\frac{K}{1+\beta}\left(\frac{S}{S^*}\right)^{-\beta}$.

### 1.5 Local volatility (Dupire)

Call prices as a function of strike, $C(K, T)$, satisfy the forward equation

$$
C_T = \tfrac12\sigma^2(K,T)K^2 C_{KK} - (r-q)K C_K - qC,
$$

with $C(K, 0) = (S_0 - K)^+$, $C(0, T) = S_0e^{-qT}$ and $C(K_{\max}, T) = 0$. `DupirePDE` prices a whole strike–maturity surface in a single solve. `local_vol_from_prices` inverts the relation numerically. In the tests, CEV prices from the non-central $\chi^2$ formula are reproduced to about $10^{-4}$.

### 1.6 Asian options (Večeř)

A fixed-strike Asian call on $A = \frac1T\int_0^T S_t\,dt$ can be replicated by a self-financing portfolio that holds $q_t$ shares of the underlying. For a continuous average,

$$
q_t = \frac{e^{-\delta(T-t)} - e^{-r(T-t)}}{(r-\delta)T}.
$$

For a discrete average,

$$
q_t = \frac{e^{-r(T-t)}}{n}\sum_{t_i > t} e^{(r-\delta)(t_i - t)}.
$$

Here $\delta$ is the dividend yield. Measure the portfolio value in units of the dividend-reinvested asset $S_te^{\delta t}$ and call the result $z_t$. Then $z_t$ is a driftless diffusion, $dz = \sigma(\varphi_t - z)\,dW$ with $\varphi_t = q_te^{-\delta t}$. The price is $S_0u(0, z_0)$, where

$$
u_t + \tfrac12\sigma^2(\varphi_t - z)^2u_{zz} = 0,\qquad u(T, z) = z^+,\qquad z_0 = q_0 - e^{-rT}\frac{K - A_{\text{past}}}{S_0}.
$$

Far from the kink, $u \to 0$ as $z \to -\infty$ and $u \to z$ as $z \to \infty$. These limits are imposed as Dirichlet conditions. For discrete averaging, $q_t$ jumps at each fixing, so the equation is solved segment by segment between fixings. The tests reproduce Linetsky's spectral values to $10^{-4}$.

### 1.7 Short rate

`ShortRatePDE` prices zero-coupon bonds, coupon bonds and European or American bond options. It solves

$$
u_\tau = \tfrac12\sigma^2r^{2\beta}u_{rr} + \kappa(\theta - r)u_r - ru,
$$

with $\beta = 0$ for Vasicek and $\beta = \frac12$ for CIR. For CIR the domain starts at $r = 0$, and the equation itself is the boundary condition there. A bond option is solved in two stages. First the bond is solved to its maturity $S$. Then the option is solved to its expiry $T$, with the payoff kink at $r^*$ placed on a node. Vasicek option prices are checked against Jamshidian's formula with

$$
\sigma_p = \frac{\sigma}{\kappa}\big(1 - e^{-\kappa(S-T)}\big)\sqrt{\frac{1 - e^{-2\kappa T}}{2\kappa}},
$$

and agree to about $10^{-6}$. CIR bond options are checked against the non-central $\chi^2$ formula.

---

## 2. Generators and densities

`to_pde(process)` maps a zoo process to its pricing equation:

- For a diffusion $dX = \mu(X)\,dt + \sigma(X)\,dW$ it returns a `DiffusionPDE` with $a = \frac12\sigma^2$ and $b = \mu$.
- For an exponential Lévy model it returns a `LevyPIDE` (Section 3). This requires the rate `r`.

`Process.to_pde()` delegates to the same function.

**Fokker–Planck.** The density $p(x,t)$ of $X_t$ solves the adjoint equation $p_t = \mathcal{A}^*p$. `FokkerPlanck1D` does not discretise $\mathcal{A}^*$ separately. Instead it uses the *discrete adjoint* of the backward operator,

$$
P = V^{-1}A^\top V,
$$

where $V$ holds the control-volume widths. Because $\mathbf{1}^\top A = 0$ row-wise for a generator, this gives $\frac{d}{dt}\sum_i V_ip_i = 0$, so total mass is conserved to machine precision. It also makes forward and backward prices consistent: $\sum_i V_i p_i(T)\,g_i$ equals the backward solution at $x_0$. The tests check Ornstein–Uhlenbeck and CIR densities in $L^1$ ($3\times10^{-5}$ for OU) and verify mass conservation to $10^{-12}$.

---

## 3. Partial integro-differential equations

For $S_t = S_0e^{X_t}$ with $X$ a Lévy process with triplet $(b, \sigma^2, \nu)$, the price in $x = \ln S$ solves

$$
u_\tau = \tfrac12\sigma^2 u_{xx} + \Big(r - q - \tfrac12\sigma^2 - \omega\Big)u_x - ru + \int_{\mathbb{R}}\big[u(x+y) - u(x)\big]\,\nu(dy),
$$

where $\omega = \int (e^y - 1)\,\nu(dy)$ is the martingale correction. `LevyJumpOperator` discretises the integral term on a uniform grid with spacing $h$. It treats jumps of different sizes differently:

- **Large jumps** ($|y| \ge h/2$) use cell-integrated weights $\nu([y_k - h/2, y_k + h/2])$. The convolution is computed with `fftconvolve`. Outside the domain, the solution is replaced by its asymptotic values.
- **Small jumps** ($|y| < h/2$) are replaced by a diffusion with variance $\int_{|y|<h/2} y^2\,\nu(dy)$ (Asmussen–Rosiński). For infinite-activity models this reduces the error from $O(h^{2-Y})$ to $O(h^2)$.
- **Interpolation error.** The variance lost by representing jumps on the grid is matched exactly by an extra local diffusion. Pure-jump models can then have a negative net local coefficient; the operator is assembled directly so that this case is handled.

**Time stepping and American options.** The integral term is treated explicitly inside a fixed-point iteration around Crank–Nicolson with Rannacher start-up. Each step therefore needs only tridiagonal solves. American options combine the penalty iteration with the jump iteration. Alternatively, Ikonen–Toivanen can be applied once per step, which adds a splitting error of $O(\Delta\tau)$.

**Verification** (`n = 801`):

| Model | Reference | Error |
| :--- | :--- | :--- |
| Merton put | series | $9\times10^{-5}$, second order |
| Kou | Kou (2002) closed form; Lewis formula for puts and Greeks | $\le 8\times10^{-5}$ for calls |
| Variance Gamma, CGMY ($Y = 0.5, 1.5$), NIG | Lewis formula | $\le 3\times10^{-4}$ |
| American Merton puts | d'Halluin–Forsyth–Labahn / compact-scheme references | $\le 1.4\times10^{-4}$ |

The Lewis formula used throughout is

$$
C = S e^{-qT} - \frac{\sqrt{SK}\,e^{-(r+q)T/2}}{\pi}\int_0^\infty \operatorname{Re}\!\left[e^{iuk}\varphi_X(u - \tfrac i2)\right]\frac{du}{u^2 + \frac14}, \qquad k = \ln(S/K) + (r-q)T.
$$

---

## 4. Two dimensions: ADI

### 4.1 Splitting

Two-factor problems have the form

$$
u_\tau = a_{xx}u_{xx} + a_{xy}u_{xy} + a_{yy}u_{yy} + b_xu_x + b_yu_y + cu + f.
$$

They are discretised on a tensor grid with second-order stencils. The mixed derivative uses a nine-point stencil at interior nodes. `DiscreteOperator2D` splits the semi-discrete operator as

$$
A = A_0 + A_1 + A_2,
$$

where $A_0$ is the mixed term and $A_1$, $A_2$ are the $x$ and $y$ parts, each carrying half of $c$. $A_1$ is tridiagonal in $x$ and $A_2$ is tridiagonal in $y$, so the implicit stages reduce to one-dimensional solves.

Edges can be `Dirichlet2D`, `Neumann2D` (ghost point), `Natural2D` (the equation with one-sided derivatives) or `Linear2D` ($u_{nn} = 0$). The mixed term is omitted on edges. This is exact whenever $a_{xy}$ or $u_{xy}$ vanishes there, which is the case for every edge used by the Heston and two-asset pricers.

### 4.2 Schemes

With $F_j(t, U) = A_jU + b_j(t)$, the Douglas scheme is

$$
\begin{aligned}
Y_0 &= U^n + \Delta t\,F(t_n, U^n),\\
Y_j &= Y_{j-1} + \theta\Delta t\big(F_j(t_{n+1}, Y_j) - F_j(t_n, U^n)\big),\qquad j = 1, 2.
\end{aligned}
$$

The Craig–Sneyd (CS), modified Craig–Sneyd (MCS) and Hundsdorfer–Verwer (HV) schemes add a correction stage and repeat the implicit sweeps. For HV, the second sweep is taken relative to $Y_2$ at $t_{n+1}$. `ADISolver` implements all four (in 't Hout & Welfert, 2007, 2009), with these defaults:

| Scheme | Default $\theta$ | Order with a mixed term |
| :--- | :--- | :--- |
| Douglas | $\frac12$ | 1 |
| Craig–Sneyd | $\frac12$ | 2 |
| Modified Craig–Sneyd | $\frac13$ | 2 |
| Hundsdorfer–Verwer | $\frac12 + \frac{\sqrt3}{6}$ | 2 |

As in one dimension, non-smooth payoffs are handled by first taking implicit damping steps (Douglas with $\theta = 1$ on half-steps). Early exercise uses the Ikonen–Toivanen splitting: the multiplier enters the explicit stage, and the projection is applied after the step (Haentjens & in 't Hout, 2015).

The tests use the exact solution $u = e^{\alpha x + \beta y + \gamma\tau}$ on stretched grids with a mixed term. They measure second order in space for CS, MCS and HV, and the time orders listed in the table above.

### 4.3 Heston

The Heston equation in $(S, v)$ is

$$
u_\tau = \tfrac12 S^2 v\,u_{SS} + \rho\sigma Sv\,u_{Sv} + \tfrac12\sigma^2v\,u_{vv} + (r-q)Su_S + \kappa(\theta - v)u_v - ru.
$$

It is solved on $[0, 8K]\times[0, 5]$ with the boundary conditions of in 't Hout & Foulon (2010):

| Edge | Condition |
| :--- | :--- |
| $S = 0$ | Dirichlet |
| $S_{\max}$ | $u_S = e^{-q\tau}$ (calls) or $0$ (puts) |
| $v = 0$ | The degenerate equation itself, with a one-sided $u_v$ (no condition needed, whether or not the Feller condition holds) |
| $v_{\max}$ | $u_v = 0$ |

The grids are sinh-stretched around the strike with $c = K/5$, and around $v = 0$ with $d = v_{\max}/500$. Where convection in $v$ dominates, the $u_v$ stencil is upwinded at second order.

**Verification:**

- **Case 1 of in 't Hout–Foulon** ($\kappa = 1.5$, $\theta = 0.04$, $\sigma = 0.3$, $\rho = -0.9$): matches the Lewis price with second-order convergence, and errors below $10^{-3}$ on $201\times101$.
- **American puts:** the Ikonen–Toivanen (2009) reference values ($K = 10$, $T = 0.25$, $S = 8, \dots, 12$, $v_0 = 0.0625$ and $0.25$) are reproduced within $2\times10^{-4}$.

### 4.4 Two assets

`TwoAssetBlackScholesPDE` uses:

- the equation itself on the axes, where the mixed term vanishes;
- linear conditions on the far edges;
- sinh grids around the spots.

**Verification:**

- Margrabe's exchange option, with second-order convergence for $\rho \in \{-0.6, 0, 0.5, 0.9\}$.
- Spread options against a conditional (Gauss–Hermite) formula, which is itself checked against adaptive quadrature.
- The identity best-of + worst-of = two vanilla calls.
- A payoff that depends on one asset reproduces the Black–Scholes price.

---

## 5. Least-squares Monte Carlo

`longstaff_schwartz` estimates continuation values by regressing discounted cash flows on weighted Laguerre polynomials $e^{-x/2}L_k(x)$ or on monomials, using in-the-money paths only.

In two-pass mode, the exercise rule is fitted on independent regression paths and then applied to the valuation paths. The resulting estimate is unbiased for a sub-optimal policy, so it is a lower bound in expectation and comes with an honest standard error. On the Longstaff–Schwartz test cases, it agrees with the Bermudan PDE value with the same 50 exercise dates to within three standard errors.

---

## 6. Using the results

**Result objects.** `PDEPricingResult`, `HestonPDEResult`, `TwoAssetPDEResult`, `LSMResult`, `DensitySolution` and `PDEQuote` all implement `__story__()`, which gives a plain-language account of the grid, the scheme and the Greeks. Solutions can be evaluated anywhere in the domain:

- one dimension: cubic interpolation;
- two dimensions: bicubic splines.

**Greeks.** Greeks come from the grid via three-point nodal stencils followed by interpolation. On a log grid, $\Delta = u_x/S$ and $\Gamma = (u_{xx} - u_x)/S^2$.

**Bridges to the rest of spxa.**

- `pde_price(instrument, market)` values `spxa.risk` instruments. `pde_pricing_fn(book)` turns a book into the callable that `StressTestingEngine` expects.
- `spxa.graph.pde.bs_pde_price_node` and `heston_pde_price_node` put PDE prices into computational graphs. Their vector–Jacobian products use the grid Greeks.

**Convergence studies.** `convergence_study` refines grids and reports observed orders and a Richardson-extrapolated value. It is the recommended way to confirm the accuracy of any new configuration.

## References

- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Vol. I, ch. 2.
- Cont, R. & Voltchkova, E. (2005). A finite difference scheme for option pricing in jump diffusion and exponential Lévy models. *SIAM J. Numer. Anal.* 43(4).
- d'Halluin, Y., Forsyth, P. & Vetzal, K. (2005). Robust numerical methods for contingent claims under jump diffusion processes. *IMA J. Numer. Anal.* 25.
- Forsyth, P. & Vetzal, K. (2002). Quadratic convergence for valuing American options using a penalty method. *SIAM J. Sci. Comput.* 23(6).
- Haentjens, T. & in 't Hout, K. (2015). ADI schemes for pricing American options under the Heston model. *Appl. Math. Finance* 22(3).
- Ikonen, S. & Toivanen, J. (2009). Operator splitting methods for pricing American options under stochastic volatility. *Numer. Math.* 113.
- in 't Hout, K. & Foulon, S. (2010). ADI finite difference schemes for option pricing in the Heston model with correlation. *Int. J. Numer. Anal. Model.* 7(2).
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.
- Longstaff, F. & Schwartz, E. (2001). Valuing American options by simulation: a simple least-squares approach. *Rev. Financ. Stud.* 14(1).
- Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numer. Math.* 43.
- Večeř, J. (2001). A new PDE approach for pricing arithmetic average Asian options. *J. Comput. Finance* 4(4).
- Windcliff, H., Forsyth, P. & Vetzal, K. (2004). Analysis of the stability of the linear boundary condition for the Black–Scholes equation. *J. Comput. Finance* 8(1).
