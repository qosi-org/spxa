# Changelog

All notable changes to spxa are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `pde/` — finite-difference PDE and PIDE solvers (`spxa.pde`, `spxa.graph.pde`). Every scheme is
  covered by a convergence test against a closed form or a published reference. Contents:
  - **Infrastructure.**
    - Grids: uniform and sinh-stretched `Grid1D` with strikes, barriers and spots placed on nodes;
      `TimeGrid` with calendar events.
    - Boundary conditions: Dirichlet, Neumann, linear and natural.
    - Operators and time stepping: second-order `DiscreteOperator1D` with automatic upwinding;
      `ThetaSolver1D` (Crank–Nicolson with Rannacher start-up); payoff projection.
    - Results: grid Greeks and `__story__()` summaries.
  - **Early exercise.** `PenaltyLCP`, `PSORLCP`, `BrennanSchwartzLCP` and `IkonenToivanenLCP`, plus
    `complementarity_residual` and recorded exercise boundaries.
  - **One-factor pricers.**
    - `BlackScholesPDE`: European, digital, American, Bermudan, single, double and discretely
      monitored barriers, discrete dividends, local volatility, and bumped vega/rho.
    - `DupirePDE` and `local_vol_from_prices`.
    - `AsianPDE` (Večeř; continuous and discrete averages).
    - `ShortRatePDE` (Vasicek/CIR bonds and European/American bond options).
  - **Generators and densities.**
    - `to_pde` and `Process.to_pde()`: diffusion generators and exponential-Lévy PIDEs.
    - `FokkerPlanck1D`: a discrete-adjoint density solver that conserves mass to machine precision.
  - **Lévy PIDEs.** `LevyJumpOperator` (cell-integrated large jumps, moment-matched small jumps,
    FFT convolution) and `LevyPIDE` for European and American options under Merton, Kou, VG, CGMY
    and NIG.
  - **Two-factor ADI.**
    - Building blocks: `Grid2D`, `Dirichlet2D`/`Neumann2D`/`Natural2D`/`Linear2D`, `LinearPDE2D`,
      `DiscreteOperator2D`.
    - `ADISolver`: Douglas, Craig–Sneyd, modified Craig–Sneyd and Hundsdorfer–Verwer, with damping
      steps and Ikonen–Toivanen early exercise.
    - `HestonPDE`: European and American, with Greeks and vega.
    - `TwoAssetBlackScholesPDE`: exchange, spread, best-of and worst-of options, European or American.
  - **Cross-checks and bridges.**
    - `longstaff_schwartz` and `lsm_american`: two-pass least-squares Monte Carlo.
    - `pde_price` and `pde_pricing_fn`: price `spxa.risk` instruments and feed `StressTestingEngine`.
    - `spxa.graph.pde.bs_pde_price_node` and `heston_pde_price_node`: graph nodes whose
      vector–Jacobian products use the grid Greeks.
  - **Verification.**
    - Convergence tools: `convergence_study`, `observed_order` and `richardson_extrapolate`.
    - Closed-form references in `spxa.pde.reference`: Black–Scholes, Reiner–Rubinstein, binomial
      trees, perpetual put, Lewis with Heston/Lévy characteristic functions, Merton series,
      Vasicek/CIR bonds and options, Margrabe, spread options.
  - **Documentation.** See `docs/theory/pde.md` and section 19 of the API reference.

### Fixed

- `zoo.levy.CGMY`: the Lévy density, total mass, tail and drift now temper upward jumps with $M$ and
  downward jumps with $G$, as the characteristic exponent does; previously they described the
  mirror-image process. For $1 \le Y < 2$ the drift now includes the $-\int_{|x|>1} x\,\nu(dx)$
  correction. The first cumulant is no longer rejected as nonexistent when $\Gamma(1-Y)$ is finite.
- `zoo.levy.NIG`: the triplet drift and tail integrated the NIG probability density instead of the
  Lévy density. The drift now follows from $\mathbb{E}[X_1] = \mu + \delta\beta/\gamma$, and the tail
  integrates the Lévy density.
- `zoo.levy.KouJumpDiffusion`: `char_func` now uses the closed-form exponent. The inherited numerical
  Lévy–Khintchine integral returned `nan` at complex arguments such as $u = -i$.

## [1.0.1] — 2026-10-30

### Added

- `morph/` — Model Morphing (`spxa.morph`, `spxa.graph.morph`): continuous interpolation between
  processes. The triplet homotopy $\psi_\alpha = (1-\alpha)\psi_0 + \alpha\psi_1$ is `EXACT` for
  Lévy processes. Also includes generator, Fisher–Rao geodesic, Wasserstein and convex-combination
  morphs; time- and state-dependent schedules (`DynamicMorphedProcess`); `ModelBarycenter`,
  `ModelEnsemble` and `SurfaceMorphEngine`; and the `Process.morph` / `spxa.morph` API.
  See `docs/theory/model_morphing.md`.
- `zoo/beyond/multivariate_hawkes.py` — Multivariate Hawkes mutually exciting point process (`MultivariateHawkesProcess`, `MultivariateHawkes`, `MHawkes`):
  $M$-dimensional Hawkes intensity $\lambda_m(t)$, branching infectivity matrix $\mathbf{\Gamma}$, spectral radius stationarity condition $\rho(\mathbf{\Gamma}) < 1$, exact expected arrival intensity $\mathbf{\lambda}_\infty = (\mathbf{I} - \mathbf{\Gamma})^{-1} \mathbf{\mu}$, and Ogata thinning event simulation.
- `zoo/levy/nts.py` — Normal Tempered Stable (NTS) Lévy process (`NormalTemperedStableProcess`, `NormalTemperedStable`, `NTSProcess`):
  Subordinated Brownian motion $X_t = \theta T_t + \sigma W(T_t)$ with Tempered Stable subordinator, exact closed-form characteristic function, exact cumulant generating function, and subordination simulation.
- `zoo/beyond/kou_sv.py` — Kou Stochastic Volatility Jump-Diffusion model (`KouSVProcess`, `KouSV`, `KouStochasticVolatility`):
  Coupled Heston mean-reverting stochastic volatility with Kou asymmetric double-exponential jumps, affine characteristic function, Lewis Fourier inversion European call/put options, and full-truncation Euler simulation.
- `zoo/beyond/diffusion_bridge.py` — General Diffusion Bridge via Doob's $h$-transform (`GeneralDiffusionBridge`, `DiffusionBridge`, `DoobBridge`):
  Drift correction $\mu^*(x, t) = \mu(x) + \sigma^2(x) \partial_x \ln p(b, T \mid x, t)$ for Brownian, Ornstein–Uhlenbeck, and Geometric Brownian Motion bridges, exact moments, and pinned path simulation.
- `zoo/beyond/rough_heston.py` — Rough Heston stochastic volatility model (`RoughHestonProcess`, `RoughHeston`, `RoughVolatility`):
  Fractional Volterra stochastic volatility of order $\alpha = H + 1/2 \in (1/2, 1)$, singular kernel $K(t) = t^{\alpha - 1} / \Gamma(\alpha)$, Mittag-Leffler expected variance, power-law implied volatility skew $|\partial \sigma_{IV} / \partial k| \sim T^{H - 0.5}$, and fractional convolution simulation.
- `zoo/beyond/g2pp.py` — Two-Factor Hull–White (G2++) short-rate diffusion model (`G2PlusPlusProcess`, `G2PlusPlus`, `TwoFactorHullWhite`):
  Two coupled OU factors driving short rate $r(t) = x(t) + y(t) + \phi(t)$, closed-form zero-coupon bond pricing $P(t, T)$, Brigo–Mercurio European bond option formulas, and exact bivariate Gaussian factor simulation.
- `zoo/beyond/three_halves.py` — 3/2 Stochastic Volatility diffusion model (`ThreeHalvesProcess`, `ThreeHalves`, `ThreeHalvesModel`):
  Super-linear volatility-of-variance $dv_t = \kappa v_t (\theta - v_t) dt + \epsilon v_t^{3/2} dW_t$, exact reciprocal CIR transformation $y_t = 1/v_t$ with automatic Feller satisfaction, Inverse-Gamma stationary law, and exact non-central chi-square path simulation.
- `zoo/beyond/schwartz_one_factor.py` — Schwartz One-Factor (1997) commodity diffusion model (`SchwartzOneFactor`, `ExponentialOU`, `Schwartz1F`):
  Mean-reverting log-spot Ornstein–Uhlenbeck dynamics, exact lognormal transition law and moments, closed-form commodity futures price term structure $F(t, T)$, and Black-76 European options on futures.
- `zoo/beyond/garch_diffusion.py` — Nelson (1990) continuous-time GARCH(1,1) diffusion limit (`GARCHDiffusion`, `ContinuousGARCH`, `GARCHDiffusionProcess`):
  Linear diffusion coefficient $g(v) = \sigma v$, exact Inverse-Gamma stationary distribution, conditional moments from linear moment ODE system, and guaranteed-positive log-Euler simulation.
- `zoo/beyond/bilinear.py` — Bilinear Diffusion process (`BilinearDiffusion`, `BilinearDiffusionProcess`, `AffineDiffusion`):
  Affine drift and affine volatility $dX_t = (a + b X_t) dt + (c + d X_t) dW_t$, Pearson diffusion family moments, exact second-moment ODE solution, and high-order Milstein boundary simulation.
- `vis/levy_inspector.py` — Lévy triplet diagnostic inspector and dual-panel density visualizer:
  Decomposes $(b, \sigma^2, \nu)$, classifies finite activity ($\int \nu(dx) < \infty$) vs infinite activity ($\nu(\mathbb{R}) = \infty$),
  finite vs infinite path variation, estimates Blumenthal-Getoor index $\beta \in [0, 2]$ and singularity order $s$,
  and plots near-zero singularity zoom vs tail decay on semi-log scale.
- `vis/surface.py` — 3D implied volatility surface and cross-model smile visualizer:
  Computes and renders 3D $(K, T, \sigma_{\mathrm{IV}})$ surfaces with isoline contours, cross-model smile comparison curves
  across diffusions, jumps, and stochastic volatility models, and volatility term structure plots.
- `vis/story_dag.py` — Process composition DAG visualizer and multi-format exporter:
  Walks `CompositionNode` trees and exports to ASCII / Unicode tree structures, Mermaid flowcharts for Markdown previews,
  Graphviz DOT files, and publication-quality hierarchical NetworkX + Matplotlib diagrams.
- `vis/__init__.py` — Unified visualization namespace exposing all diagnostic, surface, and DAG tools.
- `examples/visualization_example.py` — End-to-end demonstration script generating sample plots and diagrams.
- `tests/unit/test_vis.py` — Comprehensive unit test suite for all visualization submodules.
- `sim/milstein.py` — High-order 1D and multivariate commutative Milstein simulation schemes:
  Order 1.0 strong convergence with analytical or automatic centered finite-difference diffusion derivatives.
- `sim/runge_kutta.py` — Derivative-free Stochastic Runge-Kutta schemes:
  Strong order 1.0 Kloeden-Platen scheme (matching Milstein accuracy without diffusion derivatives) and weak order 2.0 scheme for expectation evaluation.
- `sim/qmc.py` — Quasi-Monte Carlo simulation engine:
  Sobol and Halton low-discrepancy sequences, Owen scrambling, and hierarchical Brownian bridge path construction.
- `sim/variance_reduction.py` — Comprehensive variance reduction toolkit:
  Optimal single/multivariate control variate regression estimator, antithetic Gaussian and Brownian paths, Girsanov importance sampling with Radon-Nikodym likelihood weights, and 1D stratified uniform sampling.
- `sim/predictor_corrector.py` — Drift-implicit predictor-corrector schemes for stiff SDEs:
  Unconditionally stable trapezoidal integration for stiff mean-reverting processes.
- `sim/mlmc.py` — Giles (2008) Multilevel Monte Carlo (MLMC) framework:
  Optimal $O(\epsilon^{-2})$ estimation complexity, adaptive level selection, and optimal variance allocation.
- `examples/simulation_backends_example.py` — Comprehensive demonstration of all advanced simulation backends.
- `zoo/beyond/cbi.py` — `CBISubordinator` (`CBIProcess`, `DawsonWatanabeProcess`, `ContinuousStateBranching`):
  Dawson-Watanabe continuous-state branching process with continuous and jump immigration (Kawazu & Watanabe 1971; Dawson & Li 2012):
  $dX_t = (a - b X_t) dt + \sqrt{c X_t} dW_t + dJ_t$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact closed-form Riccati solution $v(t, \lambda)$, affine Laplace transform $\mathbb{E}[e^{-\lambda X_t}]$,
  exact conditional mean and variance with compound Poisson immigration, analytical extinction probability for pure CSBP,
  and high-order Milstein simulation with non-negative reflection.
  References: Kawazu & Watanabe (1971); Dawson & Li (2012); Li (2011); Grey (1974).
- `sim/exact.py` — Vectorized `sample_cbi` Dawson-Watanabe Milstein sampler with jump immigration.
- `zoo/beyond/chen.py` — `ChenModel` (`ChenThreeFactor`, `ChenShortRate`, `ChenProcess`):
  Chen (1996) three-factor interest rate model with stochastic central tendency (mean) and stochastic variance:
  $dr_t = k (\theta_t - r_t) dt + \sqrt{\zeta_t r_t} dW_{1,t}$,
  $d\theta_t = \kappa (\bar{\theta} - \theta_t) dt + \alpha \sqrt{\theta_t} dW_{2,t}$,
  $d\zeta_t = \lambda (\bar{\zeta} - \zeta_t) dt + \beta \sqrt{\zeta_t} dW_{3,t}$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact closed-form expectations, Feller condition analytics, exact 10-dimensional matrix exponential moment system
  for second moments and conditional variance, term structure zero-coupon bond pricing and yield curve, and coupled 3-factor simulation.
  References: Chen (1996).
- `sim/exact.py` — Vectorized `sample_chen` coupled 3-factor path sampler.
- `zoo/beyond/regime_switching_gbm.py` — `RegimeSwitchingGBM` (`RSGBM`, `RegimeSwitchingBlackScholes`, `MarkovSwitchingGBM`):
  Continuous-time Markov-modulated Black-Scholes diffusion of Hamilton (1989), Naik (1993), and Guo (2001):
  $dS_t = \mu(Y_t) S_t dt + \sigma(Y_t) S_t dW_t$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact CTMC transition probabilities $P(t) = \exp(Q t)$, stationary probability distribution $\boldsymbol{\pi}$,
  arbitrary order exact matrix exponential moments $\mathbb{E}[S_t^n \mid S_0, Y_0]$, continuous characteristic function via
  matrix exponentials, closed-form Gil-Pelaez European call and put option pricing with machine-precision put-call parity,
  and coupled Markov-jump and diffusion path simulation.
  References: Hamilton (1989); Naik (1993); Guo (2001); Elliott, Aggoun, & Moore (1994).
- `sim/exact.py` — Vectorized `sample_regime_switching_gbm` coupled Markov chain and diffusion path sampler.
- `zoo/beyond/jacobi.py` — `JacobiProcess` (`JacobiDiffusion`, `WrightFisherDiffusion`, `Jacobi`):
  Jacobi bounded diffusion model on $[y_{\min}, y_{\max}]$ (canonical $[0, 1]$) of Ackerer, Filipović, & Turnes (2018):
  $dY_t = \kappa (\theta - Y_t) dt + \sigma \sqrt{\frac{(Y_t - y_{\min})(y_{\max} - Y_t)}{y_{\max} - y_{\min}}} dW_t$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact Beta stationary law $\mathrm{Beta}(\alpha, \beta)$, exact polynomial moment propagation (mean, second moment,
  and conditional variance), Feller boundary classifications at both lower and upper endpoints (entrance vs regular reflecting),
  and high-order Milstein path simulation with strict boundary preservation.
  References: Ackerer, Filipović, & Turnes (2018); Karlin & Taylor (1981); Filipović & Larsson (2016).
- `sim/exact.py` — Vectorized `sample_jacobi` Milstein path simulator for bounded intervals.
- `zoo/beyond/bns.py` — `BNSModel` (`BNSProcess`, `BarndorffNielsenShephard`, `BNS`):
  Non-Gaussian Ornstein-Uhlenbeck stochastic volatility model of Ole E. Barndorff-Nielsen & Neil Shephard (2001):
  $dX_t = (\mu + \beta \sigma_t^2) dt + \sigma_t dW_t + \rho dZ_{\lambda t}, \quad d\sigma_t^2 = -\lambda \sigma_t^2 dt + dZ_{\lambda t}$
  under `MOMENT_PROPAGATION` exactness level.
  Features Gamma-OU variance driven by a compound Poisson BDLP $Z_t$ with rate $\alpha$ and exponential jump size $\beta_{bdlp}$,
  exact stationary variance distribution $\Gamma(\alpha, \beta_{bdlp})$, exact conditional and integrated variance moments,
  continuous Fourier characteristic function $\phi(u; t)$, European call/put option pricing via Gil-Pelaez inversion with exact
  Put-Call parity, Black-Scholes implied volatility smile computation, and coupled vectorized simulation.
  References: Barndorff-Nielsen & Shephard (2001); Nicolato & Venardos (2003); Schoutens (2003).
- `sim/exact.py` — Vectorized `sample_bns` coupled asset price and stochastic variance simulator.
- `zoo/beyond/schwartz_two_factor.py` — `SchwartzTwoFactor` (`Schwartz97`, `Schwartz2F`, `SchwartzTwoFactorModel`):
  Canonical two-factor commodity pricing model of Eduardo Schwartz (1997) with mean-reverting convenience yield:
  $dS_t = (r - \delta_t) S_t dt + \sigma_1 S_t dW_1^Q(t), \quad d\delta_t = \kappa (\hat{\alpha} - \delta_t) dt + \sigma_2 dW_2^Q(t)$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact closed-form futures pricing $F(S_t, \delta_t, \tau)$ via Schwartz (1997) Proposition 1 eq. (12),
  exact conditional Gaussian moments for convenience yield $\delta_t$ and log-spot $X_t = \ln S_t$,
  exact characteristic function $\phi(u; t)$, analytical Black-76 European call and put options with exact
  Put-Call parity, and joint vectorized path simulation of spot prices and convenience yields.
  References: Schwartz (1997); Bjerksund (1991).
- `sim/exact.py` — Vectorized `sample_schwartz_two_factor` joint spot and convenience yield simulator.
- `zoo/beyond/jcir.py` — `JCIRProcess` (`JCIR`, `JumpCIRProcess`, `JCIRModel`):
  Jump-Extended Cox-Ingersoll-Ross short-rate model of Duffie & Garleanu (2001) and Maghsoodi (1996):
  $dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t + dJ_t, \quad r_0 \ge 0$
  under `MOMENT_PROPAGATION` exactness level.
  Features non-negative interest rates with compound Poisson exponential jump shocks ($Y_i \sim \mathrm{Exp}(\mu_J)$),
  exact stationary mean $\theta_\infty = \theta + (\lambda_J \mu_J)/\kappa$ and stationary variance, exact conditional
  moments at any horizon, closed-form exponential-affine zero-coupon bond pricing $P(t, T) = \exp(A(\tau) - B(\tau) r_t)$
  with exact analytical Riccati integration, term structure yields, analytical instantaneous forward rates, exact
  conditional characteristic function, and exact simulation combining non-central $\chi^2$ CIR transitions with compound
  Poisson jump increments.
  References: Duffie & Garleanu (2001); Maghsoodi (1996); Brigo & Mercurio (2006).
- `sim/exact.py` — Vectorized `sample_jcir` exact non-central chi-squared + compound Poisson jump simulator.
- `zoo/beyond/bessel.py` — `SquaredBesselProcess` (`BESQ`, `SquaredBessel`) and `BesselProcess` (`BES`, `Bessel`):
  Canonical squared Bessel and Bessel diffusion processes of Feller (1951), Pitman & Yor (1982), and Revuz & Yor (1999):
  $dX_t = \delta dt + 2\sqrt{X_t} dW_t, \quad R_t = \sqrt{X_t}, \quad dR_t = \frac{\delta - 1}{2 R_t} dt + dW_t$
  under `MOMENT_PROPAGATION` exactness level.
  Features exact transition density and CDF via non-central $\chi^2$ distribution $\frac{X_t}{t} \sim \chi^2(\delta, \frac{x_0}{t})$,
  exact closed-form cumulants $\kappa_1, \kappa_2, \kappa_3, \kappa_4$ to arbitrary order, exact Laplace transform
  $\mathbb{E}[e^{-\lambda X_t}] = (1 + 2\lambda t)^{-\delta/2} \exp\left(-\frac{\lambda x_0}{1 + 2\lambda t}\right)$,
  exact continuous characteristic function $\phi(u; t)$, Feller boundary classification at 0 (entrance for $\delta \ge 2$,
  reflecting for $0 < \delta < 2$, absorbing for $\delta = 0$), exact Bessel process confluent hypergeometric moments
  $\mathbb{E}[R_t]$, and exact path simulation with zero time discretization error.
  References: Feller (1951); Pitman & Yor (1982); Revuz & Yor (1999).
- `sim/exact.py` — Vectorized `sample_besq` and `sample_bessel` exact path simulators using scaled non-central chi-squared increments.
- `zoo/beyond/schobel_zhu.py` — `SchobelZhuProcess` (`SchobelZhu`, `SchobelZhuModel`, `SZModel`):
  Schöbel-Zhu (1999) stochastic volatility model:
  $dS_t = (r - q) S_t dt + \sigma_t S_t dW_t^S, \quad d\sigma_t = \kappa (\theta - \sigma_t) dt + \tau dW_t^\sigma$
  under `MOMENT_PROPAGATION` exactness level.
  Features Ornstein-Uhlenbeck Gaussian volatility dynamics with non-negative instantaneous variance
  $v_t = \sigma_t^2$, exact continuous characteristic function, Gil-Pelaez European Call and Put pricing
  satisfying exact Put-Call parity, Black-Scholes implied volatility smile inversion, and coupled
  vectorized path simulation.
  References: Schöbel & Zhu (1999); Lord & Kahl (2007).
- `sim/exact.py` — Vectorized `sample_schobel_zhu` simulator supporting coupled asset and volatility paths.
- `zoo/beyond/black_karasinski.py` — `BlackKarasinskiProcess` (`BlackKarasinski`, `BKProcess`, `BKModel`):
  Black-Karasinski (1991) lognormal short-rate model:
  $d(\ln r_t) = a(b - \ln r_t) dt + \sigma dW_t, \quad r_t = \exp(x_t) > 0$ under `MOMENT_PROPAGATION`
  exactness level.
  Guarantees strictly positive interest rates across all horizons. Features exact Ornstein-Uhlenbeck
  Gaussian conditional moments for log-rate $x_t = \ln(r_t)$, exact short-rate lognormal conditional
  and stationary moments, Monte Carlo zero-coupon bond pricing $P(0, T)$ and yield curve $y(0, T)$,
  and exact path simulation without discretization error for $x_t$.
  References: Black & Karasinski (1991); Brigo & Mercurio (2006).
- `sim/exact.py` — Vectorized `sample_black_karasinski` simulator supporting exact log-rate transitions.
- `zoo/levy/gh.py` — `GeneralizedHyperbolicProcess` (`GH`, `GHProcess`, `GeneralizedHyperbolic`):
  Generalized Hyperbolic (GH) Lévy process of Barndorff-Nielsen (1977) and Eberlein & Prause (2002):
  infinite-activity pure-jump Lévy process generated by the 5-parameter Generalized Hyperbolic
  distribution $GH(\lambda, \alpha, \beta, \delta, \mu)$ under `EXACT` exactness level.
  Features exact characteristic exponent $\psi(u)$ with modified Bessel $K_\lambda$ functions,
  exact cumulants $\kappa_1, \dots, \kappa_4$ derived from underlying GIG subordinator moments,
  Lévy triplet $(b, 0, \nu)$, Gil-Pelaez European Call and Put pricing with exact Put-Call parity,
  implied volatility smile inversion, subfamily classification (NIG, Hyperbolic, Hyperbolic Cosine),
  and vectorized simulation via exact Inverse Gaussian subordination and Gil-Pelaez CDF inversion.
  References: Barndorff-Nielsen (1977); Barndorff-Nielsen & Halgreen (1977); Eberlein & Prause (2002).
- `sim/exact.py` — Vectorized `sample_gh` simulator supporting exact NIG subordination and Gil-Pelaez inversion.
- `zoo/beyond/sabr.py` — `SABRProcess` (`SABRModel`, `SABR`):
  SABR (Stochastic Alpha, Beta, Rho) stochastic volatility model of Hagan et al. (2002):
  $dF_t = \alpha_t F_t^\beta dW_t^F, \quad d\alpha_t = \nu \alpha_t dW_t^\alpha$ with
  $\mathbb{E}[dW_t^F dW_t^\alpha] = \rho dt$ under `MOMENT_PROPAGATION` exactness level.
  Features Hagan et al. (2002) asymptotic Black-76 implied volatility closed form with Obłój (2008)
  corrections, at-the-money (ATM) volatility formula, Black-76 European Call and Put pricing
  satisfying Put-Call parity, exact forward and log-normal volatility moments, smile backbone
  classification, and coupled vectorized simulation via log-Euler and Euler schemes.
  References: Hagan, Kumar, Lesniewski, & Woodward (2002); Obłój (2008).
- `sim/exact.py` — Vectorized `sample_sabr` simulator supporting coupled forward price and stochastic volatility paths.
- `zoo/beyond/hull_white.py` — `HullWhiteProcess` (`HullWhite`, `HW1F`, `ExtendedVasicek`):
  Hull-White (1990) one-factor Gaussian short-rate model (extended Vasicek):
  $dr_t = a(b - r_t) dt + \sigma dW_t$ with mean-reversion speed $a > 0$, volatility $\sigma > 0$,
  and long-term target rate $b \in \mathbb{R}$ under `MOMENT_PROPAGATION` exactness level.
  Features exact zero-coupon bond pricing $P(t, T) = A(t, T) \exp(-B(t, T) r_t)$, spot yield curve,
  instantaneous forward rates, conditional short rate and integrated rate moments, autocovariance,
  Jamshidian (1989) closed-form European bond Call and Put options with exact Put-Call parity,
  caplets and floorlets, and exact path simulation without time discretization error.
  References: Hull & White (1990, 1993); Jamshidian (1989); Vasicek (1977).
- `sim/exact.py` — Vectorized `sample_hull_white` simulator with exact Gaussian transitions and Euler fallback.
- `zoo/beyond/bates.py` — `BatesProcess` (`BatesModel`, `Bates96`, `SVJ`):
  Bates (1996) stochastic volatility with jumps model coupling the Heston (1993) CIR variance diffusion
  with Merton (1976) log-normal price jumps:
  $dS_t = (\mu - q - \lambda \bar{k}) S_t dt + \sqrt{V_t} S_t dW_t^S + S_{t-} (e^Y - 1) dN_t$,
  $dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} dW_t^V$ with $\mathbb{E}[dW^S dW^V] = \rho dt$
  under `MOMENT_PROPAGATION` exactness level.
  Features continuous Albrecher et al. (2007) characteristic function factored with compound Poisson jumps,
  drift compensator $-\lambda \bar{k}$, high-precision cumulants $\kappa_1, \dots, \kappa_4$,
  European Call and Put pricing via Gil–Pelaez Fourier inversion, exact Put-Call parity,
  Black–Scholes implied volatility smile inversion, and vectorized path simulation via Lord et al. (2010)
  full-truncation CIR scheme and Poisson jump shocks.
  References: Bates (1996); Heston (1993); Albrecher et al. (2007); Lord et al. (2010).
- `sim/exact.py` — Vectorized `sample_bates` path simulator supporting both log-return $X_t$ and instantaneous variance $V_t$.
- `zoo/levy/kou.py` — `KouJumpDiffusion` (`KouProcess`, `Kou72`, `Kou02`):
  Kou (2002) double exponential jump-diffusion process combining continuous Brownian motion with
  asymmetric double exponential jump shocks: $X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i$,
  $f_Y(y) = p \eta_1 e^{-\eta_1 y}\mathbb{I}_{y \ge 0} + (1-p) \eta_2 e^{\eta_2 y}\mathbb{I}_{y < 0}$
  under `EXACT` exactness level.
  Features exact closed-form Lévy triplet $(b, \sigma^2, \nu)$ with analytical truncated drift integral,
  characteristic exponent $\psi(u)$, analytical cumulants of all orders $n$,
  European Call and Put option pricing via Gil–Pelaez / Carr–Madan Fourier inversion,
  Put-Call parity, Black–Scholes implied volatility smile/skew inversion, and exact
  simulation with zero time discretization error.
  References: Kou (2002); Cont & Tankov (2004).
- `sim/exact.py` — Vectorized `sample_kou` exact path simulator with zero discretization error.
- `zoo/beyond/cev.py` — `CEVProcess` (`CEV`, `ConstantElasticityOfVariance`):
  Constant Elasticity of Variance (CEV) local volatility diffusion process
  $dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t, S_0 = s_0 > 0$ under `MOMENT_PROPAGATION` exactness.
  Features state-dependent local return volatility $\sigma_{\text{loc}}(S) = \sigma S^{\gamma - 1}$,
  constant return variance elasticity $2(\gamma - 1)$, boundary classification with absorbing zero
  (default / bankruptcy risk) for $\gamma < 1$, exact European option pricing via Schroder's (1989)
  non-central chi-squared representation across equity leverage ($\gamma < 1$), geometric Brownian motion
  ($\gamma = 1$), and commodity inverse leverage ($\gamma > 1$) regimes, exact Put-Call parity,
  analytical absorption probability $P(S_T = 0) = 1 - F_{\chi^2}(x; 1/(1-\gamma))$, transition CDF,
  analytical Greeks (Delta, Gamma, Vega), conditional moments (closed-form for $\gamma \in \{0.5, 1.0\}$ and
  quadrature propagation for general $\gamma$), and vectorized path simulation.
  References: Cox (1975); Schroder (1989); Emanuel & MacBeth (1982); Choi & Wu (2021).
- `sim/exact.py` — Vectorized `sample_cev` simulation routine supporting second-order Milstein
  and Euler–Maruyama schemes with absorbing boundary conditions at zero and exact geometric Brownian motion
  for $\gamma = 1$.
- `zoo/levy/merton.py` — `MertonJumpDiffusion` (`MertonProcess`, `Merton76`):
  Merton (1976) jump-diffusion process combining continuous Brownian motion with
  normally distributed compound Poisson jump shocks: $X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i$,
  $Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$ under `EXACT` exactness level.
  Features exact Lévy triplet $(b, \sigma^2, \nu)$ with analytical truncated Gaussian drift integral,
  closed-form characteristic exponent, analytical cumulants of all orders $n$,
  Merton (1976) analytical option pricing formula via Poisson mixture of Black–Scholes values,
  Put-Call parity, Black-Scholes implied volatility smile/skew inversion, and exact
  vectorized simulation with zero time discretization error.
  References: Merton (1976); Cont & Tankov (2004).
- `sim/exact.py` — Vectorized `sample_merton` exact path simulator with zero discretization error.
- `zoo/beyond/heston.py` — `HestonProcess` (`Heston`):
  Continuous-time affine stochastic volatility diffusion model
  $dX_t = (\mu - \frac{1}{2}V_t)dt + \sqrt{V_t} dW_t^S$,
  $dV_t = \kappa(\theta - V_t)dt + \sigma_v \sqrt{V_t} dW_t^V$ with
  $d\langle W^S, W^V \rangle_t = \rho dt$ under `MOMENT_PROPAGATION` exactness.
  Features exact characteristic functions via Albrecher et al. (2007) continuous formulation,
  CIR instantaneous and integrated variance moments, Feller condition verification,
  analytical cumulants, European Call and Put option pricing via Gil-Pelaez Fourier inversion,
  Black-Scholes implied volatility smile/skew inversion via root-finding, and Full Truncation
  simulation (Lord et al. 2010).
  References: Heston (1993); Albrecher et al. (2007); Lord et al. (2010).
- `sim/exact.py` — Vectorized `sample_heston` simulation routine supporting
  Lord et al. (2010) Full Truncation scheme and truncated Euler scheme.
- `zoo/beyond/ornstein_uhlenbeck.py` — `OrnsteinUhlenbeckProcess` (`OrnsteinUhlenbeck`, `Vasicek`):
  Continuous-time Gaussian mean-reverting diffusion $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$
  with `MOMENT_PROPAGATION` exactness level. Features exact Gaussian transition density, exact
  conditional moments and cumulants, stationary distribution $N(\mu, \sigma^2/(2\theta))$,
  relaxation time $\tau = 1/\theta$, half-life $t_{1/2} = \ln(2)/\theta$, exponential autocovariance
  and autocorrelation, two-time covariance, Lorentzian power spectral density $S(\omega) = \sigma^2/(\pi(\theta^2+\omega^2))$,
  exact Laplace transform and characteristic function, Vasicek zero-coupon bond pricing $P(\tau) = A(\tau)e^{-B(\tau)r_0}$,
  yield curves, and asymptotic long-term yield.
  References: Uhlenbeck & Ornstein (1930); Vasicek (1977).
- `sim/exact.py` — Vectorized `sample_ou` supporting exact Gaussian increments without
  discretization bias (`method="exact"`) and Euler–Maruyama discretisation (`method="euler"`).
- `zoo/beyond/cir.py` — `CIRProcess` (`CoxIngersollRoss`): Cox–Ingersoll–Ross square-root
  diffusion process $dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t$ with `MOMENT_PROPAGATION`
  exactness level. Features Feller condition evaluation ($2\kappa\theta \ge \sigma^2$), exact non-central
  chi-squared transition density and exact simulation via `sample_cir`, exact conditional moments and
  cumulants of all orders, stationary Gamma distribution properties, exponential autocovariance,
  exact Laplace transform and characteristic function, zero-coupon bond pricing $P(\tau) = A(\tau)e^{-B(\tau)r_0}$
  via the Riccati system, yield curves, and asymptotic long-term yield.
  References: Cox, Ingersoll & Ross (1985); Feller (1951).
- `sim/exact.py` — Vectorized `sample_cir` supporting exact non-central chi-squared transition
  sampling (`method="exact"`), full-truncation Euler–Maruyama (`method="truncated_euler"`), and
  reflected Euler–Maruyama (`method="reflected_euler"`).
- `zoo/levy/compound_poisson.py` — `CompoundPoisson`: general compound Poisson process
  with homogeneous Poisson arrival counter and parameterised jump distributions
  (`NormalJump`, `ExponentialJump`, `DoubleExponentialJump`, `UniformJump`, `DiscreteJump`).
  `EXACT` exactness level with closed-form Lévy triplet `(b=λ·E[Y·𝟙_{|Y|≤1}], σ²=0, ν=λ·F_Y)`,
  exact characteristic function `φ(u;t) = exp(λt(φ_Y(u)-1))`, exact cumulants of all orders
  `κ_n = λ·t·E[Yⁿ]`, and vectorized exact Poisson thinning simulation.
  References: Kingman (1993); Cont & Tankov (2004) Sec. 3.4; Sato (1999) Ex. 8.4.
- `sim/exact.py` — Vectorized `sample_compound_poisson` using `np.repeat` and `np.bincount`
  with fallback for non-vectorized custom samplers.
- `zoo/beyond/mmpp.py` — `MarkovModulatedPoissonProcess`: N-state MMPP (CTMC-modulated Poisson
  counts). `MOMENT_PROPAGATION`; no Levy triplet (dependent increments). Exact event-driven
  simulation, finite-horizon mean/variance and cumulants of any order via block matrix
  exponentials (with a conditioning guard), stationary distribution and mean intensity.
  Reference: Fischer & Meier-Hellstern (1993), Performance Evaluation 18, 149-171.

### Changed

- Minimum NumPy raised to `numpy>=2.0`. The library uses `np.trapezoid`, which does not exist in
  NumPy 1.x, so the old `numpy>=1.24` lower bound allowed broken installs.

### Fixed

- `rough/riccati.py` — `FractionalRiccatiSolver`: corrected the sign of the linear Riccati
  coefficient to $(i\rho\nu u - \lambda)$, and the characteristic function now uses
  $V_0 I^{1-\alpha}h(T) = V_0\int_0^T F(u, h(s))\,ds$ (El Euch & Rosenbaum 2019). It reduces to
  the classical Heston closed form at $H = 1/2$.
- `rough/rbergomi.py` — `RoughBergomi` simulates the Riemann–Liouville driver and its Brownian
  motion jointly and exactly (Cholesky), so $\rho$ now correlates returns with variance and
  $\mathbb E[v_t] = \xi_0$. `theoretical_short_time_skew` gains the missing $1/(H + 3/2)$ factor
  (Fukasawa). Added `FractionalKernel.volterra_joint_covariance` and `sample_volterra_cholesky`.
- `signature/kernel.py` — the Goursat solver weighted the increment term by 5/4. It is now
  the second-order trapezoidal scheme, converging to $I_0(2)$ for $X = Y = t$.
- `signature/algebra.py` — level-3 log-signature Hall coordinates were twice their true value.
  `log_signature_dimension` computes the Möbius function exactly; the old lookup table stopped
  at $n = 6$, which gave wrong Witt dimensions from depth 10.
- `calibration/surface/{heston,bates,double_heston}.py` — the share-measure probability $P_1$
  normalised $\varphi(u - i)$ by $S_0$ instead of $\varphi(-i) = S_0e^{rT}$, so every call price
  was wrong when $r \ne 0$. Regression tests now check against the Black–Scholes and Merton limits.
- `calibration/filter/particle_filter.py` — `BootstrapParticleFilter` discarded the previous
  weights on steps without resampling. Weights are now carried forward, and the incremental
  likelihood is $\log\sum_i w_{k-1}^i p(y_k \mid x_k^i)$. Checked against the exact Kalman
  log-likelihood.
- `calibration/mle/hac.py` — `newey_west_hac(kernel="quadratic_spectral")` silently used
  Bartlett weights. It now implements the Andrews (1991) kernel and rejects unknown kernels.
- `scripts/extract_changelog.py` — the release workflow's note extractor did not match
  bracketed Keep a Changelog headings (`## [X.Y.Z] — date`), so GitHub releases got the
  placeholder body `Release X.Y.Z`.

### Documentation

- `LICENSE` (MIT), `CITATION.cff`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and
  `SECURITY.md`.
- New theory pages: information geometry, rough paths & signatures, rough volatility,
  copulas, stochastic control, and estimation & calibration.
- README: Model Morphing quickstart and corrected examples and API names.
- Added the missing `docs/javascripts/mathjax.js` (math now renders on the docs site) and
  removed the `polyfill.io` script.

## [1.0.0] — 2026-09-19

Initial public release. Establishes the algebraic core, the 13-process Lévy zoo, the beyond-Lévy
category, the ops/sim/analytics layers, multivariate support, and the story renderer.

### Added — Core engine

- `core/exactness.py` — `ExactnessLevel` enum (`EXACT` / `MOMENT_PROPAGATION` / `SIMULATION_ONLY`)
  with total ordering and `combine()` rule: operations degrade to the minimum level of their operands
- `core/exceptions.py` — `SpxaError`, `ExactnessError`, `SpxaDegradationWarning`
- `core/levy_measure.py` — `LevyMeasure` hierarchy: `CompoundLevyMeasure`, `DensityLevyMeasure`,
  `ScaledLevyMeasure`
- `core/triplet.py` — immutable `LevyTriplet(b, σ², ν)` with arithmetic (`__add__`, `.scale()`),
  numerical `char_exp()` via the Lévy–Khintchine formula, and `cumulant(n)` (Sato 1999, Thm 25.3)
- `core/properties.py` — `ProcessProperties` lattice tracking stationarity, independence,
  martingale, finite mean/variance, tail index, self-similarity, subordinator flag, Hurst index;
  propagation rules for `+`, scaling, and `@` derived from Sato (1999) Ch. 11 & 30
- `core/process.py` — abstract `Process` base class; operator overloads `+`, `*`, `@`, `-`,
  unary `-`; each returns a new immutable `_ComposedProcess` or `_ScaledProcess` instance;
  `.triplet`, `.cumulants()`, `.char_func()`, `.__story__()`, `.__story_latex__()`
- `story/narrator.py` — `CompositionNode` tree, `narrate()` plain-text derivation engine,
  `narrate_latex()` / `story_latex()` MathJax-ready LaTeX renderer

### Added — Process zoo (`spxa.zoo.levy`)

- `BrownianMotion` — triplet `(μ, σ², 0)`, exact Gaussian simulation, closed-form `char_func_exact()`
- `PoissonProcess` — discrete Lévy measure, exact inter-arrival simulation, `char_func_exact()`
- `GammaProcess` — subordinator, exact `bernstein_function()` `a·log(1+u/b)` (Schilling et al.
  2012, Ex. 3.9), exact cumulants, Gamma simulation
- `VarianceGamma` — Lévy density `C/|x|·exp(Ax−B|x|)`, closed-form cumulants to order 4 and
  series above, exact `char_func_exact()` (Madan, Carr & Chang 1998, eq. 4), exact simulation
  via Gamma subordination
- `NIG` — Bessel Lévy density, exact `char_func_exact()`, closed-form cumulants,
  simulation via InvGaussian subordination (Barndorff-Nielsen 1997)
- `AlphaStable` — exact char_func, self-similarity index `1/α`, Chambers–Mallows–Stuck simulation
  (Samorodnitsky & Taqqu 1994)
- `CGMY` — exact char_func, closed-form cumulants finite for `α < 2`, series-representation
  simulation (Carr, Geman, Madan, Yor 2002; Rosiński 2001)
- `InverseGaussianProcess` — exact Bernstein function, cumulants `(2n−1)!!·μ^{2n-1}/λ^{n-1}`,
  exact `char_func`, Wald simulation (Tweedie 1957)
- `MeixnerProcess` — stable log-cosh characteristic function, exact cumulants to order 4,
  Gil-Pelaez CDF-inversion simulation (Schoutens & Teugels 1998)
- `TemperedStable` — Lévy density `C·x^{-1−α}·exp(−λx)`, exact Bernstein function, cumulants
  `C·Γ(n−α)/λ^{n−α}`, CDF-inversion simulation (Rosiński 2007)
- `NegativeBinomialProcess` — finite-activity discrete subordinator, exact characteristic
  function, closed-form cumulants to order 4, exact NegBin sampler (Quenouille 1949)

### Added — Beyond-Lévy (`spxa.zoo.beyond`)

- `FractionalBrownianMotion` — `MOMENT_PROPAGATION`; Hurst index algebra, Davies–Harte
  simulation; raises `ExactnessError` on `.triplet` (not a semimartingale)
- `HawkesProcess` — `MOMENT_PROPAGATION`; mean intensity, Fano factor, Ogata thinning
  simulation (Hawkes 1971)
- `OULevy` — `MOMENT_PROPAGATION`; stationary mean/variance, autocovariance, exact
  discretisation simulation (Barndorff-Nielsen & Shephard 2001)

### Added — Multivariate (`spxa.zoo.multivariate`)

- `MultivariateBrownianMotion`, `MultivariateLevyTriplet`, `CorrelatedLevy`,
  `correlated_brownian_motion()`, `independent_levy_vector()`

### Added — Operations (`spxa.ops`)

- `addition.py` — `add()`, `sum_processes()`, `triplet_add()`, `exactness_for_addition()`
- `scaling.py` — `scale()`, `negate()`, `subtract()`, `triplet_scale()`, `linear_transform_triplet()`
- `subordination.py` — `subordinate()`, `char_exp_subordinated()`, `bernstein_function()`,
  subordinator validity check
- `product.py` — `ProductProcess`, cumulant/moment conversion, moment-product formula
- `integral.py` — Itô isometry, stochastic convolution char_func, `StochasticIntegral` with
  Euler–Maruyama simulation

### Added — Simulation (`spxa.sim`)

- `euler.py` — Euler–Maruyama scheme, autonomous SDE solver, geometric Lévy, convergence-order
  estimator
- `exact.py` — exact samplers for BM, Gamma, InvGaussian, VG, NIG, stable (CMS),
  compound Poisson, tempered stable
- `bridge.py` — Brownian bridge, Gamma bridge, bridge interpolation, first-passage time
  (simulated and exact)

### Added — Analytics (`spxa.analytics`)

- `divergence.py` — L2, Hellinger, Wasserstein-2, Monte Carlo KL divergence between process
  marginals
- `fourier.py` — Carr–Madan FFT option pricer, put-call parity, implied-vol via bisection;
  routes through `char_func_exact` when available (~1000× speedup for VG/NIG/BM)
- `cumulants.py` — `cumulant_table()`, `skewness()`, `excess_kurtosis()`

### Added — Tests

- `tests/exactness/` — per-process suites verifying closed-form results against published
  formulas: BrownianMotion, GammaProcess, VarianceGamma, NIG, AlphaStable, CGMY,
  InverseGaussianProcess, MeixnerProcess, TemperedStable, NegativeBinomialProcess,
  fBM/Hawkes/OULevy (268 tests total, marked `exactness` — never skippable)
- `tests/unit/` — core engine, ops, sim, multivariate (covering ExactnessLevel, triplet
  arithmetic, composition, properties, story output, all sim algorithms)
- `tests/integration/` — 17 end-to-end composition and analytics tests

### Added — Documentation and tooling

- Three executable Jupyter tutorials: first composition, finance models, physics applications
- `PROCESS_REGISTRY.md` — full table of implemented processes with exactness levels, cumulant
  availability, simulation algorithms, and primary references
- `CONTRIBUTING.md` — contribution guide covering all nine contribution types with exact
  per-type file checklists, commit conventions, versioning policy, and RFC process for
  core changes
- `examples/demo.py`, `examples/full_showcase.py` (100+ checks across all 12 capability areas)
- GitHub Actions workflows: `ci.yml`, `docs.yml`, `benchmarks.yml`, `release.yml`
- `.pre-commit-config.yaml`, `.gitignore`
- `pyproject.toml`: ruff (linting), mypy strict, pytest with `exactness` and `slow` markers,
  coverage config

### Fixed

- `core/triplet.char_exp`: split-integral SciPy compatibility (no `points` kwarg with infinite
  limits)
- `zoo/vg`: triplet drift singularity — switched to tail integrals on `[1, ∞)` anchored to the
  exact mean; `cumulants()` and `char_func()` overridden with exact closed-form formulas
- `zoo/gamma`: triplet drift replaced from quadrature to exact exponential integral `E₁(b)`
- `zoo/nig`: `char_func()` overridden to route through `char_func_exact`, eliminating
  Lévy–Khintchine numerical overflow at large `|u|`
- `zoo/beyond/fbm`: Davies–Harte circulant embedding off-by-one and broadcast error
- `sim/euler.geometric_levy`: switched to MGF-based drift correction for non-Gaussian drivers
- `ops/bernstein_function`: stabilised elementwise computation, avoiding `log(0)`
- `core/_ComposedProcess.simulate`: subordination (`@`) implemented via path interpolation
- `analytics/fourier`: replaced deprecated `np.trapz` with `np.trapezoid` (NumPy 2.x)
- `pyproject.toml`: corrected build backend to `setuptools.build_meta`
