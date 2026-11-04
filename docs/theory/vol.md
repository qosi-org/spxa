# Implied Volatility Surfaces

`spxa.vol` builds implied-volatility surfaces that are free of static arbitrage. This chapter covers the parametric layer (SVI slices, the SSVI and eSSVI surfaces, their calibration, interpolation in time, and the risk-neutral distributions they imply), and then the surface object built on it from option quotes. Option formulas, implied-volatility inversion and quote handling come from [Market conventions](market.md).

| Object | Implementation |
| :--- | :--- |
| Coordinates | `log_moneyness`, `total_variance`, `normalised_call_from_total_variance`, `total_variance_from_price` |
| SVI slices | `SVIRaw`, `SVINatural`, `SVIJumpWings` |
| Surfaces | `SSVI`, `HestonLikePhi`, `PowerLawPhi`, `ESSVI`, `ESSVISlice` |
| Calibration | `calibrate_svi`, `calibrate_svi_butterfly_free`, `calibrate_svi_surface`, `calibrate_ssvi`, `calibrate_essvi` |
| Diagnostics | `durrleman_g`, `check_butterfly`, `check_calendar`, `wing_asymptotics`, `rogers_tehranchi_slope` |
| Interpolation and densities | `PriceInterpolatedSurface`, `ImpliedDistribution`, `risk_neutral_density` |
| Surface objects | `VolSurface`, `ForwardTermStructure`, `VarianceInterpolatedSurface` |
| Building from quotes | `build_vol_surface`, `SurfaceBuild`, `SurfaceQuality`, `chain_total_variance` |
| Price-level repair and smoothing | `repair_call_prices`, `call_arbitrage_constraints`, `fengler_smooth`, `fengler_surface` |
| Delta conventions | `black_delta`, `strike_from_delta`, `atm_dns_strike` |
| Risk and persistence | `parallel_bump`, `skew_bump`, `expiry_buckets`, `bucketed_vega`, `dumps`, `loads` |
| Local volatility | `dupire_local_variance`, `LocalVolSurface`, `LocalVolGrid`, `bbf_implied_vol` |
| Andreasen–Huge | `AndreasenHugeSurface`, `calibrate_andreasen_huge`, `AndreasenHugeFit`, `andreasen_huge_grid`, `piecewise_constant_vol` |
| Local-vol simulation and checks | `local_vol_paths`, `LocalVolPaths`, `local_vol_diagnostics`, `LocalVolDiagnostics` |
| Stochastic local volatility | `Heston`, `slv_grid`, `LeverageSurface`, `slv_forward`, `calibrate_leverage`, `calibrate_leverage_particles`, `slv_paths`, `SLVPDE`, `calibrate_slv`, `as_local_vol` |

---

## 1. Coordinates and static arbitrage

Surfaces are written in log-forward moneyness and total implied variance,

$$
k = \log\frac{K}{F_T}, \qquad w(k, T) = \sigma_{\text{BS}}^2(k, T)\,T,
$$

and the undiscounted call price over the forward is $c(k, w) = \Phi(d_+) - e^k\Phi(d_-)$ with $d_\pm = -k/\sqrt w \pm \sqrt w/2$.

**Butterfly arbitrage.** A slice $w(k)$ is free of butterfly arbitrage if and only if the density of $X = \log(S_T/F_T)$,

$$
p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}\,e^{-d_-(k)^2/2}, \qquad
g(k) = \Bigl(1 - \frac{k w'}{2w}\Bigr)^2 - \frac{w'^2}{4}\Bigl(\frac1w + \frac14\Bigr) + \frac{w''}{2},
$$

is non-negative and call prices vanish as $k \to \infty$ (Gatheral & Jacquier 2014, Lemma 2.2). Here $g$ is Durrleman's function. `check_butterfly` scans $g$ on a grid and reports its minimum and the intervals where it is negative.

**Calendar arbitrage.** With proportional dividends, a surface is free of calendar-spread arbitrage if and only if $T \mapsto w(k, T)$ is non-decreasing for every $k$ (Lemma 2.1). `check_calendar` reports the worst decrease between consecutive slices.

**Wings.** Lee's moment formula links the wing slopes $\beta_R = \limsup_{k\to\infty} w/k$ and $\beta_L = \limsup_{k\to-\infty} w/|k|$ to the critical moments $\tilde p = \sup\{p : E[S^{1+p}] < \infty\}$ and $\tilde q = \sup\{q : E[S^{-q}] < \infty\}$:

$$
\beta = 2 - 4\bigl(\sqrt{x^2 + x} - x\bigr), \qquad x = \frac{(2 - \beta)^2}{8\beta},
$$

so no wing can be steeper than 2. With $w \approx \beta|k|$, the density above decays like $e^{-\lambda|k|}$ with $\lambda_R = 1 + \tilde p$ and $\lambda_L = \tilde q$. The tests check both rates from log-density slopes at $|k| = 2000$ to $4000$. They also show that $E[S^{1+p}]$ converges just below $\tilde p$ and diverges just above it. Separately, an arbitrage-free surface has $|\partial_k w| \le 4$ (Rogers & Tehranchi 2010).

---

## 2. SVI slices

The raw SVI slice of Gatheral (2004) is

$$
w(k) = a + b\Bigl(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\Bigr),
$$

with minimum $a + b\sigma\sqrt{1 - \rho^2}$ at $k^* = m - \rho\sigma/\sqrt{1 - \rho^2}$ and wing slopes $b(1 \pm \rho)$. The natural parametrisation $(\Delta, \mu, \rho, \omega, \zeta)$ maps to raw SVI by

$$
a = \Delta + \tfrac{\omega}{2}(1 - \rho^2), \quad b = \tfrac{\omega\zeta}{2}, \quad m = \mu - \rho/\zeta, \quad \sigma = \sqrt{1 - \rho^2}/\zeta.
$$

The jump-wings parametrisation $(v_t, \psi_t, p_t, c_t, \tilde v_t)$ uses the ATM variance, ATM skew, put and call wing slopes and minimum variance, all of which scale sensibly across expiries.

Not every raw SVI slice is arbitrage-free. Axel Vogt's slice $(a, b, \rho, m, \sigma) = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)$ has $\min g = -0.032864$ at $k = 0.879$. Its jump-wings parameters at $t = 1$ are $(0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249)$, as in Gatheral & Jacquier (2014, Example 3.1). There are two repairs:

* **Projection** (Gatheral & Jacquier, §5.1). Keep $(v, \psi, p)$ and set $c' = p + 2\psi$ and $\tilde v' = 4vpc'/(p + c')^2$. The result is an SSVI slice with the same ATM level and skew. For Vogt's slice $c' = 0.3493158$.
* **Constrained fit.** `calibrate_svi_butterfly_free` imposes $g \ge 10^{-8}$ on a grid. Wherever $g$ dips between grid nodes, it adds cutting planes at the local minima. Refitting Vogt's slice on 61 points of $[-1.5, 1.5]$ gives an arbitrage-free slice with rmse $0.00231$ in total variance.

`calibrate_svi` is the quasi-explicit method of Zeliade and De Marco & Martini. For fixed $(m, \sigma)$ the slice is linear in $(a, b\sigma, \rho b\sigma)$ on a box, so the inner problem is an exact bounded least-squares solve and only $(m, \sigma)$ are searched. It recovers Vogt's raw parameters exactly from noiseless quotes.

---

## 3. SSVI and eSSVI

**SSVI** (Gatheral & Jacquier 2014) writes the whole surface through the ATM total variance $\theta_t$:

$$
w(k, \theta_t) = \frac{\theta_t}{2}\Bigl(1 + \rho\varphi(\theta_t)k + \sqrt{(\varphi(\theta_t)k + \rho)^2 + 1 - \rho^2}\Bigr).
$$

The surface is free of calendar arbitrage when $\theta_t$ is non-decreasing and $0 \le \partial_\theta(\theta\varphi) \le \frac{1 + \sqrt{1 - \rho^2}}{\rho^2}\varphi$ (Theorem 4.1). It is free of butterfly arbitrage when $\theta\varphi(1 + |\rho|) < 4$ and $\theta\varphi^2(1 + |\rho|) \le 4$ (Theorem 4.2). The Heston-like curvature $\varphi(\theta) = \frac{1}{\lambda\theta}\bigl(1 - \frac{1 - e^{-\lambda\theta}}{\lambda\theta}\bigr)$ is arbitrage-free for $\lambda \ge (1 + |\rho|)/4$. The power law $\varphi(\theta) = \eta\theta^{-\gamma}(1 + \theta)^{\gamma - 1}$ is arbitrage-free for $\gamma \le 1/2$ and $\eta(1 + |\rho|) \le 2$. `calibrate_ssvi` fits $\rho$, the curvature parameters and $\theta_t$ under these constraints. It recovers $(\rho, \eta, \gamma) = (-0.7, 1.1, 0.4)$ and the Heston-like $(\rho, \lambda) = (-0.6, 2)$ to about $10^{-7}$.

**eSSVI** (Hendriks & Martini) gives each expiry its own $(\theta, \psi, \rho)$ with $\psi = \theta\varphi$. The butterfly conditions become $\psi(1 + |\rho|) < 4$ and $\psi^2(1 + |\rho|) \le 4\theta$. Two slices are free of calendar arbitrage when

$$
\theta_2 \ge \theta_1, \qquad \psi_2 \ge \psi_1, \qquad |\rho_2\psi_2 - \rho_1\psi_1| \le \psi_2 - \psi_1,
$$

which `spxa` uses as a sufficient condition. `calibrate_essvi` follows Corbetta, Cohort, Laachir & Martini (2019). Each slice is anchored at the quote $(k^*, \theta^*)$ closest to the money, through the exact relation

$$
\theta = \frac{(2\theta^* - \rho\psi k^*)^2 - \psi^2 k^{*2}}{4\theta^*}.
$$

The fit then scans $\rho$ and solves for $\psi$ within the closed-form bounds implied by the butterfly and calendar conditions. Between expiries, $\theta$, $\psi$ and $\rho\psi$ are interpolated linearly in time, which preserves both conditions. The tests recover the published SPX slices of Corbetta et al. (Table 1) to about $10^{-9}$.

---

## 4. Surfaces of SVI slices

`calibrate_svi_surface` fits raw SVI slices from the shortest expiry upwards, in the spirit of Gatheral & Jacquier (§5). Each slice must satisfy Durrleman's condition and stay above the previous slice on a common grid, $w_i(k_j) \ge w_{i-1}(k_j)$. Both constraints are refined with cutting planes. The wing slopes must also be ordered, $b_i(1 \pm \rho_i) \ge b_{i-1}(1 \pm \rho_{i-1})$, which prevents crossings beyond the grid. When the quotes themselves cross, the shorter slice is kept and the longer one is fitted as closely as the constraints allow. Slices taken from an arbitrage-free SSVI surface are reproduced to $5 \times 10^{-7}$.

**Interpolation in time.** `PriceInterpolatedSurface` interpolates prices rather than variances (Gatheral & Jacquier, §5.3). Take $\theta_t$ linear in $t$ between the ATM nodes, starting from $\theta_0 = 0$. For $t_i \le t \le t_{i+1}$, set

$$
c(k, t) = \alpha_t\,c_i(k) + (1 - \alpha_t)\,c_{i+1}(k), \qquad
\alpha_t = \frac{\sqrt{\theta_{i+1}} - \sqrt{\theta_t}}{\sqrt{\theta_{i+1}} - \sqrt{\theta_i}}.
$$

Below the first expiry, $c_0(k) = (1 - e^k)^+$ is intrinsic value. A convex combination of call-price functions is the call-price function of a mixture of distributions, so butterfly arbitrage cannot appear. Because $\alpha_t$ decreases in $t$ and the slices are ordered, prices are non-decreasing in $t$, so calendar arbitrage cannot appear either. In $s = \sqrt\theta$ the ATM price is $(s - s^3/24 + \dots)/\sqrt{2\pi}$, and the weights are linear in $s$. The only error comes from interpolating $s^3$ linearly, which is exactly $(s - s_i)(s_{i+1} - s)(s_i + s + s_{i+1})$. The interpolated ATM total variance therefore falls short of $\theta_t$ by

$$
\theta_t - w(0, t) = \frac{s}{12}(s - s_i)(s_{i+1} - s)(s_i + s + s_{i+1})\bigl(1 + O(\theta_{i+1})\bigr), \qquad s = \sqrt{\theta_t}.
$$

The tests check this to a relative accuracy of $\theta_{i+1}/20$.

---

## 5. Implied distributions

`ImpliedDistribution` turns a slice into the law of $X = \log(S_T/F_T)$. Differentiating the call price along the slice gives the survival function, which is also the price of a normalised digital call:

$$
\mathbb{Q}(X > k) = \Phi(d_-) - \varphi(d_-)\,\frac{w'(k)}{2\sqrt{w}}.
$$

Together with the density of §1, this is exact in $w$, $w'$ and $w''$. Moments, the forward ratio $E[e^X]$, call repricing, quantiles and inverse-transform samples are built on top. The integration range is widened until the exact tail mass beyond it is below $10^{-17}$. For an arbitrage-free slice the tests find unit mass and $E[e^X] = 1$ to $10^{-11}$, and they reprice calls by quadrature to the same accuracy. A slice with butterfly arbitrage shows up as a cdf that decreases somewhere; quantiles and samples refuse such slices.

---

## 6. Surfaces from option quotes

### 6.1 Forwards and quotes

`build_vol_surface` takes bid/ask chains, one per expiry, and fits a surface in four steps:

1. **Forward and discount factor.** Each expiry's forward $F_i$ and discount factor $D_i$ come from the chain if set. Otherwise they are implied from put-call parity, $C - P = D(F - K)$, by a robust line fit across strikes.
2. **Term structure.** `ForwardTermStructure` interpolates $\log F$ and $\log D$ linearly in $t$ between expiries, which means constant carry and flat discount forwards. It starts from $(0, \log S)$ when the spot is known.
3. **Total variances.** `chain_total_variance` converts the out-of-the-money mids to $(k, w)$ pairs and drops quotes that break the price bounds. Weights are uniform, by vega, or from the bid-ask spread.
4. **Fit.** The pairs are fitted with calendar-free SVI (§4), eSSVI or SSVI (§3).

The result is a `SurfaceBuild` holding a `VolSurface` and the fitted data. `SurfaceBuild.quality()` reports the following per expiry, together with a calendar check across expiries:

- the RMSE and largest implied-volatility error;
- ATM volatility and skew;
- the minimum of Durrleman's $g$ (§1).

### 6.2 Repairing prices

Some quotes cannot be fitted without arbitrage at all. Cohen, Reisinger & Wang (2020) work in normalised units $c = C/(DF)$ and $\kappa = K/F$. A grid then extends to an arbitrage-free surface, linear in $\kappa$ between quotes, exactly when four conditions hold:

- $(1 - \kappa)^+ \le c \le 1$;
- slopes lie in $[-1, 0]$, with $c(0) = 1$;
- slopes are non-decreasing (no butterfly arbitrage);
- $c$ is non-decreasing in $T$ at every strike of either slice inside both ranges.

All four conditions are linear in the prices. `repair_call_prices` therefore finds the smallest weighted $\ell^1$ change with a linear programme, and moves that stay inside the bid-ask spread can be made free. For $F = 100$, $D = 1$ and calls 13, 8 and 2 at strikes 90, 100 and 110, the slopes $-0.5$ and $-0.6$ break convexity. The cheapest repair lowers the middle price to 7.5, an $\ell^1$ change of $0.005$ in normalised units.

### 6.3 Smoothing prices

Fengler (2009) fits a natural cubic spline $g(\kappa)$ to normalised call prices by quadratic programming:

$$
\min_{g,\,\gamma}\ \sum_i w_i\bigl(y_i - g(u_i)\bigr)^2 + \lambda\,\gamma^\top R\,\gamma,
$$

Here $\gamma = g''$ at the knots, so $\gamma \ge 0$ makes $g$ convex. The spline must also satisfy $(1 - u)^+ \le g \le 1$. `fengler_surface` fits the longest expiry first and each shorter one under $g_{T_1} \le g_{T_2}$. Imposing this only at the knots leaves small violations between them. Instead, each spline piece is written in Bernstein form on the union of both knot sets, halved, and every Bernstein coefficient of the difference is constrained. The difference then stays non-negative to below $10^{-12}$ everywhere on the overlap.

---

## 7. The surface object

### 7.1 Strikes, volatilities and time interpolation

`VolSurface(model, forward)` combines a total-variance model with a forward curve. It works in strikes: prices, implied volatilities, deltas and the smile at each expiry, all broadcasting $K$ against $T$. The forward variance between two expiries at fixed $k$ is $\bigl(w(k, T_2) - w(k, T_1)\bigr)/(T_2 - T_1)$. It is non-negative for every $k$ exactly when there is no calendar arbitrage, and `forward_vol` raises otherwise.

`VolSurface.from_slices` interpolates between fitted slices, in one of three ways:

| Mode | Interpolates | Guarantee |
| :--- | :--- | :--- |
| `price` (default) | $c(k, t)$, by §4 | no static arbitrage if the slices have none |
| `fixed_k` | $w(k, t)$ linearly in $t$ at fixed $k$ | no calendar arbitrage if the slices are ordered |
| `normalised` | $w/\theta_t$ at fixed $x = k/\sqrt{\theta_t}$, with $\theta_t$ linear | exact for SSVI with $\varphi(\theta)\sqrt\theta$ constant |

Only the variance modes extrapolate beyond the last expiry. In the normalised mode $w_i(x\sqrt{\theta_i})/\theta_i$ is the same function of $x$ on every slice of such an SSVI surface. The interpolation then returns the SSVI surface itself at every intermediate $\theta_t$.

### 7.2 Delta conventions

The Black forward delta of a call is $N(d_1)$, and with premium adjustment it is $(K/F)N(d_2)$. `strike_from_delta` inverts either convention at the slice's own volatility. The premium-adjusted call delta is not monotone in $K$, so it is inverted on the branch above its maximum. The ATM delta-neutral straddle strike solves call delta + put delta $= 0$. At a fixed volatility this is $Fe^{w/2}$, or $Fe^{-w/2}$ with premium adjustment. For $F = 100$, $\sigma = 20\%$ and $T = 1$ these are 102.0201 and 98.0199, and the 25-delta call strike is $F\exp\bigl(\sigma N^{-1}(0.75) + w/2\bigr) = 116.7539$.

Under a smile, the forward delta depends on how $\sigma$ moves with the forward. Sticky strike keeps Black's delta. Sticky moneyness, $\sigma = \sigma(\log(K/F))$, gives $\Delta = \Delta_{BS} - \mathcal V_{BS}\,\sigma_k/F$. Under a negative skew this sticky-moneyness delta is the larger of the two.

### 7.3 Bumps and bucketed vega

A bump adds $\delta(k, T)$ to the implied volatility at fixed log-moneyness, $w' = (\sigma + \delta)^2T$. There are three shapes:

- `parallel_bump` is $\delta = s$;
- `skew_bump` is $\delta = sk$, which raises the ATM skew by exactly $s$;
- `expiry_buckets` are triangular hats in $T$ on pillar expiries that sum to one. A book at 0.75 years with pillars at 0.5 and 1 year therefore splits its vega equally between them.

`bucketed_vega` reprices any function of a surface under central bumps. The bucketed vegas add up to the parallel vega to first order.

### 7.4 Simulation and persistence

`VolSurface.terminal_distribution(T)` is the implied law of $\log(S_T/F_T)$ (§5), and `sample_terminal` draws $S_T$ by inverse transform. Monte Carlo then reprices the forward, calls and digitals to within four standard errors, which checks the surface end to end. `dumps` and `loads` serialise surfaces, models, forwards and curves to JSON, with a type tag on each object. The round trip is bit-exact, because JSON writes each float as the shortest decimal that reads back to it.

## 8. Local volatility

### 8.1 Dupire's formula in implied variance

Dupire (1994) showed that exactly one diffusion $dS_t = (r - q)S_t\,dt + \sigma_{\mathrm{loc}}(S_t, t)S_t\,dW_t$ reproduces a given arbitrage-free call surface. In log-forward moneyness $y = \log(K/F_T)$ and total implied variance $w(y, T)$ the formula reads (Gatheral, 2006)

$$
\sigma^2_{\mathrm{loc}}(K, T) = \frac{\partial_T w}{1 - \dfrac{y}{w}\,\partial_y w + \dfrac14\Bigl(-\dfrac14 - \dfrac1w + \dfrac{y^2}{w^2}\Bigr)(\partial_y w)^2 + \dfrac12\,\partial_{yy} w}.
$$

Rates and dividends drop out because strikes are measured against the forward. The denominator is Durrleman's $g$ (§1), and the numerator is the forward variance at fixed moneyness (§7.1). The local variance is therefore positive exactly where the surface is free of butterfly and calendar arbitrage. `LocalVolSurface` returns `nan` elsewhere: if $w = (0.04T - 0.02T^2)$ at every $y$, the local vol is $\sqrt{0.02}$ at $T = 1/2$ and undefined at $T = 1.5$. `VolSurface.local_vol` evaluates the formula with five-point differences of any surface. `LocalVolGrid` tabulates it for fast lookups.

As $T \to 0$, the implied vol is the harmonic mean of the local vol between spot and strike (Berestycki, Busca & Florent, 2002):

$$
\frac{1}{\sigma_{\mathrm{imp}}(k)} = \int_0^1 \frac{du}{\sigma_{\mathrm{loc}}(uk)}.
$$

`bbf_implied_vol` evaluates this integral. For $\sigma_{\mathrm{loc}} = a + by$ it gives $bk/\log(1 + bk/a)$. Differentiating at $k = 0$ gives the familiar rule that the short-dated local skew is twice the implied skew: for a time-independent smile $\sigma(k) = 0.2 - 0.1k$, the ATM local skew tends to $-0.2$.

### 8.2 Pricing under local volatility

A `LocalVolSurface` is a volatility function $\sigma(S, t)$ for the solvers of [PDE methods](pde.md). `dupire_pde` solves the forward Dupire equation for all strikes and expiries at once, and must give back the surface it came from. `black_scholes_pde`, `american` and `barrier` price backwards, one contract at a time. Two checks tie the two directions together: the forward and backward vanilla prices converge to the surface, and down-and-out plus down-and-in equals the vanilla. With $q = 0$ an American call under local vol is worth its European value, as it is under any diffusion.

### 8.3 Andreasen–Huge interpolation

Dupire's formula needs a smooth, arbitrage-free surface. Andreasen & Huge (2011) start from the quotes instead. They take $\vartheta(x, t)$ piecewise constant in normalised strike $x = K/F_t$, with one level per quote, and advance normalised calls $c(x, t)$ over each expiry interval with one fully implicit step:

$$
\Bigl[I - \tfrac12 (t_i - t_{i-1})\,\vartheta_i^2(x)\,x^2 D_{xx}\Bigr] c_i = c_{i-1}.
$$

The matrix is an M-matrix, so $c_i$ stays convex in $x$, within $(1 - x)^+ \le c \le 1$, and $c_i \ge c_{i-1}$. The surface is free of static arbitrage by construction, between expiries as well, where the same step is taken with a partial time step. `calibrate_andreasen_huge` fits the levels expiry by expiry, and reproduces arbitrage-free quotes to solver tolerance.

### 8.4 Monte Carlo and the forward smile

`local_vol_paths` simulates $\log S$ by Euler steps on a `LocalVolGrid`, storing only the requested times. Forward-start options pay $(S_{T_2}/S_{T_1} - k)^+$ at $T_2$. Their value depends on the joint law of $S_{T_1}$ and $S_{T_2}$, which the surface does not determine. `LocalVolPaths.forward_start` prices them under local vol. Put–call parity holds path by path: call minus put pays $S_{T_2}/S_{T_1} - k$, and its expectation is $e^{-rT_2}\bigl(e^{(r - q)(T_2 - T_1)} - k\bigr)$. With a constant volatility the price is $e^{-rT_1}$ times a Black–Scholes call on unit spot, with strike $k$, expiry $T_2 - T_1$ and the same $r$ and $q$.

### 8.5 Densities and diagnostics

By Gyöngy's theorem, the marginal laws of the local-vol diffusion are those implied by the surface. `LocalVolSurface.density` solves the Fokker–Planck equation of $\log S$ with `FokkerPlanck1D` and returns the density of $y = \log(S_T/F_T)$. It matches the Breeden–Litzenberger density $e^{rT}\partial_{KK}C$ (§5), keeps unit mass, and keeps $E[S_T] = F_T$.

`local_vol_diagnostics` combines these checks for a given surface. It maps where the local vol is undefined on a (moneyness, time) grid. If it is defined everywhere, it reprices a strike × expiry grid with one forward Dupire solve and reports the implied-vol errors, which shrink as the grid is refined. For an SSVI surface whose $\theta_t = 0.04t - 0.012t^2$ peaks at $t = 5/3$, a 40-step time grid to $t = 2$ first flags undefined points at $t = 1.7$.

## 9. Stochastic local volatility

### 9.1 The model

Local volatility fits every vanilla but implies a forward smile that flattens too quickly. Stochastic volatility has realistic dynamics but cannot fit the whole surface. SLV combines the two. In forward log-moneyness $y = \log(S_t/F_t)$,

$$
dy = -\tfrac12 L^2(t, y)\,v\,dt + L(t, y)\sqrt v\,dW^1, \qquad dv = \kappa(\theta - v)\,dt + \eta\,\xi\sqrt v\,dW^2, \qquad d\langle W^1, W^2\rangle = \rho\,dt,
$$

with a Heston variance (`Heston`, priced in forward coordinates by Lewis's integral; in 't Hout and Foulon's case 1 gives $8.8948693600540$) and a leverage function $L$. The *mixing fraction* $\eta \in [0, 1]$ scales the vol-of-variance. By Gyöngy's theorem, the marginals of $y$, and therefore all vanillas, are those of local volatility if and only if

$$
L^2(t, y)\,E[v_t \mid y_t = y] = \sigma_{LV}^2(t, y).
$$

The leverage depends on the model's own conditional expectation, so the model is a McKean nonlinear diffusion, and calibrating it means solving for that conditional expectation.

### 9.2 The forward equation

`slv_forward` evolves the joint density $p(t, y, v)$ from a delta at $(0, v_0)$. As in the one-dimensional `FokkerPlanck1D`, the forward operator is the discrete adjoint $W^{-1}A_j^{\mathsf T}W$ of each part of the backward ADI splitting of `spxa.pde`, where $W$ holds the control-volume areas. Every generator row sums to zero, so the mass is conserved to rounding. The degenerate edge $v = 0$ is "natural". The far edges in $y$ and the top of the variance grid are absorbing: with a natural condition on a non-degenerate edge, the adjoint scheme blows up. `slv_grid` clusters variance nodes towards zero, places $v_0$ on a node, and takes the top of the grid at the $1 - 10^{-6}$ non-central chi-square quantile of $v_t$ over the horizon. With $L \equiv 1$ the density reproduces the Heston smile within 5 bp on a $121 \times 61$ grid with 100 steps, and within 1.5 bp on $161 \times 81$ with 200 steps.

### 9.3 Calibration by forward induction

`calibrate_leverage` reads $E[v \mid y]$ from the current density at each step, sets $L^2 = \sigma_{LV}^2/E[v \mid y]$ with the local variance at the step midpoint, and advances the density with that leverage (Ren, Madan & Qian). Where the marginal density is negligible, the conditional expectation is extended flat. Calibrated to the SSVI surface of §8, the model reprices the smile at one year within 5 bp at $\eta = 0.5$ on the $121 \times 61$ grid, and within 2.5 bp at $\eta = 1$ on $161 \times 81$. The local variance must be tabulated finely in $y$ (`as_local_vol` uses 601 points): linear interpolation of a convex function biases the volatility upwards. As a consistency check, feeding the model its own Gyöngy local variance returns $L \equiv 1$ to $10^{-9}$.

### 9.4 Particles, Monte Carlo and the PDE

`calibrate_leverage_particles` is the particle method of Guyon and Henry-Labordère. It simulates $N$ particles and estimates $E[v \mid y]$ at each step by Gaussian kernel regression, with bandwidth $\propto \sigma\sqrt t\,N^{-1/5}$. On the same problem its leverage agrees with the forward equation's within 2.5%. `slv_paths` simulates the calibrated model with full-truncation Euler for $v$ and a log-Euler step for $y$, so $e^y$ is an exact martingale step by step. Knock-out barriers are monitored continuously through the Brownian-bridge survival probability $1 - e^{-2d_0d_1/(a^2\Delta t)}$. In the Black–Scholes limit this matches Reiner–Rubinstein at 20 steps, while discrete monitoring overprices by more than five standard errors. `SLVPDE` prices backwards in $(S, v)$. Its far spot edge uses the linearity condition, which is exact for payoffs linear in $S$ (in log-spot the same condition breaks put–call parity). Spot barriers become zero-Dirichlet edges. The PDE and Monte Carlo agree on calibrated-SLV barriers within three standard errors.

### 9.5 The mixing fraction

Every $\eta$ reprices the same vanillas, but path-dependent prices move. At $\eta = 0$ the variance is deterministic and the model is local volatility: a one-year down-and-out call (spot and strike 100, barrier 85, $r = 3\%$, $q = 1\%$) on the SSVI surface agrees with the one-factor local-vol PDE value $8.18$ within 0.2%. A one-year up-and-out call with barrier 130 is worth $5.80$, $6.00$ and $6.33$ at $\eta = 0$, $0.5$ and $1$, while the smile is matched within 6 bp in each case. `calibrate_slv` builds the model for a given $\eta$ from a local-vol surface or a `VolSurface`. In practice $\eta$ is set from barrier or forward-start quotes, which the vanillas cannot pin down.
