# Implied Volatility Surfaces

`spxa.vol` builds implied-volatility surfaces that are free of static arbitrage. This chapter covers the parametric layer: SVI slices, the SSVI and eSSVI surfaces, their calibration, interpolation in time, and the risk-neutral distributions they imply. Option formulas, implied-volatility inversion and quote handling come from [Market conventions](market.md).

| Object | Implementation |
| :--- | :--- |
| Coordinates | `log_moneyness`, `total_variance`, `normalised_call_from_total_variance`, `total_variance_from_price` |
| SVI slices | `SVIRaw`, `SVINatural`, `SVIJumpWings` |
| Surfaces | `SSVI`, `HestonLikePhi`, `PowerLawPhi`, `ESSVI`, `ESSVISlice` |
| Calibration | `calibrate_svi`, `calibrate_svi_butterfly_free`, `calibrate_svi_surface`, `calibrate_ssvi`, `calibrate_essvi` |
| Diagnostics | `durrleman_g`, `check_butterfly`, `check_calendar`, `wing_asymptotics`, `rogers_tehranchi_slope` |
| Interpolation and densities | `PriceInterpolatedSurface`, `ImpliedDistribution`, `risk_neutral_density` |

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

Below the first expiry, $c_0(k) = (1 - e^k)^+$ is intrinsic value. A convex combination of call-price functions is the call-price function of a mixture of distributions, so butterfly arbitrage cannot appear. Because $\alpha_t$ decreases in $t$ and the slices are ordered, prices are non-decreasing in $t$, so calendar arbitrage cannot appear either. The ATM price $\sqrt{\theta/2\pi}(1 - \theta/24 + \dots)$ is interpolated linearly in $\sqrt\theta$. The interpolated ATM total variance therefore lies within $\theta_t(\sqrt{\theta_{i+1}} - \sqrt{\theta_i})^2/16$ of $\theta_t$.

---

## 5. Implied distributions

`ImpliedDistribution` turns a slice into the law of $X = \log(S_T/F_T)$. Differentiating the call price along the slice gives the survival function, which is also the price of a normalised digital call:

$$
\mathbb{Q}(X > k) = \Phi(d_-) - \varphi(d_-)\,\frac{w'(k)}{2\sqrt{w}}.
$$

Together with the density of §1, this is exact in $w$, $w'$ and $w''$. Moments, the forward ratio $E[e^X]$, call repricing, quantiles and inverse-transform samples are built on top. The integration range is widened until the exact tail mass beyond it is below $10^{-17}$. For an arbitrage-free slice the tests find unit mass and $E[e^X] = 1$ to $10^{-11}$, and they reprice calls by quadrature to the same accuracy. A slice with butterfly arbitrage shows up as a cdf that decreases somewhere; quantiles and samples refuse such slices.
