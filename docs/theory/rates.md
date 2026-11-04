# Interest Rates

`spxa.rates` prices interest-rate instruments on real calendar dates. It is built on the dates, day counts, calendars, schedules and curves of [Market conventions](market.md). Every curve in this chapter is a `YieldCurve` returning discount factors $P(0, t)$. The discount curve and the projection (forwarding) curve are passed separately, so single-curve and multi-curve pricing use the same code.

| Object | Implementation |
| :--- | :--- |
| Indices and fixings | `OvernightIndex`, `IborIndex`, `SOFR`, `ESTR`, `SONIA`, `EURIBOR3M`, `EURIBOR6M`, `FixingHistory` |
| Overnight compounding | `compounded_overnight_rate`, `observation_schedule`, `projected_overnight_rate` |
| Cashflows and legs | `FixedCoupon`, `IborCoupon`, `OvernightCoupon`, `fixed_leg`, `ibor_leg`, `overnight_leg` |
| Pricing | `leg_pv`, `leg_annuity`, `cashflow_table` |
| Money market | `Deposit`, `FRA` |
| Futures | `RateFuture`, `OvernightFuture`, `hull_white_futures_convexity`, `hull_white_compounded_convexity` |
| Swaps | `Swap`, `ois_swap`, `vanilla_swap`, `BasisSwap`, `basis_swap`, `step_notional`, `linear_notional` |
| Bonds | `FixedRateBond` |
| Piecewise curves | `PiecewiseCurve`, `LogLinearDiscount`, `LinearZero`, `CubicZero`, `CubicSplineRT`, `MonotoneCubicRT`, `MonotoneConvex` |
| Helpers | `DepositHelper`, `FRAHelper`, `FutureHelper`, `SwapHelper`, `BasisSwapHelper`, `BondHelper`, `quarterly_futures` |
| Bootstrapping | `bootstrap_curve`, `BootstrapResult`, `fit_curves`, `CurveSpec`, `GlobalFitResult` |
| Turns and policy steps | `ForwardSpread`, `turn_of_year`, `SpreadAdjustedCurve`, `flat_forward_curve`, `meeting_date_curve` |
| Parametric curves | `NelsonSiegelSvensson`, `nelson_siegel`, `fit_nss`, `fit_nss_zero_rates` |
| Curve sets and diagnostics | `CurveSet`, `compare_schemes`, `bump_locality`, `forward_roughness` |
| Curve risk | `node_deltas`, `quote_deltas`, `quote_jacobian`, `key_rate_deltas`, `KeyRateShiftedCurve`, `node_gamma`, `pnl_explain` |
| Hedging and factors | `hedge_notionals`, `par_hedge`, `curve_pca`, `CurvePCA` |
| Caps, floors and swaptions | `CapFloor`, `Caplet`, `cap_floor`, `Swaption`, `swaption`, `cash_annuity` |
| SABR | `SABR`, `sabr_lognormal_vol`, `sabr_normal_vol`, `alpha_from_atm`, `calibrate_sabr`, `SABRFit` |
| Arbitrage-free SABR | `ArbitrageFreeSABR`, `SABRDensity`, `sabr_paths`, `SABRPaths` |
| Stripping, cube and CMS | `strip_caplet_vols`, `CapletVolCurve`, `SwaptionCube`, `CMSReplication` |
| Hull–White | `HullWhite`, `ModelCashflows`, `swap_cashflows`, `hull_white_paths`, `HullWhiteTree`, `HullWhitePDE`, `calibrate_hull_white` |
| Two-factor Gaussian | `G2PP`, `g2pp_paths`, `calibrate_g2pp`, `HullWhite2F` |
| Positive short rates | `BlackKarasinski`, `BlackKarasinskiTree`, `black_karasinski_paths`, `CIRPlusPlus` |
| Model interface | `ShortRateModel`, `model_price`, `model_implied_vols` |
| Gaussian HJM | `GaussianHJM`, `exponential_vol`, `hump_vol`, `hjm_paths` |
| Cheyette | `Cheyette`, `cheyette_paths`, `CheyettePDE`, `cheyette_swaption`, `swap_rate_dynamics`, `calibrate_cheyette` |
| LIBOR market model | `LiborMarketModel`, `AbcdVol`, `exponential_correlation`, `rebonato_correlation`, `reduce_rank`, `lmm_paths`, `LMMPaths`, `calibrate_lmm`, `fit_abcd`, `LMMCalibration` |

---

## 1. Indices and overnight compounding

### 1.1 Term and overnight indices

An `IborIndex` (EURIBOR 1M–12M) fixes `fixing_days` business days before its value date and accrues to a maturity found by adding its tenor with modified following and the end-of-month rule. EURIBOR 3M fixed on Monday 15 January 2024 has value date 17 January and maturity 17 April 2024. An `OvernightIndex` (SOFR on the US-SIFMA calendar, €STR on TARGET, SONIA on the UK calendar) fixes every business day for the period up to the next business day. Published rates are stored by fixing date in a `FixingHistory`.

### 1.2 Compounding in arrears

For a period $[t_s, t_e)$ whose index business days are $d_0 < \dots < d_{N-1}$, the compounded rate is

$$
R = \frac{1}{\tau}\Bigl[\prod_{i=0}^{N-1}\bigl(1 + r_i\,\tau_i\bigr) - 1\Bigr],
\qquad \tau = \sum_i \tau_i,
$$

where $\tau_i$ is the index day-count fraction of the days on which $r_i$ applies; the averaged rate is $\sum_i r_i\tau_i/\tau$. A rate set on a Friday or before a holiday applies over the following non-business days. Over the SOFR week from 1 to 8 July 2024 (4 July is a US holiday) the fixings of 1, 2, 3 and 5 July carry weights $1, 1, 2, 3$ over 360.

ISDA (2021) defines three ways of knowing the rate before payment. With a **lookback** of $p$ days, each rate is the fixing $p$ business days earlier while the weights stay those of the calculation period. With an **observation shift**, the whole observation period moves back $p$ days and both rates and weights come from it. With a **lockout** of $s$ days, the fixing of the lockout date, $s$ business days before the period end, applies to the rest of the period. For the ISDA example period from 22 September to 22 December 2020 with $s = 5$, the lockout date is 15 December and its rate applies from 15 to 21 December. A **payment delay** only moves the payment date.

### 1.3 Projection from a curve

The forward rate for the fixing on $d_i$ is $F_i = \bigl(P(d_i)/P(d_{i+1}) - 1\bigr)/\tau_i$. Without lookback, shift or lockout, the projected daily factors telescope:

$$
\prod_{i \ge j}(1 + F_i\tau_i) = \frac{P(d_j)}{P(t_e)},
$$

so `projected_overnight_rate` needs two discount factors beyond the last published fixing. The tests check that the shortcut equals the daily product to rounding.

---

## 2. Cashflows, legs and pricing

A leg is a list of coupons, each with a payment date and an `amount(projection, fixings)`. A `FixedCoupon` pays $N K \tau$, an `IborCoupon` pays $N(\text{gearing}\cdot L + s)\tau$ with the term rate $L$ fixed in advance, and an `OvernightCoupon` pays $N R\tau$ with $R$ compounded or averaged in arrears. `fixed_leg`, `ibor_leg` and `overnight_leg` build legs from a `Schedule`, with payment lags and notionals that may change period by period. The present value and the annuity are

$$
\mathrm{PV} = \sum_i \text{amount}_i\,P_d(t^{\text{pay}}_i), \qquad A = \sum_i N_i\tau_i\,P_d(t^{\text{pay}}_i),
$$

summed over the coupons paid after the valuation date, with $P_d$ the discount curve. Projected IBOR coupons use the index's own value and maturity dates. These can differ from the coupon dates when a start date has been rolled. With `use_index_dates=False` they use the coupon dates instead, and on a single curve a floating leg then telescopes exactly to $N\bigl(P(t_0) - P(t_n)\bigr)$ when there is no payment lag.

---

## 3. Deposits, FRAs and futures

### 3.1 Deposits and FRAs

A deposit starting at spot pays $N(1 + R\tau)$ at maturity. A forward rate agreement on $[t_s, t_e]$ settles at $t_s$ the discounted difference between the fixing and the contract rate, so its value is

$$
\mathrm{PV}_{\text{FRA}} = N\,\frac{\tau(F - K)}{1 + \tau F}\,P_d(t_s),
\qquad F = \frac{1}{\tau}\Bigl(\frac{P_p(t_s)}{P_p(t_e)} - 1\Bigr).
$$

A 3×6 EURIBOR FRA traded on 15 January 2024 accrues from 17 April to 17 July 2024.

### 3.2 Term-rate futures

Daily margining makes the futures rate the risk-neutral expectation of the fixing, which exceeds the forward. In the Hull–White model $dr = (\vartheta(t) - ar)\,dt + \sigma\,dW$, with $B(t) = (1 - e^{-at})/a$ and $B_{ij} = B(t_j - t_i)$,

$$
1 + \tau R^{\text{fut}} = (1 + \tau F)\,e^{\Gamma}, \qquad
\Gamma = \sigma^2 B_{12}\Bigl[B_{12}\,\frac{1 - e^{-2at_1}}{2a} + \tfrac12 B_{01}^2\Bigr].
$$

As $a \to 0$ (Ho–Lee), $\Gamma \to \sigma^2\tau t_1(t_1 + 2\tau)/2$. For $\sigma = 1\%$, $t_1 = 1$ and $\tau = 0.25$, this gives $\Gamma = 1.875 \times 10^{-5}$. IMM dates are the third Wednesdays of March, June, September and December; `imm_code` and `imm_from_code` translate them to contract codes such as `H24`.

### 3.3 SOFR futures

A three-month SOFR future settles on the rate compounded over its IMM quarter. The March 2024 contract runs from 20 March to 19 June 2024, an accrual of $91/360$. The one-month contract settles on the arithmetic average over a calendar month. Write each daily factor as $1 + r_i\tau_i = 1/P(t_i, t_{i+1})$. Under the $t_N$-forward measure the product telescopes, so its expectation is $1 + \tau F = P(0, t_0)/P(0, t_N)$. In a Gaussian model $Y = \log\prod_i 1/P(t_i, t_{i+1})$ is Gaussian, and changing to the risk-neutral measure multiplies the expectation by $e^{\mathrm{Cov}(Y, \int_0^{t_N} r)}$. In Hull–White this gives, exactly for the daily product,

$$
1 + \tau R^{\text{fut}} = (1 + \tau F)\,e^{\Gamma_c}, \qquad
\Gamma_c = \sum_{t_i > 0} B(t_{i+1} - t_i)\,c(t_i; t_N), \qquad
c(t; T) = \sigma^2\Bigl[\tfrac12 B(t)^2 + B(T - t)\,\frac{1 - e^{-2at}}{2a}\Bigr],
$$

where $c(t; T) = \mathrm{Cov}\bigl(x(t), \int_0^T x\bigr)$ for the Gaussian factor $x$. Fixings already published do not contribute. The continuous limit is $\Gamma_c = \mathrm{Var}(I) + \mathrm{Cov}\bigl(\int_0^{T_s} r, I\bigr)$ with $I = \int_{T_s}^{T_e} r$, which for Ho–Lee is

$$
\Gamma_c = \sigma^2\bigl(\tau^2 T_s + \tau^3/3 + \tau T_s^2/2\bigr).
$$

For $\sigma = 1\%$, $T_s = 1$ and $\tau = 0.25$, this gives $\Gamma_c = 1.927083 \times 10^{-5}$, more than the term-rate $\Gamma$ because the rate is still being fixed during the quarter. For the same reason a spot-starting quarter keeps $\sigma^2\tau^3/3 = 5.208 \times 10^{-7}$. The tests check $c$ against the Itô-isometry integral, the Ho–Lee formula, and an exact G1++ Monte Carlo of both contract types.

---

## 4. Swaps

### 4.1 Par rates and identities

A `Swap` exchanges a fixed leg for a floating leg. For the payer of fixed,

$$
\mathrm{PV} = \mathrm{PV}_{\text{float}} - \mathrm{PV}_{\text{fixed}} = (S - K)\,A, \qquad S = \frac{\mathrm{PV}_{\text{float}}}{A},
$$

and on a single curve without payment lags $S = N\bigl(P(t_s) - P(t_e)\bigr)/A$. The par spread is the margin over the floating index that makes the swap worth zero.

`ois_swap` builds overnight-indexed swaps: both legs follow one annual schedule on the index calendar, the floating leg compounds in arrears, and payments lag by two business days. A 5-year SOFR OIS traded on Monday 15 January 2024 (Martin Luther King Day) starts on 18 January 2024. Its first payment is on 23 January 2025 and its last on 22 January 2029. `vanilla_swap` builds fixed-versus-IBOR swaps with the EUR conventions (annual 30/360 against the index tenor). A 10-year swap against EURIBOR 6M traded on 15 January 2024 starts on 17 January and has 10 fixed and 20 floating periods. Its second fixed period ends on the rolled date 19 January 2026, so it accrues $362/360$. `basis_swap` exchanges two floating legs, for example EURIBOR 3M plus a spread against EURIBOR 6M. On a single curve, with coupon-date projection, its par spread is zero to $10^{-14}$.

### 4.2 Risk measures

`pv01` is the change in value for a one basis point rise in the fixed rate, $10^{-4}A$. `dv01` is the central difference of the value when both curves are shifted by one basis point in continuously compounded zero rate. For an OIS on a flat curve the tests compare it with the analytic derivative

$$
\mathrm{DV01} = 10^{-4}\Bigl[N\bigl(-t_s P(t_s) + t_e P(t_e)\bigr) + K\sum_i N\tau_i\,t_i\,P(t_i)\Bigr]
$$

to $10^{-6}$. For an at-the-money swap, DV01 and PV01 agree within a few percent.

### 4.3 Amortising and accreting swaps

The notional may be a sequence (one entry per period) or a function of the accrual start date, such as `step_notional` or `linear_notional`. Each coupon is linear in its notional. A swap whose notional steps down by $\Delta N_j$ at $T_j$ is therefore exactly the sum of bullet swaps of notional $\Delta N_j$ maturing at $T_j$, and its par rate is the annuity-weighted average of their par rates. The tests verify this for OIS and vanilla swaps, including their DV01.

---

## 5. Fixed-rate bonds

`FixedRateBond` generates an unadjusted coupon schedule with payments rolled to the following business day. Accrued interest uses the reference periods that ACT/ACT ICMA requires. With $n$ coupons a year, settlement $t_s$, next coupon $t_1$ and $w = n\,\tau_{\text{ICMA}}(t_s, t_1)$, the street-convention dirty price at yield $y$ is

$$
P_{\text{dirty}} = \sum_{j \ge 1} \frac{c_j}{(1 + y/n)^{w + j - 1}},
$$

and the clean price is the dirty price minus accrued interest. Macaulay duration is the time-weighted average of the discounted cash flows. Modified duration divides it by $1 + y/n$, and convexity is the second derivative of the price divided by the price. `dirty_price_from_curve` discounts on a curve with an optional z-spread added to continuously compounded zero rates, and `z_spread` inverts it.

For a 10-year 5% semi-annual bond issued on 15 January 2024 and valued on that date:

| Quantity | Value |
| :--- | :--- |
| Dirty price at a 6% yield | 92.56126256977227 |
| Clean price at a 5% yield | 100 |
| Macaulay duration at 5% | 7.98945 years |
| Accrued interest on 15 March 2024 | $2.5 \times 60/182 = 0.824176$ |
| Clean price at 5% on 15 March 2024 | 99.99319 |

The 2034 final coupon falls on Sunday 15 January and Monday 16 January is Martin Luther King Day, so it is paid on 17 January 2034. The accrual still ends on the unadjusted date.

---

## 6. Curve construction

### 6.1 Piecewise curves

A `PiecewiseCurve` stores discount factors $P_i$ at node times $t_i$, with the node $(0, 1)$ prepended. An interpolation scheme turns them into a smooth $x(t) = -\log P(t)$, and the instantaneous forward $f = x'$ is the scheme's analytic derivative. Following Hagan & West (*Interpolation methods for curve construction*, 2006), the schemes are:

| Scheme | Interpolates | Forward | Local |
| :--- | :--- | :--- | :--- |
| `LogLinearDiscount` | $\log P$ linearly | piecewise flat | yes |
| `LinearZero` | zero rate $r$ linearly | piecewise linear in $t$, jumps at nodes | yes |
| `CubicZero` | $r$ by a cubic spline | $C^1$ | no |
| `CubicSplineRT` | $rt$ by a cubic spline | $C^1$ | no |
| `MonotoneCubicRT` | $rt$ by Hyman-filtered Hermite cubics | $C^0$ | nearly |
| `MonotoneConvex` | discrete forwards, Hagan–West monotone convex | $C^0$; positive with `positivity=True` when the discrete forwards are | nearly |

Monotone convex reproduces each discrete forward $f^d_i = (x_i - x_{i-1})/(t_i - t_{i-1})$ as the average of $f$ over its interval. With zero rates 2.75%, 4%, 4.75%, 5%, 4.75% at 1–5 years, it gives forwards 0.0229375, 0.05010625, 0.0609 and 0.03625 at 0.3, 1.3, 2.1 and 4.5 years.

`compare_schemes` scores schemes on Hagan and West's criteria: minimum forward, roughness $\int f'^2$, locality, and the largest forward move per unit zero-rate bump. On the hump-and-dip curve with zero rates 8.1%, 7%, 4.4%, 7%, 4%, 3% at 0.1, 1, 4, 9, 20 and 30 years:

- `LinearZero` and `CubicZero` produce negative forwards, while `MonotoneConvex` stays positive.
- `LogLinearDiscount` has minimum forward $(0.03 \cdot 30 - 0.04 \cdot 20)/10 = 0.01$.
- A bump to $r_j$ moves the log-linear flat forward on $(t_{j-1}, t_j)$ by $t_j/(t_j - t_{j-1})$ per unit, and on $(t_j, t_{j+1})$ by $-t_j/(t_{j+1} - t_j)$. Its largest sensitivity is therefore $30/10 = 3$, reached at the last node.
- The natural splines spread a single bump across the curve. One bump can move the `CubicSplineRT` forward on all six intervals and the `CubicZero` forward on five, against at most two for the monotone schemes.

### 6.2 Helpers and the bootstrap

A helper is a quoted instrument that can compute its model quote on a mapping of named curves. Its *pillar* is the last date the quote depends on. `bootstrap_curve` sorts the helpers by pillar and puts a node at each pillar. It then solves node $i$, in $x_i = -\log P(t_i)$ with Brent's method, so that helper $i$ reprices. Projection curves are built on a known discount curve (dual-curve bootstrapping) by passing `known={"discount": ois}`.

The two kinds of scheme behave differently:

- **Local schemes.** Later nodes do not move the curve before $t_i$, so one pass reprices every helper to rounding.
- **Non-local schemes.** The passes repeat until the nodes stop moving.

`fit_curves` instead solves every node of several curves at once by Newton's method, with a finite-difference Jacobian. This handles cases a sequential bootstrap cannot: for example, long ESTR points that exist only through ESTR/EURIBOR basis swaps, which need both curves at once. For a single curve it converges to the same nodes.

`BootstrapResult.jacobian` returns $J_{ij} = \partial q_i/\partial x_j$ at the solution. It is lower triangular for a local scheme. By the implicit function theorem, `node_sensitivities` returns $\partial x/\partial q = J^{-1}$: the move of each node per unit move of one quote with the others fixed, which is exactly the re-bootstrapped curve.

### 6.3 Front end, futures and turns

`Deposit.overnight`, `Deposit.tom_next` and `Deposit.spot_next` pin the curve between the trade date and spot. Traded on Friday 8 March 2024, the O/N deposit runs over the weekend to Monday, an accrual of $3/360$. With O/N and T/N helpers, $P(t_{\text{spot}}) = 1/\bigl((1 + \tau_{ON} r_{ON})(1 + \tau_{TN} r_{TN})\bigr)$.

`quarterly_futures` builds a strip of consecutive 3M overnight futures from the next IMM date. Each `FutureHelper` carries Hull–White parameters $(a, \sigma)$ for the convexity adjustment of §3.3. With the right parameters, the strip recovers the forwards of the curve that produced the prices. With zero convexity the curve forward $F'$ equals the futures rate. It therefore overstates the true forward $F$ by exactly $1 + \tau F' = (1 + \tau F)e^{\Gamma_c}$.

Year-end funding pressure lifts the overnight forward over the turn. `turn_of_year(year, s)` returns a `ForwardSpread` adding $s$ to the instantaneous forward from the last business day of the year to the first of the next: from 31 December 2024 to 2 January 2025, or, with the 2022 year end on a Saturday, from 30 December 2022 to 2 January 2023. A `SpreadAdjustedCurve` multiplies the base discount factor by $e^{-s\,|[t_a, t_b) \cap [0, t)|}$. With `spreads=` passed to `bootstrap_curve`, the smooth base curve is fitted under the known turn.

### 6.4 Policy-meeting curves

Overnight rates step on central-bank meeting dates. `flat_forward_curve` holds the forward at $r_j$ on $[s_{j-1}, s_j)$, which is `LogLinearDiscount` with nodes at the step dates. `meeting_date_curve` fits one rate per inter-meeting period to any helpers. The fit is exact with one helper per period, and Gauss–Newton least squares when there are more, for example monthly SOFR futures, several of which can fall between two FOMC meetings. A period that no helper depends on makes the Jacobian rank-deficient, and the fit raises `ValueError` rather than invent a rate.

### 6.5 Parametric curves

Svensson's curve, with $x_k = t/\tau_k$ and $L(x) = (1 - e^{-x})/x$, has

$$
z(t) = \beta_0 + \beta_1 L(x_1) + \beta_2\bigl(L(x_1) - e^{-x_1}\bigr) + \beta_3\bigl(L(x_2) - e^{-x_2}\bigr),
\qquad
f(t) = \beta_0 + \beta_1 e^{-x_1} + \beta_2 x_1 e^{-x_1} + \beta_3 x_2 e^{-x_2},
$$

so $f(0) = z(0^+) = \beta_0 + \beta_1$ and $z(\infty) = \beta_0$. Nelson–Siegel is $\beta_3 = 0$. The model is linear in the betas for fixed decay times, so `fit_nss_zero_rates` uses variable projection: a search over $\log\tau$ with a linear least-squares solve inside. `fit_nss` fits helper quotes, bonds included, by Levenberg–Marquardt.

### 6.6 Curve sets

A `CurveSet` is a mapping of named curves with a discount role and an index-to-curve projection map. Helpers read it directly, and `pv` prices swaps and basis swaps by the index on each floating leg.

---

## 7. Curve risk

### 7.1 Node and quote deltas

A fitted curve is a function of its node values $x_j = -\log P(t_j)$, and a price $V$ is a function of the curves. `node_deltas` returns $\partial V/\partial x_j$ by central differences, leaving the interpolation scheme and any turn spreads in place. The fit makes the helper model quotes $m(x)$ equal to the market quotes $q$. By the implicit function theorem $\partial x/\partial q = J^{-1}$ with $J = \partial m/\partial x$, so

$$
\frac{\partial V}{\partial q} = J^{-\top}\,\frac{\partial V}{\partial x}.
$$

`quote_deltas` evaluates this from one Jacobian instead of one re-bootstrap per quote. The result is the sensitivity to each quote with the others held fixed, exactly what bumping a quote and refitting gives. It takes a single-curve `BootstrapResult` or a multi-curve `GlobalFitResult`. For the latter, nodes and helpers are stacked across curves, so cross effects are included.

A par swap identical to helper $i$ is a useful check. Its value is $(S - K)A$, and with the other quotes fixed only its own par rate $S = q_i$ can move. Its quote deltas are therefore exactly $A\,e_i$: the annuity on its own quote and zero elsewhere, OIS quotes included in a dual-curve fit.

### 7.2 Key-rate deltas

Key-rate deltas (Ho, 1992) need no fit. The zero curve of any `YieldCurve` is moved by triangular hats $h_k(t)$ centred on key tenors and flat beyond the first and last keys:

$$
P'(t) = P(t)\exp\Bigl(-t\sum_k b_k\,h_k(t)\Bigr).
$$

The hats sum to one, so key-rate deltas add up to the parallel delta to first order. A zero-coupon bond maturing on a key has delta $-T\,P(T)$ on that key and none elsewhere. At a flat 3% continuously compounded rate, this is $-5e^{-0.15} = -4.30354$ for $T = 5$.

### 7.3 Hedging and P&L explain

`hedge_notionals` minimises $\lVert W^{1/2}(d + Hn)\rVert_2$ over hedge notionals $n$, where $d$ is the book's risk and column $j$ of $H$ is the risk of one unit of hedge $j$. With par swaps on every pillar, $H$ is diagonal with the annuities, so $n_i = -d_i/A_i$. With fewer hedges than buckets, the residual is $W$-orthogonal to every hedge.

Between two fits on the same nodes, `pnl_explain` splits the change in value into $\delta^\top\Delta x$ and $\tfrac12\Delta x^\top\Gamma\Delta x$, with $\Gamma$ from `node_gamma`. The remainder is third order. For a zero-coupon bond on a node it is exactly $-P\,\Delta x^3/6$ to leading order. For an 8-year OIS, multiplying a market move by five multiplies the remainder by about $5^3 = 125$.

### 7.4 Principal components

`curve_pca` diagonalises the sample covariance of zero-rate changes on fixed tenors. Its first three loadings are level, slope and curvature (Litterman & Scheinkman, 1991). Signs are fixed so that the level loadings sum to a positive number, the slope rises from short to long tenors, and the curvature is positive at the middle tenor. For a book with key-rate deltas $d$ on the same tenors, the factor deltas are $d^\top v_j$. The book's P&L variance under the retained factors is $\sum_j\lambda_j(d^\top v_j)^2$, which equals the first-order P&L variance when the changes have exactly that rank.

## 8. Volatility products

### 8.1 Caps, floors and swaptions

A caplet on a term index fixing at $T$ for the period $[t_s, t_e]$ pays $\tau(L - K)^+$ at $t_e$. Under the $t_e$-forward measure $L$ is a martingale, so its value is $P(t_e)\,\tau\,\mathrm{Black}(F, K, T, \sigma)$, with $F$ the forward from the projection curve and $P$ from the discount curve. `CapFloor.price` accepts Black, shifted Black or normal (Bachelier) vols, either one flat number or a function $\sigma(T, K)$. `cap_floor` skips the first period by default, since it is already fixed: a 5Y EURIBOR6M cap has 9 caplets. Caps minus floors at the same strike equal the floating leg minus $K$ times the caplet annuity.

A physically settled payer swaption is a call on the forward swap rate $S$ under the annuity measure, worth $A\,\mathrm{Black}(S, K, T, \sigma)$. Cash-settled swaptions use the cash annuity

$$
\bar A(S) = \sum_{i=1}^{n} \frac{1/m}{(1 + S/m)^i} = \frac{1 - (1 + S/m)^{-n}}{S},
$$

which `cash_annuity` evaluates without cancellation as $S \to 0$.

### 8.2 SABR

In SABR (Hagan et al., 2002) $dF = \alpha F^\beta dW$ and $d\alpha = \nu\alpha\,dZ$, with $d\langle W, Z\rangle = \rho\,dt$. `SABR.lognormal_vol` gives Hagan's expansion, or Obłój's (2008) correction with `method="obloj"`. The two agree at the money. They differ in the wings even at $T = 0$: with $\alpha = 0.035$, $\beta = 0.5$, $\rho = -0.3$, $\nu = 0.4$ and $F = 0.03$, at $K = F/e$ Hagan gives $0.35902$ and Obłój $0.36024$. `calibrate_sabr` fixes $\beta$ and fits $(\alpha, \rho, \nu)$ to a smile by least squares. If an ATM vol is given, it solves for $\alpha$ from that vol through `alpha_from_atm`, so the ATM quote is matched exactly. A shift $s$ applies the model to $F + s$, for negative rates.

### 8.3 Arbitrage-free SABR

At long expiries and low rates, Hagan's formula implies a negative density near zero. `ArbitrageFreeSABR` instead solves the effective one-dimensional forward equation of Hagan et al. (2014) for the density, with absorbing mass at the boundary, using the variable $z$ and the Lawson–Swayne scheme of Le Floc'h & Kennedy (2017):

$$
\frac{\partial Q}{\partial T} = \frac{\partial^2 (M Q)}{\partial F^2}, \qquad M = \tfrac12 D(F)^2 E(F, T).
$$

Mass and the forward are conserved to rounding error. For $\alpha = 0.02$, $\beta = 0.5$, $\rho = -0.3$, $\nu = 0.5$, $F = 1\%$ and $T = 10$, Hagan's prices have a negative density near zero, while the PDE puts $8.74\%$ of the mass at zero. With $\beta = 0$ and $\nu = 0$ the model is Brownian motion absorbed at zero, and the PDE converges to the reflection-principle prices at second order.

`sabr_paths` simulates the model itself, with absorption at zero. With $\beta = 0$ it adds a Brownian-bridge crossing test, which removes the $O(\sqrt{\Delta t})$ bias of checking only at the grid times. The PDE is an expansion of the true dynamics, so at long expiries its prices differ from Monte Carlo by more than sampling error.

### 8.4 Caplet stripping and the swaption cube

Cap markets quote one flat vol per maturity. `strip_caplet_vols` turns these into caplet vols by bootstrapping. It sorts the caps by maturity and gives the caplets that each longer cap adds one common vol, chosen so that the cap reprices; earlier caplets keep the vols already stripped. The result is a piecewise-constant `CapletVolCurve` in expiry, usable directly as $\sigma(T, K)$ in `CapFloor.price`. If flat vols fall faster than the bootstrap can absorb, the required caplet variance becomes negative and the function raises.

`SwaptionCube` holds one SABR smile per (expiry, tenor) node, with a common $\beta$. Between nodes it interpolates $\log\alpha$, $\rho$ and $\log\nu$ bilinearly, which keeps $\alpha, \nu > 0$ and $|\rho| < 1$. Beyond the matrix the parameters are held flat.

### 8.5 CMS

A coupon paying the swap rate $S_T$ at $T_p$ is worth $A_0\,E^A[S_T\,\alpha(S_T)]$, where the annuity mapping $\alpha(S) \approx P(T, T_p)/A(T)$ comes from a terminal swap-rate model. `CMSReplication` evaluates such expectations by Carr–Madan replication with swaption prices:

$$
E^A[f(S_T)] = f(S_0) + \int_{-\infty}^{S_0} f''(K)\,P(K)\,dK + \int_{S_0}^{\infty} f''(K)\,C(K)\,dK.
$$

The standard mapping (Hagan, 2003) is $\alpha \propto G(S) = S(1 + S/m)^{-\delta}/(1 - (1 + S/m)^{-n})$. The linear mapping is $\alpha(S) = P(T_p)/A_0 + a(S - S_0)$. With the linear mapping and a lognormal swap rate, the convexity adjustment has the closed form

$$
E^{T_p}[S_T] - S_0 = \frac{a A_0}{P(T_p)}\,S_0^2\,\bigl(e^{\sigma^2 T} - 1\bigr),
$$

which the replication reproduces to $10^{-8}$ relative. With the slope $a$ taken from the standard mapping, the adjustment for a 5-year fixing of a 10-year annual swap rate at $S_0 = 3\%$ and $\sigma = 20\%$ is $10.2$ bp. The standard mapping itself gives $10.8$ bp. The smile matters as well. The payoff is most convex at high rates, so raising the vols at high strikes (a larger $\nu$) raises the adjustment, while a downward skew ($\beta < 1$, $\rho < 0$) lowers it; `examples/rates_vol_example.py` shows the second case.

## 9. Short-rate models

### 9.1 Hull–White

`HullWhite` writes the short rate as $r_t = f(0, t) + x_t$ with

$$
dx_t = \bigl(y(t) - a x_t\bigr)\,dt + \sigma(t)\,dW_t, \qquad x_0 = 0, \qquad y(t) = \int_0^t e^{-2a(t-u)}\sigma(u)^2\,du,
$$

which fits the discount curve by construction (Andersen & Piterbarg's form; $\sigma$ may be piecewise constant). Bond prices are $P(t, T) = \frac{P(0,T)}{P(0,t)}\exp\bigl(-B x_t - \tfrac12 B^2 y(t)\bigr)$ with $B = (1 - e^{-a(T-t)})/a$, so zero-bond options are Black formulas on the bond forward, and caplets are bond puts. A swap with discount-curve cashflows (`swap_cashflows`, with the projection basis kept as a deterministic spread) is a portfolio of bonds. When its amounts change sign once, Jamshidian's decomposition prices the swaption as bond options struck at $P(T, t_i | x^*)$. Otherwise `swaption` integrates against the Gaussian.

### 9.2 Trees, PDE and simulation

`HullWhiteTree` is the Hull–White (1994) trinomial tree, with the drift fitted step by step so that every grid bond reprices exactly. `HullWhitePDE` solves the pricing equation in $x$ on the `spxa.pde` solvers, with exercise events and American constraints. `hull_white_paths` simulates $(x, \int x)$ exactly, so the bank-account discount factor reprices the curve without bias. These engines handle early exercise. On a flat 3% curve, with $a = 0.05$ and $\sigma = 1\%$, an American put expiring in 2 years on the 6-year bond, struck at $P(0, 6)$, is worth $0.0094954$, against $0.0020894$ for the European. Exercise pays as soon as rates fall far enough to lift the bond above its current price.

### 9.3 Calibration

`calibrate_hull_white` fixes $a$ and fits $\sigma$ to swaptions or caps. In bootstrap mode, $\sigma$ is piecewise constant between instrument expiries and each level is solved in turn. In global mode, $a$ and one $\sigma$ are fitted together by least squares. `calibrate_g2pp` fits all five G2++ parameters to a grid of swaptions. It keeps $a > b$ (so $x$ is the fast factor) and $|\rho| < 1$ through its parametrisation, and it recovers the parameters of a synthetic $3 \times 3$ grid to $10^{-5}$. `model_implied_vols` quotes any `ShortRateModel` (Hull–White, G2++, CIR++) in Black or Bachelier vols for comparison with the market.

### 9.4 G2++ and the two-factor Hull–White model

In G2++ (Brigo & Mercurio, §4.2) $r = x + y + \varphi(t)$, with two correlated Ornstein–Uhlenbeck factors of speeds $a, b$ and vols $\sigma, \eta$. Bonds stay exponential-affine. Swaptions reduce to a one-dimensional integral: given $x_T$, the payoff is monotone in $y_T$, and the conditional expectation is a sum of normal distribution functions. Correlation decorrelates the curve: a negative $\rho$ cancels part of the short-end variance while long rates follow the slow factor. With $a = 0.5$, $\sigma = 1.2\%$, $b = 0.05$ and $\eta = 0.9\%$, a 5Y×20Y swaption is worth $12.2$ times a 5Y×1Y at $\rho = -0.7$, and $8.3$ times at $\rho = 0.7$. Hull and White's two-factor model, with a stochastic drift $u$ in the short rate, is the same model: `HullWhite2F.to_g2pp` and `from_g2pp` map the parameters both ways, preserving the instantaneous covariance of $(dr, du)$.

### 9.5 Positive rates: Black–Karasinski and CIR++

Gaussian models give negative rates. For $a = 0.1$ and $\sigma = 0.85\%$, the 10-year Hull–White tree reaches $-9.2\%$ at its lowest node. In Black–Karasinski $\log r$ is an Ornstein–Uhlenbeck process, so rates stay positive. There are no closed forms, so `BlackKarasinskiTree` solves for each step's drift by Newton's method to fit the curve, and `black_karasinski_paths` simulates on the fitted drift. CIR++ (Brigo & Mercurio, 2001) shifts a CIR factor by $\varphi = f^M - f^{\rm CIR}$. It keeps CIR's closed forms: bond options are scaled non-central $\chi^2$ formulas, and swaptions follow by Jamshidian, since bond prices decrease in $x$. Rates stay above $\varphi$. With $\kappa = 0.3$, $\theta = 2.5\%$, $\sigma = 0.08$ and $x_0 = 2\%$ on the flat 3% curve, $\varphi$ is positive out to 30 years. The square-root diffusion also produces a smile. For a 2Y×5Y swaption at the money and $\pm 100$ bp, this CIR++ gives normal vols of $27.48$, $46.42$ and $57.09$ bp, while Hull–White ($a = 0.05$, $\sigma = 1\%$) is nearly flat at $86.64$, $87.14$ and $87.65$ bp.

## 10. HJM and Cheyette

### 10.1 Gaussian HJM

Heath, Jarrow and Morton model the whole forward curve. With a deterministic vector volatility $\sigma(t, T)$ driving independent Brownian motions,

$$
df(t, T) = \sigma(t, T)\cdot\Sigma(t, T)\,dt + \sigma(t, T)\cdot dW_t, \qquad \Sigma(t, T) = \int_t^T \sigma(t, s)\,ds,
$$

and the drift is forced by no-arbitrage, so `GaussianHJM` fits the initial curve for any choice of volatility. Bonds are lognormal, and under the $T$-forward measure $P(t, S)/P(t, T)$ has log-variance $\int_0^T |\Sigma(u, S) - \Sigma(u, T)|^2\,du$. Bond options and caplets are therefore Black formulas. `exponential_vol` ($\sigma e^{-a(T-t)}$) gives Hull–White, and two exponential factors with correlated loadings give G2++. `GaussianHJM.from_hull_white` covers piecewise-constant volatility: its jump times are passed as `breakpoints`, so the quadrature never straddles a jump. `hump_vol` gives $(\alpha + \beta\tau)e^{-\gamma\tau}$ in time to maturity $\tau = T - t$. It peaks at $\tau = 1/\gamma - \alpha/\beta$, the hump seen in historical forward-rate volatilities.

### 10.2 Simulating the curve

`hjm_paths` evolves log deflated bonds $Z(t, T_j) = P(t, T_j)/B(t)$ on a maturity grid. Each step freezes $\Sigma$ at its start, and the increment $-\tfrac12|\Sigma|^2\Delta t - \Sigma\cdot\Delta W$ has $E[e^{\cdot}] = 1$ exactly. Every $Z(\cdot, T_j)$ is then a discrete martingale, and the curve is repriced with no discretisation bias. The deflator is $Z(t, t)$. The remaining error sits in option prices, through the left-point variance $\sum_k |\Sigma(t_k, S) - \Sigma(t_k, T)|^2\Delta t$. The simulated payoff is priced exactly by Black at that variance, which converges at first order.

### 10.3 Cheyette

A separable volatility $\sigma_r(t)\,e^{-\kappa(T - t)}$ makes HJM Markovian in two states. Cheyette's model (Andersen & Piterbarg, ch. 13) lets the short-rate volatility depend on the state, $\sigma_r(t, x)$, while keeping that structure:

$$
dx = (y - \kappa x)\,dt + \sigma_r\,dW, \qquad dy = (\sigma_r^2 - 2\kappa y)\,dt, \qquad P(t, T) = \frac{P(0, T)}{P(0, t)}e^{-G x - \frac12 G^2 y},
$$

with $G = (1 - e^{-\kappa(T - t)})/\kappa$. `Cheyette` uses the linear local volatility $\sigma_r = \max(\sigma(t) + \beta x, 0)$. With $\beta = 0$, $y$ is deterministic and the model is Hull–White. A positive $\beta$ raises the volatility as rates rise, which produces the upward-sloping normal-vol skew of rate markets.

### 10.4 Engines

`cheyette_paths` freezes $\sigma_r$ and $y$ over each step and solves the step exactly. For $\beta = 0$ this is the exact Hull–White transition. `CheyettePDE` solves the two-dimensional pricing equation. Because $y$ has no diffusion, each step splits into semi-Lagrangian advection in $y$ and an implicit solve in $x$ on each $y$-slice. Exercise dates apply $\max(V, g)$, so the PDE prices American and Bermudan claims as well.

### 10.5 Swaptions and calibration

`cheyette_swaption` follows Andersen & Piterbarg's projection. Freezing $y$ at $\bar y(t)$, the swap-rate volatility $S_x\sigma_r$ is linearised into a displaced diffusion $\lambda(t)(b(t)S + (1 - b(t))S_0)$. The skew is averaged with weights $\lambda^2(t)\int_0^t\lambda^2$ (Piterbarg), and the swaption is priced in displaced Black. For a 2Y×5Y swaption with $\kappa = 0.05$ and $\sigma = 1\%$ on the flat 3% curve, the averaged skew is $\bar b = 0.04$ at $\beta = 0$ and $1.21$ at $\beta = 0.3$. With $\beta = 0.3$ the PDE gives normal vols of $70.6$, $85.9$ and $100.7$ bp at strikes $2.5\%$, $3.46\%$ and $4.5\%$, and the approximation gives $71.0$, $86.5$ and $101.5$ bp, within 1 bp across the smile. With $\beta = 0$ the same strikes are nearly flat, at $86.7$, $87.1$ and $87.6$ bp. `calibrate_cheyette` fits one $\sigma$ level per expiry, plus $\beta$, to a grid of strikes at each expiry using this formula.

## 11. LIBOR and forward market models

### 11.1 Parametrisation

The LIBOR market model (Brace, Gątarek & Musiela; Jamshidian) takes the simply compounded forwards $L_i$ on a tenor structure $T_0 < \dots < T_n$ as its state. Each one is lognormal under its own payment measure, possibly displaced by $\delta$:

$$
dL_i = \sigma_i(t)\,(L_i + \delta)\,dW_i^{T_{i+1}}, \qquad d\langle W_i, W_j\rangle = \rho_{ij}\,dt,
$$

so caplets are Black formulas, exactly as quoted. `LiborMarketModel` uses Rebonato's time-homogeneous shape $\sigma_i(t) = k_i f(T_i - t)$ with $f(\tau) = (a + b\tau)e^{-c\tau} + d$ (`AbcdVol`). Its hump sits at $\tau = 1/c - a/b$, which is $1.72$ years for $(a, b, c, d) = (0.05, 0.09, 0.44, 0.11)$. `AbcdVol.covariance` integrates $f(T_i - t)f(T_j - t)$ in closed form. The correlation is `exponential_correlation`, $e^{-\beta|T_i - T_j|}$, or Rebonato's `rebonato_correlation`, $\rho_\infty + (1 - \rho_\infty)e^{-\beta|T_i - T_j|}$. `reduce_rank` keeps its leading principal components and renormalises the diagonal. On 20 semi-annual forwards with $\beta = 0.1$ and $\rho_\infty = 0.4$, three factors leave a largest correlation error of $0.089$.

### 11.2 Simulation

`lmm_paths` simulates the log displaced forwards under the spot LIBOR measure, whose numeraire rolls over the shortest bond. Forward $i$ then has drift

$$
\sigma_i\sum_{j = q(t)}^{i}\frac{\tau_j\rho_{ij}\sigma_j(L_j + \delta)}{1 + \tau_j L_j},
$$

summed over the forwards still alive. Each step draws Gaussian increments with the exact integrated covariance of the step, so only the drift is discretised. It is either frozen at the start of the step or averaged with a predicted end value (predictor–corrector). The deflator $P(0, T_0)/\prod_{j<k}(1 + \tau_j L_j(T_j))$ prices any payoff fixed on the tenor dates. Deflated bonds are martingales, and caplets reproduce Black within Monte Carlo error.

### 11.3 Swaptions and calibration

The forward swap rate over $[T_\alpha, T_\beta]$ is a weighted average $S = \sum_i w_i L_i$, with $w_i = \tau_i P(0, T_{i+1})/A$. Freezing the weights and the forwards gives Rebonato's formula (`LiborMarketModel.swaption_vol`):

$$
\sigma_{\alpha\beta}^2 T_\alpha = \frac{1}{(S + \delta)^2}\sum_{i, j = \alpha}^{\beta - 1} w_i w_j (L_i + \delta)(L_j + \delta)\,\rho_{ij}\int_0^{T_\alpha}\sigma_i\sigma_j\,dt.
$$

The swaption is then Black on the annuity. On the flat 3% curve, the abcd shape above with the Rebonato correlation gives a 2Y×5Y volatility of $17.84\%$, at an ATM swap rate of $3.02\%$. Its caplets carry between $18.3\%$ and $19.7\%$. With perfect correlation the swaption volatility would be $18.66\%$. Simulation agrees with the formula within Monte Carlo error (three standard errors plus 1%) at strikes of 2.5%, ATM and 4%.

`calibrate_lmm` calibrates in two stages. `fit_abcd` fits the shape to the caplet volatilities, and the $k_i$ then make every caplet exact. The correlation parameters $(\beta, \rho_\infty)$ are fitted to swaption volatilities through Rebonato's formula, so no simulation is needed.
