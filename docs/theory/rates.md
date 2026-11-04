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
