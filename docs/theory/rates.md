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
