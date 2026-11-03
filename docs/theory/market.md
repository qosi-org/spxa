# Market Conventions and Building Blocks

`spxa.market` holds the pieces that `spxa.rates` and `spxa.vol` share: dates and accrual conventions, holiday calendars, payment schedules, discount and forward curves, the standard option formulas, implied volatility, option quotes, static-arbitrage checks and forwards implied by put–call parity. Everything works with plain `datetime.date` objects and year fractions; no external market-data library is needed.

| Object | Implementation |
| :--- | :--- |
| Dates and tenors | `to_date`, `add_months`, `add_tenor`, `Tenor` |
| Day counts | `DayCount`, `year_fraction` |
| Calendars | `get_calendar` (TARGET, US-FED, US-SIFMA, UK), `JointCalendar` |
| Schedules | `adjust`, `Schedule`, `AccrualPeriod` |
| Curves and forwards | `FlatCurve`, `InterpolatedDiscountCurve`, `EquityForward` |
| Option formulas | `black_price`, `bachelier_price`, `shifted_black_price`, `black_scholes_price` and their Greeks |
| Implied volatility | `implied_black_volatility`, `implied_normal_volatility` |
| Quotes and arbitrage | `OptionChain`, `OptionSurfaceQuotes`, `check_surface_quotes` |
| Parity and carry | `implied_forward`, `ImpliedForwardCurve` |

---

## 1. Dates, day counts and calendars

### 1.1 Adding months

Adding $n$ months to a date keeps the day of the month and clips it to the length of the target month, so 31 January plus one month is 28 or 29 February. With the **end-of-month rule** a date that is the last day of its month maps to the last day of the target month: 28 February 2023 plus one month is 31 March 2023 with the rule and 28 March without it. Tenors such as `"3M"`, `"2Y"`, `"1W"` and `"10D"` are parsed by `Tenor`.

### 1.2 Day counts

A day count turns a period $[d_1, d_2)$ into an accrual fraction. `DayCount` implements:

| Convention | Fraction |
| :--- | :--- |
| ACT/360, ACT/365F | actual days over 360 or 365 |
| ACT/ACT ISDA | days in each calendar year over that year's length |
| ACT/ACT ICMA | days over (frequency × days in the reference coupon period) |
| 30/360 Bond Basis, 30/360 US, 30E/360, 30E/360 ISDA | $\bigl(360\,\Delta Y + 30\,\Delta M + \Delta D\bigr)/360$ with convention-specific day adjustments |
| BUS/252 | business days over 252 (needs a calendar) |

The tests reproduce the ISDA examples. For a regular semi-annual period from 1 November 2003 to 1 May 2004, ACT/ACT ISDA gives $61/365 + 121/366 = 0.497724380567$ while ICMA gives exactly $0.5$. For the long first coupon from 15 August 2002 to 15 July 2003, ISDA gives $0.915068493151$ and ICMA $0.915760869565$. The 30/360 variants differ only at month ends: from 28 February 2007 to 31 March 2007, Bond Basis counts 33 days, 30/360 US counts 30, 30E/360 counts 32 and 30E/360 ISDA counts 30.

### 1.3 Calendars

Holiday rules are computed from first principles: fixed dates, weekend observance rules, nth-weekday rules and Easter (the anonymous Meeus/Jones/Butcher algorithm). The TARGET calendar closes on New Year's Day, Good Friday, Easter Monday, 1 May, Christmas and 26 December, which gives 256 TARGET business days in 2024. `JointCalendar` combines calendars, treating a day as a holiday if any member (`rule="holiday_if_any"`, the default) or every member (`"holiday_if_all"`) closes.

### 1.4 Business-day conventions and schedules

`adjust(date, calendar, convention)` implements Following, Modified Following, Preceding, Modified Preceding and Unadjusted. `Schedule` generates unadjusted dates backwards (or forwards) from the termination date at the tenor of the frequency, then adjusts them. It supports front and back stubs, short or long, explicit first and next-to-last dates and the end-of-month rule. Each `AccrualPeriod` keeps the reference period that ACT/ACT ICMA needs for stubs.

---

## 2. Curves and forwards

A `YieldCurve` returns discount factors $P(0, t)$; zero rates and simple or compounded forward rates are derived from them. `InterpolatedDiscountCurve` is linear in $\log P$ (piecewise-constant instantaneous forwards) and extrapolates the last forward flat.

`EquityForward` combines spot $S$, a funding (repo) curve $P_f$, a continuous dividend curve $Q$ and discrete dividends. Cash dividends $d_i$ at $t_i$ and proportional dividends $\delta_j$ give

$$
F(t) = \frac{Q(t)}{P_f(t)} \Bigl[ S\prod_{t_j \le t}(1-\delta_j) - \sum_{t_i \le t} d_i\,\frac{P_f(t_i)}{Q(t_i)}\prod_{t_i < t_j \le t}(1-\delta_j) \Bigr],
$$

so a cash dividend is reduced only by the proportional dividends paid after it.

---

## 3. Option formulas

With forward $F$, strike $K$, expiry $T$, discount factor $P$ and $\eta = \pm 1$ for calls and puts,

$$
\text{Black: } P\eta\bigl[F\Phi(\eta d_1) - K\Phi(\eta d_2)\bigr], \quad d_{1,2} = \frac{\ln(F/K)}{\sigma\sqrt T} \pm \tfrac12\sigma\sqrt T,
$$

$$
\text{Bachelier: } P\bigl[\eta(F-K)\Phi(\eta d) + \sigma_N\sqrt T\,\varphi(d)\bigr], \quad d = \frac{F-K}{\sigma_N\sqrt T}.
$$

Shifted Black applies Black to $F + s$ and $K + s$, allowing negative rates. Each formula returns its Greeks analytically; the tests check every Greek against finite differences and identities such as volga $=$ vega $\cdot d_1 d_2/\sigma$. At the money the Bachelier price is $P\sigma_N\sqrt{T/(2\pi)}$.

---

## 4. Implied volatility

### 4.1 Black

Write $x = \ln(F/K)$, $s = \sigma\sqrt T$ and $\beta = V/(P\sqrt{FK})$. The normalised call price is

$$
b(x, s) = e^{x/2}\Phi\!\left(\frac{x}{s} + \frac s2\right) - e^{-x/2}\Phi\!\left(\frac{x}{s} - \frac s2\right),
$$

and puts follow from $b(x, s) - b(-x, s) = e^{x/2} - e^{-x/2}$, so every price reduces to an out-of-the-money value. Following Jäckel's normalised formulation, the solver works with $\log b$, evaluated without cancellation:

- directly when the two terms are well separated;
- through the scaled complementary error function $\operatorname{erfcx}$ in the far wings;
- otherwise through the positive integral $b(x, s) = \int_0^s e^{x/2}\varphi(x/u + u/2)\,du$, computed by composite Gauss–Legendre quadrature on geometric panels.

The equation $\log b(x, s) = \log\beta$ is solved by a safeguarded Newton iteration that keeps a shrinking bracket, starting near the inflection point $s = \sqrt{2|x|}$. At the money, $s = 2\sqrt 2\,\operatorname{erf}^{-1}(\beta)$ exactly. Against 50-digit reference prices the round trip is accurate to better than $10^{-13}$ relative error for out-of-the-money options over $x \in [-15, 15]$ and $s \in [10^{-4}, 15]$.

### 4.2 Bachelier

The out-of-the-money time value is $\sigma_N\sqrt T\,\psi(-u)$ with $u = |F-K|/(\sigma_N\sqrt T)$ and $\psi(d) = d\Phi(d) + \varphi(d)$. Writing $\psi(-u) = \varphi(u)\bigl(1 - u M(u)\bigr)$ with the Mills ratio $M$ cancels catastrophically for large $u$, so there the solver uses the positive representation

$$
\psi(-u) = \frac{\varphi(u)}{u}\int_0^\infty M\!\left(u + \frac wu\right) e^{-w - w^2/(2u^2)}\,dw
$$

with Gauss–Laguerre quadrature. The solver iterates on $u$; at the money $\sigma_N\sqrt T = \sqrt{2\pi}\,V/P$. Round trips are accurate to $5\times 10^{-14}$.

Prices outside the no-arbitrage bounds return `NaN`, or raise with `errors="raise"`.

---

## 5. Quotes, static arbitrage and parity

`OptionChain` stores the bids and asks of one expiry and produces mids, out-of-the-money selections and implied volatilities; `OptionSurfaceQuotes` groups chains by expiry.

**Static arbitrage.** Call prices of one expiry are free of static arbitrage when $P(F-K)^+ \le C(K) \le PF$, when $0 \le C(K_1) - C(K_2) \le P(K_2 - K_1)$ for $K_1 < K_2$, and when $C$ is convex in $K$. `check_call_prices` reports each violation with the profit of the corresponding portfolio of quoted options. Across expiries, the normalised price $C(kF_T, T)/(P_T F_T)$ must be non-decreasing in $T$ at fixed forward moneyness $k$ when dividends are proportional. `check_calendar_arbitrage` interpolates the later expiry only along chords: a convex combination of calls dominates the call at the combined strike, so every reported violation is a genuine arbitrage.

**Put–call parity.** $C(K) - P(K) = P(F - K)$ is linear in $K$, with slope $-P$ and intercept $PF$. `implied_forward` fits it by weighted least squares, with weights equal to the inverse squared bid–ask width of each synthetic forward. It can restrict the fit to strikes near the money and rejects outliers beyond a multiple of the robust scale. On exact prices it recovers $F$ to $10^{-13}$ and $P$ to $10^{-12}$ relative error.

**Carry.** A forward fixes only the net carry $\ln(F/S)/T$. Given the funding discount factor, the dividend yield is $q = -\ln(F P_f/S)/T$. `ImpliedForwardCurve` interpolates market forwards linearly in $\ln(F/S)$ and converts to an `EquityForward` that reproduces them at every time.

---

## References

- ISDA (1998). *"EMU and Market Conventions: Recent Developments"* (ACT/ACT examples).
- Jäckel, P. (2015). Let's be rational. *Wilmott*, 2015(75), 40–53.
- Jäckel, P. (2017). Implied normal volatility. *Wilmott*, 2017(90), 54–57.
- Carr, P. & Madan, D. (2005). A note on sufficient conditions for no arbitrage. *Finance Research Letters*, 2(3), 125–130.
- Gatheral, J. (2006). *The Volatility Surface*. Wiley.
