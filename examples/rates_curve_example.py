"""
Building EUR discount and projection curves with spxa.rates.

Bootstraps an ESTR curve from OIS quotes under several interpolation schemes,
then a EURIBOR 6M curve on top of it (dual-curve), adds a turn-of-year spread,
and turns the bootstrap Jacobian into quote sensitivities, checked against a
full rebuild.

Run with: python examples/rates_curve_example.py
"""

import numpy as np

from spxa.rates import (
    ESTR,
    EURIBOR6M,
    FRA,
    Deposit,
    DepositHelper,
    FRAHelper,
    LinearZero,
    LogLinearDiscount,
    MonotoneConvex,
    MonotoneCubicRT,
    SwapHelper,
    bootstrap_curve,
    compare_schemes,
    ois_swap,
    turn_of_year,
    vanilla_swap,
)

REF = "2024-01-15"
# illustrative mid quotes (not market data)
OIS = {"3M": 3.905, "6M": 3.83, "1Y": 3.56, "2Y": 3.02, "3Y": 2.78, "5Y": 2.60,
       "7Y": 2.56, "10Y": 2.58, "15Y": 2.62, "20Y": 2.58, "30Y": 2.42}  # fmt: skip
E6_FRA = {"6M": 3.62, "12M": 3.08}
E6_SWAP = {"2Y": 3.18, "3Y": 2.98, "5Y": 2.82, "7Y": 2.79, "10Y": 2.81, "15Y": 2.84,
           "20Y": 2.79, "30Y": 2.62}  # fmt: skip

print("=" * 72)
print("spxa.rates — bootstrapping EUR curves")
print("=" * 72)


def estr_helpers() -> list:
    return [SwapHelper(ois_swap(REF, t, 0.0, ESTR), q / 100) for t, q in OIS.items()]


# 1. ESTR under four schemes
print("\n1. ESTR discount curve from 11 OIS quotes")
results = {}
for scheme in (LogLinearDiscount(), LinearZero(), MonotoneCubicRT(), MonotoneConvex()):
    res = bootstrap_curve(estr_helpers(), REF, scheme)
    results[repr(scheme)] = res
    print(f"   {repr(scheme):34s} max residual < 1e-12: {res.max_residual < 1e-12}")
ois = results["MonotoneConvex(positivity=False)"].curve
times, zeros = ois.node_times, ois.node_zero_rates
print("   scheme                             min forward   locality")
for row in compare_schemes(times, zeros, [LogLinearDiscount(), LinearZero(), MonotoneConvex()]):
    print(f"   {row.scheme:34s} {row.min_forward:10.4%}   {row.locality}")

# 2. EURIBOR 6M on the ESTR curve
print("\n2. EURIBOR 6M projection curve (dual-curve bootstrap)")
proj_h: list = [DepositHelper(Deposit.from_tenor(REF, "6M", 0.0), 0.0393, curve="EUR-6M")]
for start, q in E6_FRA.items():
    proj_h.append(FRAHelper(FRA.from_index(EURIBOR6M, REF, start, 0.0), q / 100, curve="EUR-6M"))
for t, q in E6_SWAP.items():
    proj_h.append(SwapHelper(vanilla_swap(REF, t, 0.0, EURIBOR6M), q / 100, forward="EUR-6M"))
e6 = bootstrap_curve(proj_h, REF, MonotoneConvex(), name="EUR-6M", known={"discount": ois})
print(f"   {len(proj_h)} helpers, max residual < 1e-12: {e6.max_residual < 1e-12}")
for t in (2.0, 5.0, 10.0, 30.0):
    basis = (e6.curve.zero_rate(t) - ois.zero_rate(t)) * 1e4
    print(
        f"   {t:4.0f}Y zero: ESTR {ois.zero_rate(t):.4%}  6M {e6.curve.zero_rate(t):.4%}  basis {basis:5.1f} bp"
    )
swap10 = vanilla_swap(REF, "10Y", 0.0281, EURIBOR6M)
print(f"   10Y payer at the quoted 2.81% prices to zero: {abs(swap10.pv(ois, e6.curve)) < 1e-12}")

# 3. A turn-of-year spread of 25 bp on ESTR
print("\n3. Turn of year 2024 (+25 bp on the overnight forward)")
turn = turn_of_year(2024, 0.0025)
turned = bootstrap_curve(estr_helpers(), REF, MonotoneConvex(), spreads=[turn]).curve
print(f"   turn from {turn.start} to {turn.end}")
for d in ("2024-12-30", "2024-12-31", "2025-01-02"):
    ta = turned.time(np.datetime64(d))
    jump = (turned.forward_rate(ta, ta + 1 / 365) - ois.forward_rate(ta, ta + 1 / 365)) * 1e4
    print(f"   O/N forward from {d}: {jump:+6.1f} bp versus the plain curve")

# 4. Risk: from the Jacobian to quote sensitivities
print("\n4. Quote sensitivities from the bootstrap Jacobian (log-linear ESTR)")
ll = results["LogLinearDiscount()"]
dxdq = ll.node_sensitivities()
j = list(OIS).index("10Y")
h = 1e-4
bumped = [
    SwapHelper(hh.swap, hh.quote + (h if i == j else 0.0)) for i, hh in enumerate(estr_helpers())
]
rebuilt = bootstrap_curve(bumped, REF, LogLinearDiscount()).node_values
predicted = ll.node_values + h * dxdq[:, j]
err = np.max(np.abs(predicted - rebuilt))
print(
    f"   +1 bp on the 10Y quote: Jacobian vs rebuild agree to second order (< 1e-6): {err < 1e-6}"
)
print(f"   nodes before 10Y unmoved: {bool(np.all(np.abs(dxdq[:j, j]) < 1e-10))}")
