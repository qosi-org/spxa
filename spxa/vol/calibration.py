r"""Calibration of SVI slices.

Quasi-explicit SVI (Zeliade; De Marco & Martini 2009)
    In the variable :math:`y = (k - m)/\sigma` raw SVI reads

    .. math:: w = a + d\,y + c\sqrt{y^2 + 1}, \qquad c = b\sigma,\; d = \rho b\sigma,

    which is *linear* in :math:`(a, c, d)` for fixed :math:`(m, \sigma)`.
    The admissible set :math:`0 \le |d| \le c`, :math:`c + |d| \le s\sigma`
    (wing slopes :math:`b(1 \pm \rho) \le s`) and :math:`0 \le a \le \max w`
    becomes a box in :math:`(a, u, v) = (a, c + d, c - d)`:
    :math:`u, v \in [0, s\sigma]`. The inner problem is therefore a
    bounded linear least-squares problem solved exactly, and only the two
    parameters :math:`(m, \sigma)` are searched numerically.

    The paper's :math:`a \ge 0` is stronger than the positivity of the
    minimum variance :math:`a + \sqrt{uv} \ge 0`. The inner solve first
    allows :math:`a \ge -\max w`; if the solution has non-negative minimum
    variance it is optimal for the exact constraint, otherwise the
    :math:`a \ge 0` box is used.

    ``max_slope`` :math:`s` defaults to 2, Lee's moment bound on the
    asymptotic slope of total variance; De Marco & Martini use the weaker
    Rogers–Tehranchi bound :math:`s = 4` (pass ``max_slope=4``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import lsq_linear, minimize, minimize_scalar

from spxa.vol.svi import SVIRaw

__all__ = ["SVIFit", "calibrate_svi", "svi_inner_fit"]


def _arrays(k: Any, w: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kk = np.asarray(k, float).ravel()
    ww = np.asarray(w, float).ravel()
    if kk.shape != ww.shape or kk.size < 5:
        raise ValueError("need at least 5 (k, w) pairs of equal length")
    if np.any(ww <= 0) or not np.all(np.isfinite(ww)):
        raise ValueError("total variances must be positive and finite")
    if weights is None:
        sw = np.ones_like(kk)
    else:
        wt = np.asarray(weights, float).ravel()
        if wt.shape != kk.shape or np.any(wt < 0):
            raise ValueError("weights must be non-negative, one per point")
        sw = np.sqrt(wt)
    return kk, ww, sw


def svi_inner_fit(
    k: Any,
    w: Any,
    m: float,
    sigma: float,
    weights: Any = None,
    max_slope: float = 2.0,
) -> tuple[SVIRaw, float]:
    """Best raw SVI for fixed ``(m, sigma)``; returns ``(slice, weighted SSE)``."""
    kk, ww, sw = _arrays(k, w, weights)
    return _inner(kk, ww, sw, m, sigma, max_slope)


def _inner(
    kk: np.ndarray, ww: np.ndarray, sw: np.ndarray, m: float, sigma: float, max_slope: float
) -> tuple[SVIRaw, float]:
    y = (kk - m) / sigma
    z = np.sqrt(y * y + 1.0)
    # w = a + u (z + y)/2 + v (z - y)/2 with u = c + d, v = c - d
    A = np.column_stack([np.ones_like(y), 0.5 * (z + y), 0.5 * (z - y)]) * sw[:, None]
    hi = max_slope * sigma
    top = float(ww.max())
    # first allow a < 0; keep it if the minimum variance a + sqrt(u v) stays
    # non-negative (then it is optimal), otherwise fall back to a >= 0
    res = lsq_linear(A, ww * sw, bounds=([-top, 0.0, 0.0], [top, hi, hi]), method="bvls")
    if res.x[0] + np.sqrt(max(res.x[1] * res.x[2], 0.0)) < 0.0:
        res = lsq_linear(A, ww * sw, bounds=([0.0, 0.0, 0.0], [top, hi, hi]), method="bvls")
    a, u, v = (float(x) for x in res.x)
    c, d = 0.5 * (u + v), 0.5 * (u - v)
    sse = float(np.sum((A @ res.x - ww * sw) ** 2))
    if c <= 0.0:
        # flat smile: any rho; choose rho = 0 and a vanishing b
        return SVIRaw(a, 0.0, 0.0, m, sigma), sse
    rho = float(np.clip(d / c, -1.0 + 1e-15, 1.0 - 1e-15))
    return SVIRaw(a, c / sigma, rho, m, sigma), sse


def _negative_minima(g: Any, kv: np.ndarray, tol: float = 1e-12) -> list[float]:
    """Locations of the local minima of ``g`` where it is negative.

    Every discrete local minimum on ``kv`` with a small value is refined by
    a bounded scalar search between its neighbours, so dips narrower than
    the scan spacing are found.
    """
    gv = g(kv)
    out: list[float] = []
    for i in range(kv.size):
        lo, hi = max(i - 1, 0), min(i + 1, kv.size - 1)
        if gv[i] > gv[lo] or gv[i] > gv[hi] or gv[i] > 1e-3:
            continue
        if lo == i or hi == i:
            kmin, gmin = float(kv[i]), float(gv[i])
        else:
            res = minimize_scalar(
                lambda t: float(g(np.array([t]))[0]),
                bounds=(float(kv[lo]), float(kv[hi])),
                method="bounded",
                options={"xatol": 1e-12},
            )
            kmin, gmin = float(res.x), float(res.fun)
            if gv[i] < gmin:
                kmin, gmin = float(kv[i]), float(gv[i])
        if gmin < -tol:
            out.append(kmin)
    return out


@dataclass(frozen=True)
class SVIFit:
    """Result of :func:`calibrate_svi`."""

    params: SVIRaw
    rmse: float
    sse: float
    nfev: int
    success: bool


def _slsqp_converged(r: Any, cons: list[dict[str, Any]], bounds: Any, tol: float = 1e-10) -> bool:
    """Whether an SLSQP result is a solution.

    Besides ``success``, this accepts exit mode 8 ("positive directional
    derivative for linesearch") at a feasible point. With the very tight
    ``ftol`` used here, SLSQP returns mode 8 when the last line search cannot
    lower the objective in floating point, i.e. at the optimum. Whether a run
    ends in mode 0 or mode 8 depends on the SciPy version and the CPU's
    rounding.
    """
    if r.success:
        return True
    if r.status != 8:
        return False
    x = np.asarray(r.x, float)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    if np.any(x < lo - tol) or np.any(x > hi + tol):
        return False
    return all(float(np.min(np.atleast_1d(c["fun"](x)))) >= -tol for c in cons)


def calibrate_svi(
    k: Any,
    w: Any,
    weights: Any = None,
    max_slope: float = 2.0,
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
    m_bounds: Optional[tuple[float, float]] = None,
    n_starts: int = 4,
) -> SVIFit:
    """Fit raw SVI to total variances ``w`` at log-moneyness ``k``.

    The outer search over :math:`(m, \\sigma)` starts from the best points
    of a coarse grid and refines each with bounded Nelder–Mead; the inner
    linear problem is solved exactly (see the module docstring).
    """
    kk, ww, sw = _arrays(k, w, weights)
    if max_slope <= 0:
        raise ValueError("max_slope must be positive")
    span = float(kk.max() - kk.min())
    if m_bounds is None:
        m_bounds = (float(kk.min()) - span, float(kk.max()) + span)
    lo = np.array([m_bounds[0], np.log(sigma_bounds[0])])
    hi = np.array([m_bounds[1], np.log(sigma_bounds[1])])
    nfev = 0

    def obj(x: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        m, ls = np.clip(x, lo, hi)
        return _inner(kk, ww, sw, float(m), float(np.exp(ls)), max_slope)[1]

    ms = np.linspace(
        m_bounds[0] + 0.25 * (m_bounds[1] - m_bounds[0]),
        m_bounds[1] - 0.25 * (m_bounds[1] - m_bounds[0]),
        9,
    )
    ls = np.linspace(lo[1], hi[1], 9)
    grid = sorted(((obj(np.array([m, s])), m, s) for m in ms for s in ls))
    best_x, best_f = np.array(grid[0][1:]), grid[0][0]
    ok = False
    for _, m0, s0 in grid[:n_starts]:
        r = minimize(
            obj,
            np.array([m0, s0]),
            method="Nelder-Mead",
            bounds=list(zip(lo, hi)),
            options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 4000},
        )
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
        ok = ok or bool(r.success)
    m, s = np.clip(best_x, lo, hi)
    params, sse = _inner(kk, ww, sw, float(m), float(np.exp(s)), max_slope)
    wsum = float(np.sum(sw * sw))
    return SVIFit(params, float(np.sqrt(sse / wsum)) if wsum > 0 else 0.0, sse, nfev, ok)


def calibrate_svi_butterfly_free(
    k: Any,
    w: Any,
    weights: Any = None,
    max_slope: float = 2.0,
    grid: Any = None,
    initial: Optional[SVIRaw] = None,
    min_g: float = 1e-8,
    lower: Optional[SVIRaw] = None,
    min_gap: float = 0.0,
) -> SVIFit:
    r"""Raw SVI fit with Durrleman's condition imposed on a grid.

    Starts from the quasi-explicit fit (or ``initial``) and solves

    .. math:: \min \sum_i \omega_i\,(w(k_i) - w_i)^2 \quad\text{s.t.}\quad
              g(k_j) \ge g_{\min},\; b(1 \pm \rho) \le s,\;
              a + b\sigma\sqrt{1 - \rho^2} > 0

    with SLSQP and analytic :math:`g`, on ``grid`` (default 401 points
    covering the data and twice its width either side), in the spirit of
    Gatheral & Jacquier (2014, §5). After each solve :math:`g` is scanned
    on a ten times finer grid, its negative local minima are located by a
    bounded scalar search and added as constraints (cutting planes) until
    none remain. Slices that already satisfy the condition are returned
    unchanged up to the solver tolerance.

    With ``lower`` (the previous expiry's slice) the fit is also kept free
    of calendar arbitrage against it: :math:`w(k_j) - w_{\text{lower}}(k_j)
    \ge` ``min_gap`` on the same grid with the same cutting-plane
    refinement, and the wing slopes satisfy :math:`b(1 \pm \rho) \ge
    b_{\text{lower}}(1 \pm \rho_{\text{lower}})`, which controls the sign
    of the difference beyond the grid.
    """
    from spxa.vol.arbitrage import durrleman_g

    kk, ww, sw = _arrays(k, w, weights)
    if grid is None:
        span = float(kk.max() - kk.min())
        grid = np.linspace(kk.min() - 2 * span, kk.max() + 2 * span, 401)
    gk = np.asarray(grid, float)
    p0 = initial if initial is not None else calibrate_svi(kk, ww, weights, max_slope).params
    scale = float(np.sum((ww * sw) ** 2)) or 1.0
    nfev = 0

    def unpack(x: np.ndarray) -> tuple[float, float, float, float, float]:
        return float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])

    def wfun(x: np.ndarray, kv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b, rho, m, sig = unpack(x)
        y = kv - m
        r = np.sqrt(y * y + sig * sig)
        return a + b * (rho * y + r), b * (rho + y / r), b * sig * sig / r**3

    def obj(x: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        return float(np.sum(((wfun(x, kk)[0] - ww) * sw) ** 2)) / scale

    def g_on(x: np.ndarray, kv: np.ndarray) -> np.ndarray:
        wv, d1, d2 = wfun(x, kv)
        return np.asarray(durrleman_g(kv, np.maximum(wv, 1e-12), d1, d2))

    def gap_on(x: np.ndarray, kv: np.ndarray) -> np.ndarray:
        assert lower is not None
        return wfun(x, kv)[0] - np.asarray(lower.total_variance(kv), float)

    cal: list[dict[str, Any]] = []
    if lower is not None:
        lr, ll = lower.right_slope, lower.left_slope
        cal = [
            {"type": "ineq", "fun": lambda x: x[1] * (1 + x[2]) - lr},
            {"type": "ineq", "fun": lambda x: x[1] * (1 - x[2]) - ll},
        ]
    base = [
        *cal,
        {"type": "ineq", "fun": lambda x: max_slope - x[1] * (1 + x[2])},
        {"type": "ineq", "fun": lambda x: max_slope - x[1] * (1 - x[2])},
        {
            "type": "ineq",
            "fun": lambda x: x[0] + x[1] * x[4] * np.sqrt(max(1 - x[2] ** 2, 0.0)) - 1e-10,
        },
    ]
    span_k = float(kk.max() - kk.min())
    bounds = [
        (-float(ww.max()), float(ww.max())),
        (0.0, max_slope),
        (-0.999999, 0.999999),
        (float(kk.min()) - span_k, float(kk.max()) + span_k),
        (1e-4, 10.0),
    ]
    fine = np.linspace(gk.min(), gk.max(), 10 * gk.size)
    x = p0.to_array()
    active = gk
    active_cal = gk
    # a touching calendar constraint dips O(h^2) between constraint points; aiming
    # 2e-12 above min_gap and cutting dips deeper than 1e-12 keeps w - lower >= min_gap
    floor = min_gap + 2e-12
    ok = False
    # cutting planes: points where g (or the calendar gap) dips between grid nodes
    # join the constraint set
    for _ in range(12 if lower is None else 30):
        pts, cpts = active, active_cal
        cons = [{"type": "ineq", "fun": lambda x, pts=pts: g_on(x, pts) - min_g}, *base]
        if lower is not None:
            cons.append({"type": "ineq", "fun": lambda x, c=cpts: gap_on(x, c) - floor})
        r = minimize(
            obj,
            x,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 500},
        )
        x = r.x
        cuts = _negative_minima(lambda kv, x=x: g_on(x, kv), fine)
        ccuts = (
            _negative_minima(lambda kv, x=x: gap_on(x, kv) - floor, fine, tol=1e-12)
            if lower is not None
            else []
        )
        if not cuts and not ccuts:
            ok = _slsqp_converged(r, cons, bounds)
            break
        active = np.union1d(active, cuts)
        active_cal = np.union1d(active_cal, ccuts)
    params = SVIRaw(*unpack(x))
    sse = obj(x) * scale
    wsum = float(np.sum(sw * sw))
    return SVIFit(params, float(np.sqrt(sse / wsum)), sse, nfev, ok)


def isotonic_increasing(y: Any, weights: Any = None) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    yy = np.asarray(y, float).ravel()
    wt = np.ones_like(yy) if weights is None else np.asarray(weights, float).ravel()
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for v, w_ in zip(yy, wt):
        vals.append(float(v))
        wts.append(float(w_))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            n2 = sizes[-2] + sizes[-1]
            del vals[-1], wts[-1], sizes[-1]
            vals[-1], wts[-1], sizes[-1] = v2, w2, n2
    return np.repeat(vals, sizes)


@dataclass(frozen=True)
class SSVIFit:
    """Result of :func:`calibrate_ssvi`."""

    surface: Any
    thetas: np.ndarray
    rmse: float
    success: bool


def calibrate_ssvi(
    expiries: Any,
    ks: Any,
    ws: Any,
    phi: str = "power_law",
    no_arbitrage: bool = True,
) -> SSVIFit:
    r"""Fit an SSVI surface to total-variance slices.

    1. The ATM total variance of each slice is read from a quasi-explicit
       SVI fit at :math:`k = 0` and made non-decreasing in :math:`T` by
       isotonic regression (no calendar arbitrage at the money).
    2. :math:`\rho` and the curvature parameters are fitted to all quotes
       by SLSQP. With ``no_arbitrage`` the sufficient conditions are
       imposed: :math:`\eta(1 + |\rho|) \le 2`, :math:`\gamma \le 1/2` for
       the power law (Gatheral–Jacquier Remark 4.4) or
       :math:`\lambda \ge (1 + |\rho|)/4` for the Heston-like
       :math:`\varphi` (Corollary 4.1).

    Parameters
    ----------
    expiries : sequence of float
        Increasing expiries.
    ks, ws : sequences of arrays
        Log-moneyness and total variance per expiry.
    phi : ``"power_law"`` or ``"heston"``
    """
    from spxa.vol.ssvi import SSVI, ATMVarianceCurve, HestonLikePhi, PowerLawPhi

    T = np.asarray(expiries, float)
    if T.ndim != 1 or len(ks) != T.size or len(ws) != T.size:
        raise ValueError("need one (k, w) slice per expiry")
    if np.any(np.diff(T) <= 0) or T[0] <= 0:
        raise ValueError("expiries must be positive and increasing")
    if phi not in ("power_law", "heston"):
        raise ValueError("phi must be 'power_law' or 'heston'")
    slices = [(np.asarray(k, float), np.asarray(w, float)) for k, w in zip(ks, ws)]
    raw_theta = np.array([float(calibrate_svi(k, w, max_slope=4.0).params(0.0)) for k, w in slices])
    thetas = isotonic_increasing(raw_theta)
    curve = ATMVarianceCurve(T, thetas)
    k_all = np.concatenate([k for k, _ in slices])
    w_all = np.concatenate([w for _, w in slices])
    th_all = np.concatenate([np.full(k.size, th) for (k, _), th in zip(slices, thetas)])
    scale = float(np.sum(w_all**2))

    def surf_w(x: np.ndarray) -> np.ndarray:
        rho = float(x[0])
        if phi == "power_law":
            ph = PowerLawPhi(float(x[1]), float(x[2]))(th_all)
        else:
            ph = HestonLikePhi(float(x[1]))(th_all)
        y = ph * k_all
        return 0.5 * th_all * (1 + rho * y + np.sqrt((y + rho) ** 2 + 1 - rho * rho))

    def obj(x: np.ndarray) -> float:
        return float(np.sum((surf_w(x) - w_all) ** 2)) / scale

    cons: list[dict[str, Any]] = []
    if phi == "power_law":
        x0 = np.array([-0.5, 0.8, 0.4])
        bounds = [(-0.999, 0.999), (1e-4, 20.0), (1e-3, 0.5 if no_arbitrage else 0.999)]
        if no_arbitrage:
            cons = [
                {"type": "ineq", "fun": lambda x: 2.0 - x[1] * (1 + x[0])},
                {"type": "ineq", "fun": lambda x: 2.0 - x[1] * (1 - x[0])},
            ]
    else:
        x0 = np.array([-0.5, 1.0])
        bounds = [(-0.999, 0.999), (1e-4, 1e3)]
        if no_arbitrage:
            cons = [
                {"type": "ineq", "fun": lambda x: x[1] - (1 + x[0]) / 4},
                {"type": "ineq", "fun": lambda x: x[1] - (1 - x[0]) / 4},
            ]
    best = None
    for start_rho in (-0.7, -0.3, 0.0, 0.3):
        x0 = x0.copy()
        x0[0] = start_rho
        r = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-15, "maxiter": 500},
        )
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    x = best.x
    phi_obj: Any = (
        PowerLawPhi(float(x[1]), float(x[2])) if phi == "power_law" else HestonLikePhi(float(x[1]))
    )
    surface = SSVI(float(x[0]), phi_obj, curve)
    rmse = float(np.sqrt(obj(x) * scale / w_all.size))
    return SSVIFit(surface, thetas, rmse, _slsqp_converged(best, cons, bounds))


@dataclass(frozen=True)
class SVISurfaceFit:
    """Result of :func:`calibrate_svi_surface`."""

    expiries: np.ndarray
    slices: list[SVIRaw]
    fits: list[SVIFit]

    @property
    def rmse(self) -> np.ndarray:
        return np.array([f.rmse for f in self.fits])

    @property
    def success(self) -> bool:
        return all(f.success for f in self.fits)

    def total_variance(self, k: Any, i: int) -> Any:
        return self.slices[i].total_variance(k)


def calibrate_svi_surface(
    expiries: Any,
    ks: Any,
    ws: Any,
    weights: Any = None,
    max_slope: float = 2.0,
    min_g: float = 1e-8,
    min_gap: float = 0.0,
    grid: Any = None,
) -> SVISurfaceFit:
    r"""Raw SVI slices free of butterfly and calendar arbitrage (Gatheral & Jacquier 2014, §5).

    Slices are fitted from the shortest expiry up. Each one starts from its
    unconstrained quasi-explicit fit and is solved by
    :func:`calibrate_svi_butterfly_free` with Durrleman's condition and,
    from the second slice on, :math:`w_i \ge w_{i-1} +` ``min_gap`` on a
    common grid (default 601 points covering all the data and twice its
    width either side) plus ordered wing slopes. Gatheral and Jacquier
    penalise crossings instead; constraints with cutting planes give an
    exactly calendar-free result on the grid.
    """
    T = np.asarray(expiries, float)
    if T.ndim != 1 or T.size == 0 or np.any(np.diff(T) <= 0) or T[0] <= 0:
        raise ValueError("expiries must be positive and strictly increasing")
    if len(ks) != T.size or len(ws) != T.size:
        raise ValueError("need one strike and one variance array per expiry")
    wts = [None] * T.size if weights is None else list(weights)
    if grid is None:
        lo = min(float(np.min(k)) for k in ks)
        hi = max(float(np.max(k)) for k in ks)
        span = hi - lo
        grid = np.linspace(lo - 2 * span, hi + 2 * span, 601)
    slices: list[SVIRaw] = []
    fits: list[SVIFit] = []
    prev: Optional[SVIRaw] = None
    for k, w, wt in zip(ks, ws, wts):
        fit = calibrate_svi_butterfly_free(
            k, w, wt, max_slope, grid=grid, min_g=min_g, lower=prev, min_gap=min_gap
        )
        slices.append(fit.params)
        fits.append(fit)
        prev = fit.params
    return SVISurfaceFit(T, slices, fits)
