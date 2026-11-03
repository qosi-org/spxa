r"""Extended SSVI (eSSVI) of Hendriks & Martini (2019).

Each expiry carries its own SSVI slice :math:`(\theta, \psi, \rho)` with
:math:`\psi = \theta\varphi`:

.. math:: w(k) = \frac{\theta}{2}\Bigl(1 + \frac{\rho\psi}{\theta}k
          + \sqrt{\Bigl(\frac{\psi}{\theta}k + \rho\Bigr)^2 + 1 - \rho^2}\Bigr).

No butterfly arbitrage (Gatheral–Jacquier, Theorem 4.2) holds when

.. math:: \psi(1 + |\rho|) < 4, \qquad \psi^2(1 + |\rho|) \le 4\theta,

and two slices :math:`T_1 < T_2` are free of calendar spread arbitrage when
(Hendriks & Martini; Corbetta, Cohort, Laachir & Martini 2019, §2.2)

.. math:: \theta_2 \ge \theta_1, \qquad \psi_2 \ge \psi_1, \qquad
          |\rho_2\psi_2 - \rho_1\psi_1| \le \psi_2 - \psi_1.

:class:`ESSVI` interpolates :math:`\theta`, :math:`\psi` and
:math:`\rho\psi` linearly in time between slices, scales
:math:`(\theta, \psi)` by :math:`t/T_1` before the first expiry and keeps
:math:`(\psi, \rho)` while extending :math:`\theta` beyond the last one;
Corbetta et al. (§5) prove this preserves both conditions, so the
continuous surface is free of static arbitrage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from spxa.vol.svi import SVINatural, SVIRaw

__all__ = ["ESSVI", "ESSVISlice", "anchored_theta", "calibrate_essvi", "essvi_calendar_free"]


@dataclass(frozen=True)
class ESSVISlice:
    """One eSSVI slice: ATM total variance ``theta``, ``psi = theta * phi`` and ``rho``."""

    theta: float
    psi: float
    rho: float

    def __post_init__(self) -> None:
        if self.theta <= 0 or self.psi <= 0:
            raise ValueError("theta and psi must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def phi(self) -> float:
        return self.psi / self.theta

    def total_variance(self, k: Any) -> Any:
        y = self.phi * np.asarray(k, float)
        r = self.rho
        w = 0.5 * self.theta * (1.0 + r * y + np.sqrt((y + r) ** 2 + 1.0 - r * r))
        return float(w) if np.ndim(w) == 0 else w

    __call__ = total_variance

    def to_raw(self) -> SVIRaw:
        return SVINatural(0.0, 0.0, self.rho, self.theta, self.phi).to_raw()

    def butterfly_free(self) -> bool:
        """Sufficient conditions of Gatheral–Jacquier Theorem 4.2."""
        a = 1.0 + abs(self.rho)
        return self.psi * a < 4.0 and self.psi * self.psi * a <= 4.0 * self.theta


def essvi_calendar_free(first: ESSVISlice, second: ESSVISlice, tol: float = 0.0) -> bool:
    r"""Calendar condition between a slice and a later one.

    :math:`\theta_2 \ge \theta_1`, :math:`\psi_2 \ge \psi_1` and
    :math:`|\rho_2\psi_2 - \rho_1\psi_1| \le \psi_2 - \psi_1` (up to ``tol``).
    """
    dpsi = second.psi - first.psi
    return (
        second.theta >= first.theta - tol
        and dpsi >= -tol
        and abs(second.rho * second.psi - first.rho * first.psi) <= dpsi + tol
    )


class ESSVI:
    """Continuous eSSVI surface from slices at increasing expiries.

    Parameters
    ----------
    times : sequence of float
        Increasing positive expiries.
    slices : sequence of ESSVISlice
        One slice per expiry.
    long_theta_slope : float, optional
        Growth rate of :math:`\\theta` beyond the last expiry (default: the
        slope of the last bucket, or :math:`\\theta_N/T_N` with one slice).
    """

    def __init__(
        self,
        times: Sequence[float],
        slices: Sequence[ESSVISlice],
        long_theta_slope: Optional[float] = None,
    ) -> None:
        t = np.asarray(times, float)
        if t.ndim != 1 or t.size == 0 or t.size != len(slices):
            raise ValueError("need one slice per expiry")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("expiries must be positive and increasing")
        self.times = t
        self.slices = list(slices)
        self._theta = np.array([s.theta for s in slices])
        self._psi = np.array([s.psi for s in slices])
        self._rhopsi = np.array([s.rho * s.psi for s in slices])
        if long_theta_slope is None:
            if t.size > 1:
                long_theta_slope = (self._theta[-1] - self._theta[-2]) / (t[-1] - t[-2])
            else:
                long_theta_slope = self._theta[0] / t[0]
        if long_theta_slope < 0:
            raise ValueError("long_theta_slope must be non-negative")
        self.long_theta_slope = float(long_theta_slope)

    def params(self, T: float) -> ESSVISlice:
        """Interpolated slice at expiry ``T`` (Corbetta et al. §5)."""
        T = float(T)
        if T <= 0:
            raise ValueError("T must be positive")
        t = self.times
        if T <= t[0]:
            lam = T / t[0]
            s0 = self.slices[0]
            return ESSVISlice(lam * s0.theta, lam * s0.psi, s0.rho)
        if T >= t[-1]:
            sN = self.slices[-1]
            return ESSVISlice(sN.theta + self.long_theta_slope * (T - t[-1]), sN.psi, sN.rho)
        i = int(np.searchsorted(t, T)) - 1
        lam = (T - t[i]) / (t[i + 1] - t[i])
        theta = (1 - lam) * self._theta[i] + lam * self._theta[i + 1]
        psi = (1 - lam) * self._psi[i] + lam * self._psi[i + 1]
        rhopsi = (1 - lam) * self._rhopsi[i] + lam * self._rhopsi[i + 1]
        return ESSVISlice(theta, psi, rhopsi / psi)

    def slice(self, T: float) -> SVIRaw:
        return self.params(T).to_raw()

    def total_variance(self, k: Any, T: float) -> Any:
        return self.params(T).total_variance(k)

    def implied_vol(self, k: Any, T: float) -> Any:
        w = np.asarray(self.total_variance(k, T), float)
        v = np.sqrt(w / float(T))
        return float(v) if v.ndim == 0 else v

    def is_arbitrage_free(self, tol: float = 0.0) -> bool:
        """Butterfly conditions on every node and calendar conditions between neighbours."""
        if not all(s.butterfly_free() for s in self.slices):
            return False
        return all(essvi_calendar_free(a, b, tol) for a, b in zip(self.slices, self.slices[1:]))

    def violations(self) -> list[str]:
        """Human-readable list of failed node conditions."""
        out = []
        for i, s in enumerate(self.slices):
            if not s.butterfly_free():
                out.append(f"butterfly at T={self.times[i]:g}")
        for i, (a, b) in enumerate(zip(self.slices, self.slices[1:])):
            if not essvi_calendar_free(a, b):
                out.append(f"calendar between T={self.times[i]:g} and T={self.times[i + 1]:g}")
        return out


def anchored_theta(rho: float, psi: Any, k_star: float, theta_star: float) -> Any:
    r"""ATM variance of the slice through :math:`(k^*, \theta^*)`.

    Solving :math:`w(k^*; \theta, \rho, \psi) = \theta^*` exactly gives

    .. math:: \theta = \frac{(2\theta^* - \rho\psi k^*)^2 - \psi^2 k^{*2}}{4\theta^*}
              = \theta^* - \rho\psi k^* - \frac{(1 - \rho^2)k^{*2}}{4\theta^*}\psi^2,

    whose first-order part is the anchor of Corbetta et al.
    """
    return (
        theta_star
        - rho * psi * k_star
        - (1.0 - rho * rho) * k_star * k_star / (4.0 * theta_star) * psi * psi
    )


def _quad_below(a: float, b: float, c: float) -> tuple[float, float]:
    """Interval where ``a x^2 + b x + c <= 0`` (``a >= 0``); empty as ``(1, 0)``."""
    if a == 0.0:
        if b == 0.0:
            return (-np.inf, np.inf) if c <= 0 else (1.0, 0.0)
        return (-np.inf, -c / b) if b > 0 else (-c / b, np.inf)
    disc = b * b - 4 * a * c
    if disc < 0:
        return (1.0, 0.0)
    r = np.sqrt(disc)
    # numerically stable roots
    q = -0.5 * (b + np.copysign(r, b))
    x1, x2 = sorted((q / a, c / q if q != 0 else 0.0))
    return x1, x2


def _psi_bounds(
    rho: float, k_star: float, theta_star: float, prev: Optional[ESSVISlice]
) -> tuple[float, float]:
    """Admissible ``psi`` for the anchored slice (exact anchor, Corbetta et al. §2)."""
    a = 1.0 + abs(rho)
    c = (1.0 - rho * rho) * k_star * k_star / (4.0 * theta_star)
    lo, hi = 0.0, 4.0 / a
    # psi^2 (1 + |rho|) <= 4 theta(psi)  (this also keeps theta > 0)
    l2, h2 = _quad_below(a + 4.0 * c, 4.0 * rho * k_star, -4.0 * theta_star)
    lo, hi = max(lo, l2), min(hi, h2)
    if prev is not None:
        pp, rp = prev.psi, prev.rho
        lo = max(lo, (pp - rp * pp) / (1.0 - rho), (pp + rp * pp) / (1.0 + rho))
        # theta(psi) >= theta_prev
        l3, h3 = _quad_below(c, rho * k_star, prev.theta - theta_star)
        lo, hi = max(lo, l3), min(hi, h3)
    return lo, hi


def calibrate_essvi(
    expiries: Sequence[float],
    ks: Sequence[Any],
    ws: Sequence[Any],
    n_rho: int = 41,
    margin: float = 1e-9,
) -> ESSVI:
    r"""Slice-by-slice arbitrage-free eSSVI calibration (Corbetta et al. 2019).

    Each slice is anchored at the quote closest to the money,
    :math:`(k^*, \theta^*)`, and parametrised by :math:`(\rho, \psi)`, with
    :math:`\theta` from :func:`anchored_theta` (the exact version of the
    paper's first-order :math:`\theta = \theta^* - \rho\psi k^*`, so the
    slice passes through the anchor). For a given :math:`\rho` the
    butterfly conditions and the calendar conditions against the previous
    slice reduce to an interval for :math:`\psi`, over which the
    least-squares error in total variance is minimised by a bounded scalar
    search; :math:`\rho` is scanned on a grid of ``n_rho`` points and the
    best value refined by a bounded search on the profile error. Every
    slice therefore satisfies the eSSVI no-arbitrage conditions, and so
    does the interpolated surface.
    """
    from scipy.optimize import minimize_scalar

    T = np.asarray(expiries, float)
    if T.ndim != 1 or len(ks) != T.size or len(ws) != T.size:
        raise ValueError("need one (k, w) slice per expiry")
    slices: list[ESSVISlice] = []
    prev: Optional[ESSVISlice] = None
    for k_raw, w_raw in zip(ks, ws):
        k = np.asarray(k_raw, float)
        w = np.asarray(w_raw, float)
        if k.shape != w.shape or k.size < 3:
            raise ValueError("each slice needs at least 3 quotes")
        j = int(np.argmin(np.abs(k)))
        k_star, theta_star = float(k[j]), float(w[j])

        def sse(rho: float, psi: float) -> float:
            theta = anchored_theta(rho, psi, k_star, theta_star)
            y = psi / theta * k
            model = 0.5 * theta * (1 + rho * y + np.sqrt((y + rho) ** 2 + 1 - rho * rho))
            return float(np.sum((model - w) ** 2))

        def profile(rho: float) -> tuple[float, float]:
            lo, hi = _psi_bounds(rho, k_star, theta_star, prev)
            lo, hi = lo + margin, hi - margin
            if hi <= lo:
                return np.inf, np.nan
            r = minimize_scalar(
                lambda p: sse(rho, p), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}
            )
            return float(r.fun), float(r.x)

        grid = np.linspace(-0.99, 0.99, n_rho)
        vals = [profile(float(r))[0] for r in grid]
        i = int(np.argmin(vals))
        if not np.isfinite(vals[i]):
            raise ValueError("no arbitrage-free slice is compatible with the previous expiry")
        step = grid[1] - grid[0]
        r = minimize_scalar(
            lambda x: profile(float(x))[0],
            bounds=(max(grid[i] - step, -0.999), min(grid[i] + step, 0.999)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        rho = float(r.x) if r.fun <= vals[i] else float(grid[i])
        psi = profile(rho)[1]
        s = ESSVISlice(anchored_theta(rho, psi, k_star, theta_star), psi, rho)
        slices.append(s)
        prev = s
    return ESSVI(T, slices)
