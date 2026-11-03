r"""Stochastic volatility inspired (SVI) slices.

The **raw** SVI total variance of Gatheral (2004) is

.. math:: w(k) = a + b\Bigl(\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\Bigr),

with :math:`b \ge 0`, :math:`|\rho| < 1`, :math:`\sigma > 0` and
:math:`a + b\sigma\sqrt{1-\rho^2} \ge 0` (non-negative variance). Its
minimum is attained at :math:`k^* = m - \rho\sigma/\sqrt{1-\rho^2}`, and
the wings are asymptotically linear with slopes :math:`b(1 \pm \rho)`.
Roger Lee's moment formula requires both slopes to be at most 2, i.e.
:math:`b(1 + |\rho|) \le 2`.

Two equivalent parametrisations (Gatheral & Jacquier 2014, Section 3):

**Natural** :math:`(\Delta, \mu, \rho, \omega, \zeta)`,

.. math:: w(k) = \Delta + \frac{\omega}{2}\Bigl(1 + \zeta\rho(k-\mu)
          + \sqrt{(\zeta(k-\mu)+\rho)^2 + 1 - \rho^2}\Bigr),

related to the raw one by :math:`a = \Delta + \tfrac{\omega}{2}(1-\rho^2)`,
:math:`b = \omega\zeta/2`, :math:`m = \mu - \rho/\zeta`,
:math:`\sigma = \sqrt{1-\rho^2}/\zeta`.

**Jump-wings** :math:`(v_t, \psi_t, p_t, c_t, \tilde v_t)` for expiry
:math:`t`: ATM variance :math:`v_t = w(0)/t`, ATM skew
:math:`\psi_t = w'(0)/(2\sqrt{w(0)})`, put and call wing slopes
:math:`p_t = b(1-\rho)/\sqrt{w(0)}`, :math:`c_t = b(1+\rho)/\sqrt{w(0)}`
and minimum variance :math:`\tilde v_t = \min_k w(k)/t`. These are
trader-interpretable and, unlike the raw parameters, scale sensibly
across expiries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SVIJumpWings", "SVINatural", "SVIRaw"]


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


@dataclass(frozen=True)
class SVIRaw:
    """Raw SVI parameters ``(a, b, rho, m, sigma)`` of one expiry (validated)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "rho", "m", "sigma"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.min_variance < -1e-14 * max(1.0, abs(self.a)):
            raise ValueError(
                f"negative minimum total variance a + b sigma sqrt(1 - rho^2) = {self.min_variance:.6g}"
            )

    # -- evaluation -------------------------------------------------------
    def _r(self, k: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(k, float) - self.m
        return x, np.sqrt(x * x + self.sigma**2)

    def total_variance(self, k: Any) -> Any:
        """:math:`w(k)`."""
        x, r = self._r(k)
        return _out(self.a + self.b * (self.rho * x + r))

    __call__ = total_variance

    def dw(self, k: Any) -> Any:
        r""":math:`w'(k) = b\,(\rho + (k-m)/\sqrt{(k-m)^2+\sigma^2})`."""
        x, r = self._r(k)
        return _out(self.b * (self.rho + x / r))

    def d2w(self, k: Any) -> Any:
        r""":math:`w''(k) = b\sigma^2/((k-m)^2+\sigma^2)^{3/2}`."""
        _, r = self._r(k)
        return _out(self.b * self.sigma**2 / r**3)

    def implied_vol(self, k: Any, T: float) -> Any:
        r"""Black implied volatility :math:`\sqrt{w(k)/T}`."""
        if T <= 0:
            raise ValueError("T must be positive")
        w = np.asarray(self.total_variance(k))
        return _out(np.sqrt(np.maximum(w, 0.0) / T))

    # -- shape --------------------------------------------------------------
    @property
    def min_variance(self) -> float:
        r""":math:`\min_k w(k) = a + b\sigma\sqrt{1-\rho^2}`."""
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho**2)

    @property
    def argmin(self) -> float:
        r""":math:`k^* = m - \rho\sigma/\sqrt{1-\rho^2}`."""
        return self.m - self.rho * self.sigma / math.sqrt(1.0 - self.rho**2)

    @property
    def left_slope(self) -> float:
        r""":math:`\lim_{k\to-\infty} -w(k)/k = b(1-\rho)`."""
        return self.b * (1.0 - self.rho)

    @property
    def right_slope(self) -> float:
        r""":math:`\lim_{k\to\infty} w(k)/k = b(1+\rho)`."""
        return self.b * (1.0 + self.rho)

    def satisfies_lee_bound(self) -> bool:
        """Both wing slopes at most 2 (Lee's moment formula)."""
        return self.b * (1.0 + abs(self.rho)) <= 2.0

    # -- conversion -----------------------------------------------------------
    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, p: Any) -> "SVIRaw":
        a, b, rho, m, sigma = (float(v) for v in np.asarray(p, float).ravel())
        return cls(a, b, rho, m, sigma)

    def to_natural(self) -> "SVINatural":
        q = math.sqrt(1.0 - self.rho**2)
        zeta = q / self.sigma
        omega = 2.0 * self.b / zeta
        return SVINatural(
            delta=self.a - 0.5 * omega * q * q,
            mu=self.m + self.rho / zeta,
            rho=self.rho,
            omega=omega,
            zeta=zeta,
        )

    def to_jump_wings(self, t: float) -> "SVIJumpWings":
        if t <= 0:
            raise ValueError("t must be positive")
        w0 = float(self.total_variance(0.0))
        if w0 <= 0:
            raise ValueError("jump-wings parameters need a positive ATM variance")
        sw = math.sqrt(w0)
        rr = math.hypot(self.m, self.sigma)
        return SVIJumpWings(
            v=w0 / t,
            psi=0.5 * self.b * (self.rho - self.m / rr) / sw,
            p=self.b * (1.0 - self.rho) / sw,
            c=self.b * (1.0 + self.rho) / sw,
            v_min=self.min_variance / t,
            t=t,
        )


@dataclass(frozen=True)
class SVINatural:
    """Natural SVI parameters ``(delta, mu, rho, omega, zeta)``."""

    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def __post_init__(self) -> None:
        if self.omega < 0:
            raise ValueError("omega must be non-negative")
        if self.zeta <= 0:
            raise ValueError("zeta must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    def total_variance(self, k: Any) -> Any:
        y = self.zeta * (np.asarray(k, float) - self.mu)
        root = np.sqrt((y + self.rho) ** 2 + 1.0 - self.rho**2)
        return _out(self.delta + 0.5 * self.omega * (1.0 + self.rho * y + root))

    __call__ = total_variance

    def to_raw(self) -> SVIRaw:
        q2 = 1.0 - self.rho**2
        return SVIRaw(
            a=self.delta + 0.5 * self.omega * q2,
            b=0.5 * self.omega * self.zeta,
            rho=self.rho,
            m=self.mu - self.rho / self.zeta,
            sigma=math.sqrt(q2) / self.zeta,
        )


@dataclass(frozen=True)
class SVIJumpWings:
    r"""SVI jump-wings parameters ``(v, psi, p, c, v_min)`` for expiry ``t``.

    Conversion to raw parameters (Gatheral & Jacquier 2014, Lemma 3.3) uses
    :math:`b = \sqrt{w_t}(c+p)/2`, :math:`\rho = 1 - p\sqrt{w_t}/b`,
    :math:`\beta = \rho - 2\psi\sqrt{w_t}/b = m/\sqrt{m^2+\sigma^2}`,
    which must lie in :math:`[-1, 1]`, then :math:`\alpha = \sigma/m` and
    :math:`m = (v - \tilde v)t / \bigl(b(-\rho + \operatorname{sign}(\alpha)\sqrt{1+\alpha^2}
    - \alpha\sqrt{1-\rho^2})\bigr)`; :math:`\beta = 0` is the case :math:`m = 0`.
    """

    v: float
    psi: float
    p: float
    c: float
    v_min: float
    t: float

    def __post_init__(self) -> None:
        if self.t <= 0 or self.v <= 0:
            raise ValueError("t and v must be positive")
        if self.p < 0 or self.c < 0 or self.p + self.c == 0:
            raise ValueError("wing slopes must be non-negative and not both zero")

    def to_raw(self) -> SVIRaw:
        t = self.t
        w = self.v * t
        sw = math.sqrt(w)
        b = 0.5 * sw * (self.c + self.p)
        rho = 1.0 - self.p * sw / b
        beta = rho - 2.0 * self.psi * sw / b
        if not -1.0 < beta < 1.0:
            raise ValueError(
                f"inconsistent jump-wings parameters (beta = {beta:.6g} outside (-1, 1))"
            )
        q = math.sqrt(1.0 - rho * rho)
        dv = (self.v - self.v_min) * t
        if beta == 0.0:
            m = 0.0
            if q == 1.0:
                raise ValueError("m = 0 and rho = 0 leave sigma undetermined")
            sigma = dv / (b * (1.0 - q))
        else:
            alpha = math.copysign(math.sqrt(1.0 / (beta * beta) - 1.0), beta)
            denom = b * (-rho + math.copysign(math.sqrt(1.0 + alpha * alpha), alpha) - alpha * q)
            m = dv / denom
            sigma = alpha * m
        if sigma <= 0:
            raise ValueError("inconsistent jump-wings parameters (sigma <= 0)")
        a = self.v_min * t - b * sigma * q
        return SVIRaw(a, b, rho, m, sigma)

    def butterfly_free_projection(self) -> "SVIJumpWings":
        r"""Gatheral & Jacquier (2014, §5.1): keep :math:`(v, \psi, p)` and set

        .. math:: c' = p + 2\psi, \qquad \tilde v' = v\,\frac{4pc'}{(p + c')^2}.

        The result is an SSVI slice with :math:`\theta = vt`,
        :math:`\rho = 2\psi/(p + c')` and :math:`\varphi = (p + c')/\sqrt{\theta}`,
        free of butterfly arbitrage when :meth:`satisfies_ssvi_bounds` holds.
        """
        c2 = self.p + 2.0 * self.psi
        if c2 <= 0.0 or self.p <= 0.0:
            raise ValueError("projection needs p > 0 and p + 2 psi > 0")
        vmin = self.v * 4.0 * self.p * c2 / (self.p + c2) ** 2
        return SVIJumpWings(self.v, self.psi, self.p, c2, vmin, self.t)

    def satisfies_ssvi_bounds(self) -> bool:
        r"""Remark 4.2: :math:`\sqrt{vt}\max(p, c) \le 4` and :math:`(p + c)\max(p, c) \le 8`.

        For an SSVI-shaped slice (:math:`c - p = 2\psi`) these are the
        sufficient butterfly conditions of Theorem 4.2.
        """
        mx = max(self.p, self.c)
        return math.sqrt(self.v * self.t) * mx <= 4.0 and (self.p + self.c) * mx <= 8.0

    def to_ssvi_natural(self) -> "SVINatural":
        """The SSVI slice of an SSVI-shaped parameter set (``c - p = 2 psi``)."""
        if abs(self.c - self.p - 2.0 * self.psi) > 1e-12 * max(1.0, self.c + self.p):
            raise ValueError("not an SSVI slice: c - p must equal 2 psi")
        theta = self.v * self.t
        rho = 2.0 * self.psi / (self.p + self.c)
        return SVINatural(0.0, 0.0, rho, theta, (self.p + self.c) / math.sqrt(theta))

    def total_variance(self, k: Any) -> Any:
        return self.to_raw().total_variance(k)
