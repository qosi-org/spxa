r"""Stochastic volatility inspired (SVI) slices.

The **raw** SVI total variance of Gatheral (2004) is

.. math:: w(k) = a + b\Bigl(\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\Bigr),

with :math:`b \ge 0`, :math:`|\rho| < 1`, :math:`\sigma > 0` and
:math:`a + b\sigma\sqrt{1-\rho^2} \ge 0` (non-negative variance). Its
minimum is attained at :math:`k^* = m - \rho\sigma/\sqrt{1-\rho^2}`, and
the wings are asymptotically linear with slopes :math:`b(1 \pm \rho)`.
Roger Lee's moment formula requires both slopes to be at most 2, i.e.
:math:`b(1 + |\rho|) \le 2`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SVIRaw"]


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
