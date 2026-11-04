r"""Model risk: how far prices move across models that fit the same market.

Following Cont (2006), take a set of pricing models :math:`\mathcal Q`, each
consistent with the liquid quotes up to a calibration error
:math:`\epsilon_Q`. The upper and lower prices of a claim :math:`X` are

.. math::

    \bar\pi(X) = \sup_{Q:\ \epsilon_Q \le \delta} E_Q[X], \qquad
    \underline\pi(X) = \inf_{Q:\ \epsilon_Q \le \delta} E_Q[X],

and the model risk is the width :math:`\mu(X) = \bar\pi(X) - \underline\pi(X)`.
It is a coherent measure in :math:`X`: subadditive, positively homogeneous,
zero on cash and unchanged by adding positions that every model prices the
same (the calibration instruments). Admitting models in proportion to their
fit instead of with a hard tolerance gives the penalised (convex) version

.. math::

    \bar\pi_\lambda(X) = \max_Q\bigl(E_Q[X] - \lambda\epsilon_Q\bigr), \qquad
    \underline\pi_\lambda(X) = \min_Q\bigl(E_Q[X] + \lambda\epsilon_Q\bigr),

whose width falls from the full range at :math:`\lambda = 0` to the spread among
the best-fitting models as :math:`\lambda \to \infty`. Model-free bounds (MOT,
:func:`spxa.vol.robust_bounds`) contain every martingale model with the same
marginals, so :meth:`ModelRisk.share` measures how much of the model-free
interval a model family explores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

__all__ = ["ModelRisk", "smile_rmse"]


@dataclass(frozen=True)
class ModelRisk:
    """Prices of one claim (or a vector of claims) under named models.

    ``prices`` has shape ``(n_models,)`` or ``(n_models, n_claims)``; ``errors``
    are the calibration errors of the models (any consistent unit, e.g.
    implied-volatility RMSE); ``std_errors`` are Monte Carlo standard errors
    of the prices (zero for exact pricers).
    """

    names: tuple[str, ...]
    prices: np.ndarray
    errors: np.ndarray
    std_errors: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0 or len(set(self.names)) != n:
            raise ValueError("need at least one model and distinct names")
        for a in (self.prices, self.std_errors):
            if a.shape[0] != n:
                raise ValueError("one row of prices per model")
        if self.errors.shape != (n,) or np.any(self.errors < 0):
            raise ValueError("one non-negative calibration error per model")

    @classmethod
    def from_prices(
        cls,
        prices: Mapping[str, Any],
        errors: Optional[Mapping[str, float]] = None,
        std_errors: Optional[Mapping[str, Any]] = None,
    ) -> "ModelRisk":
        """Build from dicts keyed by model name (errors and standard errors default to zero)."""
        names = tuple(prices)
        p = np.array([np.asarray(prices[n], float) for n in names])
        e = np.array([float((errors or {}).get(n, 0.0)) for n in names])
        s = np.array(
            [np.asarray((std_errors or {}).get(n, np.zeros(p.shape[1:])), float) for n in names]
        )
        return cls(names, p, e, np.broadcast_to(s, p.shape).copy())

    @classmethod
    def from_samples(
        cls,
        samples: Mapping[str, Any],
        payoff: Callable[..., Any],
        errors: Optional[Mapping[str, float]] = None,
        discount: float = 1.0,
    ) -> "ModelRisk":
        """Monte Carlo prices ``discount * mean(payoff(*samples[name]))`` with their standard errors."""
        prices, ses = {}, {}
        for name, args in samples.items():
            pay = discount * np.asarray(payoff(*args), float)
            prices[name] = pay.mean(axis=0)
            ses[name] = pay.std(axis=0, ddof=1) / math.sqrt(pay.shape[0])
        return cls.from_prices(prices, errors, ses)

    def _mask(self, tolerance: float) -> np.ndarray:
        mask = self.errors <= tolerance
        if not mask.any():
            raise ValueError(f"no model fits within the tolerance {tolerance}")
        return mask

    def admissible(self, tolerance: float = math.inf) -> tuple[str, ...]:
        """Names of the models whose calibration error is at most ``tolerance``."""
        return tuple(n for n, ok in zip(self.names, self._mask(tolerance)) if ok)

    def upper(self, tolerance: float = math.inf) -> Any:
        """Highest price among admissible models."""
        return self.prices[self._mask(tolerance)].max(axis=0)

    def lower(self, tolerance: float = math.inf) -> Any:
        """Lowest price among admissible models."""
        return self.prices[self._mask(tolerance)].min(axis=0)

    def width(self, tolerance: float = math.inf) -> Any:
        """Cont's coherent model-risk measure: upper minus lower price."""
        return self.upper(tolerance) - self.lower(tolerance)

    def penalised(self, lam: float) -> tuple[Any, Any]:
        """Penalised lower and upper prices, each model charged ``lam`` per unit of calibration error."""
        if lam < 0:
            raise ValueError("the penalty must be non-negative")
        pen = (lam * self.errors).reshape((-1,) + (1,) * (self.prices.ndim - 1))
        return (self.prices + pen).min(axis=0), (self.prices - pen).max(axis=0)

    def penalised_width(self, lam: float) -> Any:
        """Width of the penalised interval (may be negative when the penalty dominates)."""
        lo, hi = self.penalised(lam)
        return hi - lo

    def relative_width(self, tolerance: float = math.inf) -> Any:
        """Width over the mid of the admissible range."""
        return self.width(tolerance) / (0.5 * (self.upper(tolerance) + self.lower(tolerance)))

    def significant(self, n_se: float = 3.0, tolerance: float = math.inf) -> Any:
        """Whether the width exceeds the combined Monte Carlo error of the extreme models."""
        m = self._mask(tolerance)
        p, s = self.prices[m], self.std_errors[m]
        hi, lo = p.argmax(axis=0), p.argmin(axis=0)
        cols = np.arange(p.shape[1]) if p.ndim > 1 else None
        s_hi = s[hi, cols] if cols is not None else s[hi]
        s_lo = s[lo, cols] if cols is not None else s[lo]
        return self.width(tolerance) > n_se * np.hypot(s_hi, s_lo)

    def within(self, lower: Any, upper: Any, n_se: float = 3.0) -> bool:
        """Whether every model price lies in ``[lower, upper]`` up to ``n_se`` standard errors."""
        slack = n_se * self.std_errors
        return bool(np.all(self.prices + slack >= lower) and np.all(self.prices - slack <= upper))

    def share(self, lower: Any, upper: Any, tolerance: float = math.inf) -> Any:
        """Fraction of a model-free interval ``[lower, upper]`` spanned by the admissible models."""
        return self.width(tolerance) / (np.asarray(upper, float) - np.asarray(lower, float))

    def combine(self, weights: Any) -> "ModelRisk":
        """The portfolio ``sum_j weights[j] * claim_j``, priced model by model. Standard errors
        are combined as if the claims were priced on independent paths."""
        w = np.asarray(weights, float)
        if self.prices.ndim != 2 or w.shape != (self.prices.shape[1],):
            raise ValueError("need one weight per claim")
        se = np.sqrt((self.std_errors**2) @ (w * w))
        return ModelRisk(self.names, self.prices @ w, self.errors, se)


def smile_rmse(model_vols: Any, market_vols: Any, weights: Any = None) -> float:
    """Weighted root-mean-square implied-volatility error, the usual calibration error."""
    m, q = np.asarray(model_vols, float), np.asarray(market_vols, float)
    if m.shape != q.shape:
        raise ValueError("model and market vols must have the same shape")
    w = np.ones_like(m) if weights is None else np.broadcast_to(np.asarray(weights, float), m.shape)
    return float(np.sqrt((w * (m - q) ** 2).sum() / w.sum()))
