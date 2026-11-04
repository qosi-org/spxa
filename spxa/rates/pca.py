r"""Principal components of yield-curve moves.

Given daily (or weekly) zero rates on fixed tenors, the covariance of
their changes :math:`\Sigma = \frac{1}{n-1}\sum_t (\Delta z_t - \bar{\Delta z})
(\Delta z_t - \bar{\Delta z})^\top` has eigenvectors (loadings) and
eigenvalues (factor variances). For yield curves the first three are, in
the familiar terminology of Litterman & Scheinkman (1991), level, slope and
curvature, and together explain almost all of the variance.

Eigenvectors are only defined up to sign. :func:`curve_pca` fixes the sign
so that the loadings read naturally:

* component 1 (level): positive sum of loadings;
* component 2 (slope): long end above the short end;
* component 3 onwards: positive loading at the middle tenor.

A position's key-rate deltas :math:`d_k` (on the same tenors) map to factor
deltas :math:`d^\top v_j`, its sensitivity per unit of factor :math:`j`, and
to a factor-model variance :math:`\sum_j \lambda_j (d^\top v_j)^2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["CurvePCA", "curve_pca"]


@dataclass(frozen=True)
class CurvePCA:
    """Result of :func:`curve_pca`.

    ``components[:, j]`` is the unit loading vector of factor ``j`` on
    ``tenors``; ``variances[j]`` its variance per observation interval.
    """

    tenors: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    total_variance: float
    mean: np.ndarray

    @property
    def explained_ratio(self) -> np.ndarray:
        return self.variances / self.total_variance

    @property
    def volatilities(self) -> np.ndarray:
        """Factor standard deviations per observation interval."""
        return np.sqrt(self.variances)

    def scores(self, changes: Any) -> np.ndarray:
        """Factor scores of (demeaned) curve changes, shape ``(..., n_components)``."""
        return (np.asarray(changes, float) - self.mean) @ self.components

    def reconstruct(self, scores: Any) -> np.ndarray:
        """Curve changes from factor scores (the mean added back)."""
        return np.asarray(scores, float) @ self.components.T + self.mean

    def scenario(self, component: int, n_sd: float = 1.0) -> np.ndarray:
        """Zero-rate shift of ``n_sd`` standard deviations of one factor."""
        return n_sd * self.volatilities[component] * self.components[:, component]

    def factor_deltas(self, key_deltas: Sequence[float]) -> np.ndarray:
        """Sensitivity per unit of each factor: :math:`d^\\top v_j`."""
        d = np.asarray(key_deltas, float)
        if d.shape != self.tenors.shape:
            raise ValueError("need one key-rate delta per tenor")
        return d @ self.components

    def factor_variance(self, key_deltas: Sequence[float]) -> float:
        """Variance of the position's P&L explained by the retained factors."""
        f = self.factor_deltas(key_deltas)
        return float(np.sum(self.variances * f * f))


def curve_pca(
    rates: Any,
    tenors: Sequence[float],
    n_components: int = 3,
    differences: bool = True,
) -> CurvePCA:
    """PCA of zero-rate changes.

    Parameters
    ----------
    rates
        Array ``(n_obs, n_tenors)`` of zero rates, oldest first. With
        ``differences=False`` the rows are taken as the changes already.
    tenors
        The tenors of the columns (strictly increasing).
    n_components
        Number of factors kept.
    """
    z = np.asarray(rates, float)
    t = np.asarray(tenors, float)
    if z.ndim != 2 or z.shape[1] != t.size:
        raise ValueError("rates must be (n_obs, n_tenors)")
    if np.any(np.diff(t) <= 0):
        raise ValueError("tenors must be strictly increasing")
    dz = np.diff(z, axis=0) if differences else z
    if dz.shape[0] < 2:
        raise ValueError("need at least two observations of changes")
    if not 1 <= n_components <= t.size:
        raise ValueError("n_components must lie in [1, n_tenors]")
    mean = dz.mean(axis=0)
    cov = np.cov(dz, rowvar=False, ddof=1).reshape(t.size, t.size)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = np.maximum(lam[order], 0.0), vec[:, order]
    mid = t.size // 2
    for j in range(t.size):
        v = vec[:, j]
        if j == 0:
            s = np.sum(v)
        elif j == 1:
            s = v[-1] - v[0]
        else:
            s = v[mid]
        if s < 0:
            vec[:, j] = -v
    k = n_components
    return CurvePCA(t, vec[:, :k].copy(), lam[:k].copy(), float(np.trace(cov)), mean)
