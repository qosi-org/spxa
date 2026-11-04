r"""Hull–White trinomial tree fitted to the discount curve.

The construction of Hull & White (*Numerical procedures for implementing
term structure models I: single-factor models*, J. Derivatives, 1994):

1. Build a symmetric trinomial tree for :math:`dR^* = -aR^*dt + \sigma dW`
   with spacing :math:`\Delta x = \sigma\sqrt{3\Delta t}`, truncated at
   :math:`|j| \le j_{\max} = \lceil 0.184/(a\Delta t)\rceil`. Inside, the
   branching goes to :math:`j+1, j, j-1`, and at the edges it bends
   inwards. With :math:`M = e^{-a\Delta t} - 1`, the probabilities match
   the conditional mean :math:`jM\Delta x` and the variance
   :math:`\sigma^2\Delta t` of each step.
2. Shift the nodes at step :math:`m` by :math:`\alpha_m`, chosen by forward
   induction on the Arrow–Debreu prices :math:`Q_{m,j}` so that the tree
   prices every zero-coupon bond on its time grid exactly:

   .. math:: \alpha_m = \frac{1}{\Delta t}\log\frac{\sum_j Q_{m,j}\,e^{-j\Delta x\,\Delta t}}{P(0, (m+1)\Delta t)}.

The node rate :math:`r_{m,j} = \alpha_m + j\Delta x` applies over
:math:`[m\Delta t, (m+1)\Delta t]`. The tree uses the model's volatility
at the first step, so it needs a constant :math:`\sigma`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.hull_white import HullWhite

__all__ = ["HullWhiteTree", "TrinomialShortRateTree"]


class TrinomialShortRateTree:
    """Trinomial geometry of Hull & White (1994) with curve fitting by forward induction.

    Node ``j`` at step ``m`` has state :math:`\\alpha_m + j\\Delta x`, and
    :meth:`node_rates` maps it to the short rate over the step.
    Subclasses define the map and how :math:`\\alpha_m` is solved.
    """

    def __init__(
        self, curve: YieldCurve, a: float, sigma: float, horizon: float, n_steps: int
    ) -> None:
        if horizon <= 0 or n_steps < 1:
            raise ValueError("need horizon > 0 and n_steps >= 1")
        if a <= 0 or sigma <= 0:
            raise ValueError("need a > 0 and sigma > 0")
        self.curve = curve
        self.dt = dt = horizon / n_steps
        self.n_steps = n_steps
        self.dx = dx = sigma * math.sqrt(3 * dt)
        self.jmax = jmax = max(1, math.ceil(0.184 / (a * dt)))
        M = math.expm1(-a * dt)
        j = np.arange(-jmax, jmax + 1)
        jm, j2m2 = j * M, (j * M) ** 2
        pu = 1 / 6 + (j2m2 + jm) / 2
        pm = 2 / 3 - j2m2
        pd = 1 / 6 + (j2m2 - jm) / 2
        shift = np.zeros(j.size, int)  # centre of the branching: j + shift
        # top: branch to j, j-1, j-2
        pu[-1] = 7 / 6 + (j2m2[-1] + 3 * jm[-1]) / 2
        pm[-1] = -1 / 3 - j2m2[-1] - 2 * jm[-1]
        pd[-1] = 1 / 6 + (j2m2[-1] + jm[-1]) / 2
        shift[-1] = -1
        # bottom: branch to j+2, j+1, j
        pu[0] = 1 / 6 + (j2m2[0] - jm[0]) / 2
        pm[0] = -1 / 3 - j2m2[0] + 2 * jm[0]
        pd[0] = 7 / 6 + (j2m2[0] - 3 * jm[0]) / 2
        shift[0] = 1
        self._p = np.stack([pu, pm, pd])
        self._shift = shift
        self.alpha = np.zeros(n_steps)
        Q = np.zeros(j.size)
        Q[jmax] = 1.0
        self._Q = [Q.copy()]
        for m in range(n_steps):
            w = self.width(m)
            idx = np.arange(jmax - w, jmax + w + 1)
            target = float(curve.discount((m + 1) * dt))
            self.alpha[m] = self._solve_alpha(Q[idx], j[idx] * dx, target)
            disc = Q[idx] * np.exp(-self.node_rates(self.alpha[m], j[idx] * dx) * dt)
            Qn = np.zeros(j.size)
            for k, s in zip((1, 0, -1), range(3)):
                np.add.at(Qn, idx + self._shift[idx] + k, disc * self._p[s, idx])
            Q = Qn
            self._Q.append(Q.copy())

    def node_rates(self, alpha: float, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _solve_alpha(self, Q: np.ndarray, x: np.ndarray, target: float) -> float:
        raise NotImplementedError

    def width(self, m: int) -> int:
        """Half-width of the tree at step ``m``."""
        return min(m, self.jmax)

    def times(self) -> np.ndarray:
        return np.arange(self.n_steps + 1) * self.dt

    def step(self, t: float) -> int:
        m = round(t / self.dt)
        if abs(m * self.dt - t) > 1e-9 * max(1.0, t) or not 0 <= m <= self.n_steps:
            raise ValueError(f"time {t} is not on the tree grid")
        return m

    def rates(self, m: int) -> np.ndarray:
        """Short rates :math:`r_{m,j}` at the live nodes of step ``m`` (low to high)."""
        w = self.width(m)
        return self.node_rates(self.alpha[m], np.arange(-w, w + 1) * self.dx)

    def arrow_debreu(self, m: int) -> np.ndarray:
        w = self.width(m)
        return self._Q[m][self.jmax - w : self.jmax + w + 1]

    def rollback(self, values: np.ndarray, m: int) -> np.ndarray:
        """Discounted expectation of step-``m`` node ``values`` at step ``m - 1``."""
        w_prev, w = self.width(m - 1), self.width(m)
        jmax = self.jmax
        idx = np.arange(jmax - w_prev, jmax + w_prev + 1)
        full = np.zeros(2 * jmax + 1)
        full[jmax - w : jmax + w + 1] = values
        cont = sum(
            self._p[s, idx] * full[idx + self._shift[idx] + k] for k, s in zip((1, 0, -1), range(3))
        )
        return np.exp(-self.rates(m - 1) * self.dt) * cont

    def discount_bonds(self, m: int, maturity: float) -> np.ndarray:
        """Tree prices of the bond maturing at ``maturity`` at the nodes of step ``m``."""
        n = self.step(maturity)
        v = np.ones(2 * self.width(n) + 1)
        for k in range(n, m, -1):
            v = self.rollback(v, k)
        return v

    def price(
        self,
        payoff: Callable[[int, np.ndarray], np.ndarray],
        maturity: float,
        exercise: Optional[Sequence[float]] = None,
        exercise_value: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> float:
        """Roll back ``payoff(m, node_rates)`` from ``maturity``.

        If ``exercise`` times are given, the holder may take
        ``exercise_value(m, node_rates)`` (default: ``payoff``) instead of
        continuing at those steps (Bermudan or American style).
        """
        n = self.step(maturity)
        v = np.asarray(payoff(n, self.rates(n)), float)
        ex_steps = {self.step(t) for t in (exercise or [])}
        ev = exercise_value or payoff
        for k in range(n, 0, -1):
            v = self.rollback(v, k)
            if k - 1 in ex_steps:
                v = np.maximum(v, ev(k - 1, self.rates(k - 1)))
        return float(v[0])

    def zcb_option(
        self, T: float, S: float, K: float, option: str = "call", american: bool = False
    ) -> float:
        """Option on the bond maturing at ``S``, expiring at ``T`` (both on the grid)."""
        eta = 1.0 if option == "call" else -1.0
        mT = self.step(T)
        bonds = {m: self.discount_bonds(m, S) for m in (range(mT + 1) if american else [mT])}

        def pay(m: int, r: np.ndarray) -> np.ndarray:
            return np.maximum(eta * (bonds[m] - K), 0.0)

        ex = list(self.times()[: mT + 1]) if american else None
        return self.price(pay, T, exercise=ex)


class HullWhiteTree(TrinomialShortRateTree):
    """Trinomial tree for ``model`` on ``n_steps`` equal steps up to ``horizon``."""

    def __init__(self, model: HullWhite, horizon: float, n_steps: int) -> None:
        if model.sigma.size != 1:
            raise ValueError("the trinomial tree needs a constant volatility")
        self.model = model
        super().__init__(model.curve, model.a, float(model.sigma[0]), horizon, n_steps)

    def node_rates(self, alpha: float, x: np.ndarray) -> np.ndarray:
        return alpha + x

    def _solve_alpha(self, Q: np.ndarray, x: np.ndarray, target: float) -> float:
        # closed form: sum Q e^{-(alpha + x) dt} = P(0, t_{m+1})
        return math.log(float(np.sum(Q * np.exp(-x * self.dt))) / target) / self.dt
