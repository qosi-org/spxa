r"""
Least-squares Monte Carlo for Bermudan exercise (Longstaff & Schwartz, 2001).

A Markov state :math:`X` is simulated to the exercise dates
:math:`t_1 < \dots < t_m`. Going backwards, the continuation value at
:math:`t_k` is estimated by regressing the realised discounted value of the
current policy on basis functions of :math:`X_{t_k}` over the in-the-money
paths, and the policy exercises where the exercise value exceeds that
estimate. The regression coefficients are fitted on one set of paths and
the policy is then valued on an independent set, so the result is a
**lower bound**: an unbiased estimate of the value of a feasible policy.

:class:`ExerciseProblem` describes a problem through a transition function,
so the same code serves every model with an exact or discretised step;
:func:`hull_white_exercise_problem` builds one for a Bermudan swaption in
Hull–White. The matching upper bound is
:func:`spxa.rates.lsm.andersen_broadie`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from spxa.rates.bermudan import BermudanSwaption
from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_mc import hull_white_transition

__all__ = [
    "ExerciseProblem",
    "LSMPolicy",
    "LSMResult",
    "hull_white_exercise_problem",
    "lsm",
]

Step = Callable[[float, float, np.ndarray, np.random.Generator], "tuple[np.ndarray, np.ndarray]"]


@dataclass(frozen=True)
class ExerciseProblem:
    """A Bermudan problem on a simulated Markov state.

    ``step(t0, t1, X, rng)`` maps states ``X`` (shape ``(n, d)``) at ``t0``
    to states at ``t1`` and the discount factors :math:`D(t_0, t_1)`;
    ``exercise_value(k, X)`` and ``basis(k, X)`` act on the states at
    ``times[k]``.
    """

    times: np.ndarray
    x0: np.ndarray
    step: Step
    exercise_value: Callable[[int, np.ndarray], np.ndarray]
    basis: Callable[[int, np.ndarray], np.ndarray]

    @property
    def m(self) -> int:
        return int(np.asarray(self.times).size)

    def simulate(
        self, n_paths: int, rng: np.random.Generator, start: int = -1, X: "np.ndarray | None" = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """States at ``times[start + 1:]`` and discount factors from the start time to each.

        ``start = -1`` starts at time zero from ``x0``; otherwise from the
        states ``X`` at ``times[start]``.
        """
        t = np.asarray(self.times, float)
        if start < 0:
            cur = np.tile(np.asarray(self.x0, float), (n_paths, 1))
            t_prev = 0.0
        else:
            assert X is not None
            cur, t_prev = np.asarray(X, float), float(t[start])
        D = np.ones(cur.shape[0])
        states, discounts = [], []
        for k in range(start + 1, self.m):
            cur, d = self.step(t_prev, float(t[k]), cur, rng)
            D = D * d
            states.append(cur)
            discounts.append(D)
            t_prev = float(t[k])
        return states, np.array(discounts)


@dataclass(frozen=True)
class LSMPolicy:
    """Exercise rule from regression coefficients (one set per date except the last)."""

    problem: ExerciseProblem
    coefficients: tuple[np.ndarray, ...]

    def continuation(self, k: int, X: np.ndarray) -> np.ndarray:
        return self.problem.basis(k, X) @ self.coefficients[k]

    def exercise(self, k: int, X: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Exercise where ``g > 0`` and ``g`` exceeds the estimated continuation value."""
        if k == self.problem.m - 1:
            return g > 0
        return (g > 0) & (g >= self.continuation(k, X))

    def value_from(self, start: int, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Discounted (to ``times[start]``) payoff of the policy from ``times[start + 1]`` on,
        one simulated continuation per row of ``X``."""
        states, D = self.problem.simulate(X.shape[0], rng, start, X)
        out = np.zeros(X.shape[0])
        alive = np.ones(X.shape[0], bool)
        for j, k in enumerate(range(start + 1, self.problem.m)):
            g = self.problem.exercise_value(k, states[j])
            ex = alive & self.exercise(k, states[j], g)
            out[ex] = D[j][ex] * g[ex]
            alive &= ~ex
        return out


@dataclass(frozen=True)
class LSMResult:
    price: float
    stderr: float
    policy: LSMPolicy
    in_sample: float


def lsm(
    problem: ExerciseProblem, n_paths: int, seed: int = 0, n_train: "int | None" = None
) -> LSMResult:
    """Fit the policy on ``n_train`` paths, then value it on ``n_paths`` independent paths."""
    rng = np.random.default_rng(seed)
    n_fit = n_train or n_paths
    states, D = problem.simulate(n_fit, rng)
    m = problem.m
    # realised discounted (to t_k) cashflow of the current policy, backwards
    g = problem.exercise_value(m - 1, states[m - 1])
    cash = np.maximum(g, 0.0)
    when = np.full(n_fit, m - 1)
    coefs: list[np.ndarray] = [np.zeros(0)] * (m - 1)
    for k in range(m - 2, -1, -1):
        gk = problem.exercise_value(k, states[k])
        disc = D[when, np.arange(n_fit)] / D[k]
        y = cash * disc
        itm = gk > 0
        A = problem.basis(k, states[k])
        if itm.sum() > A.shape[1]:
            beta, *_ = np.linalg.lstsq(A[itm], y[itm], rcond=None)
        else:
            beta = np.zeros(A.shape[1])
        coefs[k] = beta
        ex = itm & (gk >= A @ beta)
        cash = np.where(ex, gk, cash)
        when = np.where(ex, k, when)
    in_sample = float(np.mean(cash * D[when, np.arange(n_fit)]))
    policy = LSMPolicy(problem, tuple(coefs))
    x0 = np.tile(np.asarray(problem.x0, float), (n_paths, 1))
    v = policy.value_from(-1, x0, np.random.default_rng(seed + 1_000_003))
    return LSMResult(float(v.mean()), float(v.std(ddof=1) / math.sqrt(n_paths)), policy, in_sample)


def hull_white_exercise_problem(model: HullWhite, bermudan: BermudanSwaption) -> ExerciseProblem:
    """Bermudan swaption in Hull–White: state :math:`x`, basis :math:`1, x, x^2, u, u^2`
    with :math:`u` the co-terminal swap value."""
    scale = max(float(np.abs(cf.amounts).max()) for cf in bermudan.underlyings)

    def step(
        t0: float, t1: float, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x, d = hull_white_transition(model, t0, t1, X[:, 0], rng)
        return x[:, None], d

    def exercise_value(k: int, X: np.ndarray) -> np.ndarray:
        return np.asarray(bermudan.exercise_value(k, model, X[:, 0]), float)

    def basis(k: int, X: np.ndarray) -> np.ndarray:
        x = X[:, 0] / 0.01
        u = np.asarray(bermudan.underlying_value(k, model, X[:, 0]), float) / scale
        return np.column_stack([np.ones_like(x), x, x * x, u, u * u])

    return ExerciseProblem(
        np.asarray(bermudan.exercise_times, float), np.zeros(1), step, exercise_value, basis
    )
