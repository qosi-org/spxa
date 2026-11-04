r"""
Backward-induction engines for Bermudan swaptions.

On a lattice the Bermudan value is the dynamic programme

.. math:: V(t_m) = g_m, \qquad V(t_k) = \max\bigl(g_k, E_{t_k}[D(t_k, t_{k+1})V(t_{k+1})]\bigr),

rolled back with the model's pricing equation between exercise dates. The
exercise values are the model values of the co-terminal swaps,
:math:`g_k(x) = (\sum_i a^{(k)}_i P(t_k, t_i \mid x))^+`, evaluated in closed
form at the lattice nodes.

* :func:`bermudan_pde` uses :class:`~spxa.rates.hw_pde.HullWhitePDE`, where
  the exercise dates are events of the time grid, so they are hit exactly.
* :func:`bermudan_pde` also accepts a
  :class:`~spxa.rates.cheyette_pde.CheyettePDE`; the exercise values are then
  functions of both states, :math:`P(t_k, t_i \mid x, y)`.
* :func:`bermudan_tree` uses a Hull–White trinomial tree, mapping node short
  rates to the model state through :math:`x = r - \varphi(t)`. Exercise
  dates must lie on the tree's uniform grid (see :func:`bermudan_from_times`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.rates.bermudan import BermudanSwaption
from spxa.rates.cheyette_pde import CheyettePDE
from spxa.rates.hw_pde import HullWhitePDE
from spxa.rates.hw_tree import HullWhiteTree

__all__ = ["bermudan_pde", "bermudan_tree"]


def bermudan_pde(engine: Any, bermudan: BermudanSwaption) -> float:
    """Bermudan value on a finite-difference engine."""
    t = np.asarray(bermudan.exercise_times, float)
    if isinstance(engine, HullWhitePDE):
        m = engine.model

        def g(k: int) -> Any:
            return lambda x: bermudan.exercise_value(k, m, x)

        exercise = {float(t[k]): g(k) for k in range(t.size - 1)}
        return engine.value(g(t.size - 1), float(t[-1]), exercise=exercise)
    if isinstance(engine, CheyettePDE):
        c = engine.model

        def h(k: int) -> Any:
            cf, tk = bermudan.underlyings[k], float(t[k])

            def g(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
                v = sum(a * c.discount_bond(tk, ti, X, Y) for ti, a in zip(cf.times, cf.amounts))
                return np.maximum(np.asarray(v, float), 0.0)

            return g

        return engine.solve(
            h(t.size - 1), float(t[-1]), exercise={float(t[k]): h(k) for k in range(t.size - 1)}
        )
    raise TypeError(f"no Bermudan PDE for {type(engine).__name__}")


def bermudan_tree(tree: HullWhiteTree, bermudan: BermudanSwaption) -> float:
    """Bermudan value on a Hull–White trinomial tree.

    The exercise dates must be tree steps, and the tree's horizon must
    extend at least one step beyond the last of them.
    """
    m = tree.model
    t = np.asarray(bermudan.exercise_times, float)
    steps = {tree.step(float(tk)): k for k, tk in enumerate(t)}
    if max(steps) >= tree.n_steps:
        raise ValueError("the tree must extend beyond the last exercise date")

    def value(n: int, r: np.ndarray) -> np.ndarray:
        k = steps[n]
        tk = float(t[k])
        return np.asarray(bermudan.exercise_value(k, m, r - m.phi(tk)), float)

    return tree.price(value, float(t[-1]), exercise=list(t[:-1]), exercise_value=value)
