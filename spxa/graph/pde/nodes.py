r"""
Computational-graph nodes backed by the finite-difference pricers.

Each factory returns an :class:`~spxa.graph.node.OperatorNode` whose
forward pass solves a pricing PDE and whose vector–Jacobian product supplies
derivatives to :class:`~spxa.graph.autodiff.tape.GradientTape`:

* with respect to the spot (and the instantaneous variance for Heston) from
  the grid Greeks of the same solve;
* with respect to every other input by central bump-and-revalue.

The computational domain is frozen at the first evaluation, so repeated
evaluations with perturbed inputs differ only through the inputs and the
bumped derivatives (and second derivatives obtained by differencing
first derivatives) are smooth. Results are cached on the input values.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Union

from spxa.graph.node import ConstantNode, Node, OperatorNode
from spxa.graph.types import PortDescriptor
from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.heston import HestonPDE

__all__ = ["bs_pde_price_node", "heston_pde_price_node"]

NodeLike = Union[Node, float]


def _as_node(x: NodeLike, name: str) -> Node:
    return x if isinstance(x, Node) else ConstantNode(float(x), name=name)


class _CachedPricer:
    """Memoises ``inputs -> (price, d price / d inputs[k] for grid-Greek inputs)``."""

    def __init__(
        self,
        solve: Callable[..., tuple[float, dict[int, float]]],
        bump_sizes: Sequence[float],
    ) -> None:
        self.solve = solve
        self.bump_sizes = list(bump_sizes)
        self._cache: dict[tuple[float, ...], tuple[float, dict[int, float]]] = {}

    def __call__(self, *x: float) -> tuple[float, dict[int, float]]:
        key = tuple(float(v) for v in x)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = self.solve(*key)
        return self._cache[key]

    def price(self, *x: float) -> float:
        return self(*x)[0]

    def vjp(self, adj: Any, *x: float) -> list[float]:
        base = [float(v) for v in x]
        _, grads = self(*base)
        out = []
        for k, h in enumerate(self.bump_sizes):
            if k in grads:
                out.append(float(adj) * grads[k])
                continue
            step = h * max(1.0, abs(base[k]))
            up = list(base)
            dn = list(base)
            up[k] += step
            dn[k] -= step
            out.append(float(adj) * (self.price(*up) - self.price(*dn)) / (2.0 * step))
        return out


def _check_contract(style: str, maturity: float) -> None:
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be positive")


def bs_pde_price_node(
    spot: NodeLike,
    volatility: NodeLike,
    rate: NodeLike,
    *,
    strike: float,
    maturity: float,
    option: str = "call",
    style: str = "european",
    dividend: NodeLike = 0.0,
    name: str = "bs_pde_price",
    **solver_options: Any,
) -> OperatorNode:
    """Black–Scholes PDE price as a graph node with inputs ``(spot, vol, rate, q)``.

    ``style`` is ``'european'`` or ``'american'``; ``solver_options`` go to
    :class:`~spxa.pde.black_scholes.BlackScholesPDE` (``n_space``,
    ``n_time``, ``method`` for American, ...).
    """
    _check_contract(style, maturity)
    frozen: dict[str, float] = {}

    def solve(s: float, sigma: float, r: float, q: float) -> tuple[float, dict[int, float]]:
        model = BlackScholesPDE(r, sigma, q=q)
        if "s_max" not in frozen:
            frozen["s_max"] = solver_options.get(
                "s_max", model.default_s_max(max(s, strike), strike, maturity)
            )
        opts = dict(solver_options, s_max=frozen["s_max"])
        if style == "european":
            res = model.european(strike, maturity, s, option, **opts)
        else:
            res = model.american(strike, maturity, s, option, **opts)
        return float(res.price), {0: float(res.delta)}

    pricer = _CachedPricer(solve, bump_sizes=(1e-4, 1e-4, 1e-5, 1e-5))
    inputs = [
        _as_node(spot, f"{name}_spot"),
        _as_node(volatility, f"{name}_vol"),
        _as_node(rate, f"{name}_rate"),
        _as_node(dividend, f"{name}_div"),
    ]
    node = OperatorNode(
        op_fn=pricer.price,
        inputs=inputs,
        name=name,
        output_descriptor=PortDescriptor(name="price", description=f"{style} {option} PDE price"),
        vjp_fn=pricer.vjp,
    )
    node.pde_pricer = pricer  # type: ignore[attr-defined]
    return node


def heston_pde_price_node(
    spot: NodeLike,
    v0: NodeLike,
    kappa: NodeLike,
    theta: NodeLike,
    sigma: NodeLike,
    rho: NodeLike,
    rate: NodeLike,
    *,
    strike: float,
    maturity: float,
    option: str = "call",
    style: str = "european",
    dividend: float = 0.0,
    name: str = "heston_pde_price",
    s_max: Optional[float] = None,
    **solver_options: Any,
) -> OperatorNode:
    """Heston ADI price with inputs ``(spot, v0, kappa, theta, sigma, rho, rate)``.

    Derivatives in ``spot`` and ``v0`` come from the grid; the model
    parameters are bumped (useful for calibration through the graph).
    """
    _check_contract(style, maturity)

    def solve(
        s: float, v: float, k: float, th: float, sg: float, rh: float, r: float
    ) -> tuple[float, dict[int, float]]:
        model = HestonPDE(r, k, th, sg, rh, q=dividend)
        fn = model.european if style == "european" else model.american
        res = fn(strike, maturity, s, v, option, s_max=s_max, **solver_options)
        return float(res.price), {0: float(res.delta), 1: float(res.dv)}

    pricer = _CachedPricer(solve, bump_sizes=(1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5))
    labels = ("spot", "v0", "kappa", "theta", "sigma", "rho", "rate")
    raw = (spot, v0, kappa, theta, sigma, rho, rate)
    node = OperatorNode(
        op_fn=pricer.price,
        inputs=[_as_node(x, f"{name}_{lab}") for x, lab in zip(raw, labels)],
        name=name,
        output_descriptor=PortDescriptor(
            name="price", description=f"Heston {style} {option} ADI price"
        ),
        vjp_fn=pricer.vjp,
    )
    node.pde_pricer = pricer  # type: ignore[attr-defined]
    return node
