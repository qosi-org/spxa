"""
Base Process class.

All stochastic processes in spxa inherit from Process. The class is immutable:
every operator returns a new Process instance. This makes composition trees
inspectable and reproducible, and is what makes .__story__() possible.

Design invariants:
  1. Process instances are immutable after construction.
  2. Every operation produces a new Process with updated triplet, properties,
     exactness level, and composition node.
  3. ExactnessLevel degrades to the minimum of operands under any operation.
  4. SpxaDegradationWarning is emitted whenever exactness degrades, with a
     mathematical reason.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind, narrate


class Process(ABC):
    """
    Abstract base class for all stochastic processes in spxa.

    Subclasses must implement `_triplet()`, `_properties()`, and `simulate()`.
    Operator overloading (`+`, `*`, `@`) is implemented here and returns
    composed Process instances.

    Parameters
    ----------
    exactness :
        Analytical guarantee for this process.
    _node :
        Composition tree node. Set automatically by operators; primitive
        processes set this in their own __init__.
    """

    def __init__(
        self,
        exactness: ExactnessLevel = ExactnessLevel.EXACT,
        _node: Optional[CompositionNode] = None,
    ) -> None:
        self._exactness = exactness
        self._node = _node or CompositionNode(kind=NodeKind.PRIMITIVE, name=type(self).__name__)

    @property
    def exactness(self) -> ExactnessLevel:
        """The ExactnessLevel of this process."""
        return self._exactness

    @property
    def triplet(self) -> LevyTriplet:
        """
        The Lévy–Khintchine characteristic triplet (b, σ², ν).

        Only available when exactness == ExactnessLevel.EXACT.

        Raises
        ------
        ExactnessError
            If the process does not have an exact triplet.
        """
        if self._exactness != ExactnessLevel.EXACT:
            raise ExactnessError(
                method="triplet",
                required="EXACT",
                actual=self._exactness.name,
                reason=self._degradation_reason(),
            )
        return self._triplet()

    @property
    def properties(self) -> ProcessProperties:
        """The property lattice of this process."""
        return self._properties()

    @abstractmethod
    def _triplet(self) -> LevyTriplet:
        """Return the LevyTriplet. Called only when exactness == EXACT."""

    @abstractmethod
    def _properties(self) -> ProcessProperties:
        """Return the ProcessProperties for this process."""

    @abstractmethod
    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate sample paths.

        Parameters
        ----------
        n_steps :
            Number of time steps per path.
        n_paths :
            Number of independent paths.
        T :
            Terminal time. Steps are equally spaced on [0, T].
        rng :
            Random number generator. If None, uses numpy default.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == 0 always.
        """

    def cumulants(self, order: int) -> dict[int, float]:
        """
        Return cumulants κ_1, …, κ_order of X_1 (per unit time).

        Available for EXACT and MOMENT_PROPAGATION processes up to the
        highest order with a finite moment.

        Parameters
        ----------
        order :
            Highest cumulant order to compute.
        """
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="cumulants",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT:
            t = self._triplet()
            return {n: t.cumulant(n) for n in range(1, order + 1)}
        return self._moment_propagation_cumulants(order)

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Evaluate the characteristic function E[e^{iuX_t}] at frequency u.

        Exact for EXACT processes (via Lévy–Khintchine). Not available for
        SIMULATION_ONLY.

        Parameters
        ----------
        u :
            Frequency or array of frequencies.
        t :
            Time. Default 1.
        """
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="char_func",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT:
            return self._triplet().char_exp(u, t)
        raise ExactnessError(
            method="char_func",
            required="EXACT",
            actual=self._exactness.name,
            reason="Characteristic function requires an exact triplet.",
        )

    def __story__(self) -> str:
        """
        Return a human-readable derivation of this process's distributional
        properties, tracing how the current triplet and property lattice
        were obtained through composition.

        The output is designed to be readable in a terminal or rendered as
        a Markdown/LaTeX block in a notebook.
        """
        lines = ["=" * 60]
        lines.append(f"Story of: {self!r}")
        lines.append(f"Exactness: {self._exactness.name}")
        lines.append("-" * 60)
        lines.append("Composition trace:")
        lines.append(narrate(self._node))
        lines.append("-" * 60)
        lines.append("Properties:")
        props = self._properties()
        lines.append(f"  Stationary increments : {props.has_stationary_increments}")
        lines.append(f"  Independent increments: {props.has_independent_increments}")
        lines.append(f"  Martingale            : {props.is_martingale}")
        lines.append(f"  Finite mean           : {props.has_finite_mean}")
        lines.append(f"  Finite variance       : {props.has_finite_variance}")
        if props.tail_index is not None:
            lines.append(f"  Tail index α          : {props.tail_index}")
        if props.self_similarity_index is not None:
            lines.append(f"  Self-similarity H     : {props.self_similarity_index}")
        if props.hurst_index is not None:
            lines.append(f"  Hurst index H         : {props.hurst_index}")
        if props.notes:
            lines.append("  Notes:")
            for note in props.notes:
                lines.append(f"    - {note}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __story_latex__(self) -> str:
        """
        Return a LaTeX derivation of this process's composition and properties.

        Output is a ``\\begin{aligned}...\\end{aligned}`` block suitable for
        display in a Jupyter notebook:

        .. code-block:: python

            from IPython.display import display, Math
            display(Math(Z.__story_latex__()))

        See Also
        --------
        __story__ : Plain-text version.
        spxa.story.story_latex : Standalone function.
        """
        from spxa.story.narrator import story_latex

        return story_latex(self)

    def _degradation_reason(self) -> str:
        """Return the reason for exactness degradation from the composition notes."""
        notes = self._properties().notes
        if notes:
            return notes[-1]
        return (
            f"This process has ExactnessLevel.{self._exactness.name}. "
            "Check .__story__() for the composition history."
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        Fallback cumulant computation for MOMENT_PROPAGATION processes.
        Subclasses should override with exact formulas where available.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement moment propagation cumulants. "
            "Override _moment_propagation_cumulants() or use simulate() and compute empirically."
        )

    def __add__(self, other: Process) -> _ComposedProcess:
        """
        Return the process Z = X + Y (sum of independent processes).

        If both are EXACT Lévy processes, the result is EXACT with triplet
        (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y).

        Otherwise degrades to the minimum ExactnessLevel with a warning.
        """
        new_exactness = ExactnessLevel.combine(self._exactness, other._exactness)

        if new_exactness != ExactnessLevel.EXACT:
            reason = (
                f"Adding {type(self).__name__} (ExactnessLevel.{self._exactness.name}) "
                f"and {type(other).__name__} (ExactnessLevel.{other._exactness.name}): "
                "result is not an exact Lévy process."
            )
            warnings.warn(reason, SpxaDegradationWarning, stacklevel=2)

        new_props = ProcessProperties.combine_addition(self._properties(), other._properties())
        node = CompositionNode(
            kind=NodeKind.ADDITION,
            name=f"{self!r} + {other!r}",
            children=[self._node, other._node],
            math_note="Triplet addition: (b,σ²,ν) = (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y)",
            reference="Sato (1999), Prop. 11.10",
        )
        return _ComposedProcess(
            left=self,
            right=other,
            operation="+",
            exactness=new_exactness,
            props=new_props,
            node=node,
        )

    def __radd__(self, other: object) -> Process:
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, c: float) -> _ScaledProcess:
        """Return the process Z = c * X."""
        if not isinstance(c, (int, float)):
            return NotImplemented
        return _make_scaled(self, float(c))

    def __rmul__(self, c: float) -> _ScaledProcess:
        """Return the process Z = c * X (right multiplication)."""
        if not isinstance(c, (int, float)):
            return NotImplemented
        return _make_scaled(self, float(c))

    def __neg__(self) -> _ScaledProcess:
        return _make_scaled(self, -1.0)

    def __sub__(self, other: Process) -> Process:
        return self.__add__(_make_scaled(other, -1.0))

    def __matmul__(self, subordinator: Process) -> _ComposedProcess:
        """
        Return the subordinated process Z_t = X_{T_t}.

        The left operand X is the parent process, the right operand T must
        be a subordinator (non-decreasing Lévy process).

        Reference: Sato (1999), Theorem 30.1.
        """
        if not subordinator._properties().is_subordinator:
            raise ValueError(
                f"{type(subordinator).__name__} is not a subordinator. "
                "The right operand of @ must be a non-decreasing Lévy process."
            )

        new_exactness = ExactnessLevel.combine(self._exactness, subordinator._exactness)
        new_props = ProcessProperties.combine_subordination(
            self._properties(), subordinator._properties()
        )
        node = CompositionNode(
            kind=NodeKind.SUBORDINATION,
            name=f"{self!r} @ {subordinator!r}",
            children=[self._node, subordinator._node],
            math_note="ψ_Z(u) = -φ(-ψ_X(u)) where φ is the Laplace exponent of T",
            reference="Sato (1999), Thm. 30.1",
        )
        return _ComposedProcess(
            left=self,
            right=subordinator,
            operation="@",
            exactness=new_exactness,
            props=new_props,
            node=node,
        )

    def morph(
        self,
        other: Process,
        alpha: float = 0.5,
        method: str = "triplet",
        coupling: str = "independent",
        **kwargs: Any,
    ) -> Any:
        r"""
        Morph this process into ``other`` via homotopy parameter :math:`\alpha \in [0, 1]`.

        .. math::
            P_\alpha = \text{Morph}(P_0, P_1; \alpha)

        Parameters
        ----------
        other :
            Target process corresponding to :math:`\alpha = 1`.
        alpha :
            Homotopy parameter in :math:`[0, 1]`. Default 0.5.
        method :
            Morphing scheme: ``"triplet"``, ``"generator"``, ``"geodesic"``,
            ``"wasserstein"``, or ``"convex_combination"``.
        coupling :
            Simulation coupling: ``"independent"`` or ``"common"``.

        Returns
        -------
        MorphedProcess
            A new MorphedProcess instance.
        """
        from spxa.ops.morph import morph

        return morph(self, other, alpha=alpha, method=method, coupling=coupling, **kwargs)

    def to_pde(self, **kwargs: Any) -> Any:
        r"""
        Pricing / Kolmogorov equation of this process.

        Delegates to :func:`spxa.pde.to_pde`. Diffusions (Brownian motion,
        Ornstein–Uhlenbeck, CIR, CEV) give a
        :class:`~spxa.pde.generator.DiffusionPDE` for the backward equation
        :math:`u_\tau = \mathcal{A}u` and its Fokker–Planck adjoint; Lévy
        processes with jumps give the exponential-Lévy pricing PIDE
        :class:`~spxa.pde.pide.LevyPIDE` and require ``r`` (and optionally
        ``q``).

        Raises
        ------
        NotImplementedError
            When no PDE representation is registered for this process.
        """
        from spxa.pde.generator import to_pde

        return to_pde(self, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(exactness={self._exactness.name})"


class _ScaledProcess(Process):
    """Process Z = c * X. Internal — users get this via X * c or c * X."""

    def __init__(
        self,
        base: Process,
        scale: float,
        exactness: ExactnessLevel,
        props: ProcessProperties,
        node: CompositionNode,
    ) -> None:
        super().__init__(exactness=exactness, _node=node)
        self._base = base
        self._scale = scale
        self._props = props

    def _triplet(self) -> LevyTriplet:
        return self._base._triplet().scale(self._scale)

    def _properties(self) -> ProcessProperties:
        return self._props

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        paths = self._base.simulate(n_steps, n_paths, T, rng)
        return self._scale * paths

    def __repr__(self) -> str:
        return f"{self._scale} * {self._base!r}"


class _ComposedProcess(Process):
    """Process produced by + or @. Internal."""

    def __init__(
        self,
        left: Process,
        right: Process,
        operation: str,
        exactness: ExactnessLevel,
        props: ProcessProperties,
        node: CompositionNode,
    ) -> None:
        super().__init__(exactness=exactness, _node=node)
        self._left = left
        self._right = right
        self._operation = operation
        self._props = props

    def _triplet(self) -> LevyTriplet:
        if self._operation == "+":
            return self._left._triplet() + self._right._triplet()
        raise ExactnessError(
            method="_triplet",
            required="EXACT",
            actual=self._exactness.name,
            reason="Subordinated processes do not have a simple triplet addition rule; "
            "use char_func() or simulate().",
        )

    def _properties(self) -> ProcessProperties:
        return self._props

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if self._operation == "+":
            return self._left.simulate(n_steps, n_paths, T, rng) + self._right.simulate(
                n_steps, n_paths, T, rng
            )
        if self._operation == "@":
            # Subordination: Z_t = X_{T_t}
            # Simulate the subordinator to get random time changes,
            # then simulate the parent process at those random times.
            # Uses a path-by-path approach: for each path, simulate the
            # subordinator time grid, then interpolate the parent process.
            rng = rng or np.random.default_rng()

            # Simulate subordinator paths on a fine grid
            sub_paths = self._right.simulate(n_steps, n_paths, T, rng)
            # sub_paths[:, k] = T_k (random time at step k)

            # For each path, simulate parent at fine resolution and interpolate
            # We simulate the parent on a finer grid covering [0, max(T_T)]
            max_time = float(sub_paths[:, -1].max()) * 1.5 + 1e-6
            fine_steps = max(n_steps * 4, 500)
            parent_paths = self._left.simulate(fine_steps, n_paths, max_time, rng)
            fine_grid = np.linspace(0, max_time, fine_steps + 1)

            paths = np.zeros((n_paths, n_steps + 1))
            for i in range(n_paths):
                paths[i] = np.interp(sub_paths[i], fine_grid, parent_paths[i])

            return paths
        raise NotImplementedError(f"Simulation not implemented for operation '{self._operation}'.")

    def __repr__(self) -> str:
        return f"({self._left!r} {self._operation} {self._right!r})"


def _make_scaled(base: Process, c: float) -> _ScaledProcess:
    new_exactness = base._exactness
    new_props = ProcessProperties.combine_scaling(base._properties(), c)
    node = CompositionNode(
        kind=NodeKind.SCALING,
        name=f"{c} * {base!r}",
        children=[base._node],
        math_note=f"Triplet scaling by c={c}: (cb+Δb, c²σ², ν(·/c))",
        reference="Sato (1999), Prop. 11.10",
    )
    return _ScaledProcess(
        base=base,
        scale=c,
        exactness=new_exactness,
        props=new_props,
        node=node,
    )
