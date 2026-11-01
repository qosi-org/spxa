r"""
spxa.pde — finite-difference PDE and PIDE solvers for pricing and transition densities.

Every Markov process in spxa has a generator :math:`\mathcal{A}`; the
Feynman–Kac theorem turns expectations of the form

.. math::
    u(x, \tau) = \mathbb{E}\!\left[e^{-\int_0^\tau r(X_s)\,ds} g(X_\tau) \,\middle|\, X_0 = x\right]

into the backward equation :math:`u_\tau = \mathcal{A}u - r u`,
:math:`u(x, 0) = g(x)`, written here in time to maturity :math:`\tau`.
This subpackage discretises such equations with second-order accurate,
verified finite-difference schemes.

Contents
--------
* grids (uniform and sinh-stretched, aligned to strikes and barriers), time
  grids with events, boundary conditions and theta / Rannacher stepping;
* early exercise: penalty, PSOR, Brennan–Schwartz and Ikonen–Toivanen;
* one-factor models: Black–Scholes (European, American, Bermudan, barrier,
  discrete dividends), local volatility and Dupire, Asian options (Večeř),
  short-rate bonds and bond options;
* generators of zoo processes (:func:`to_pde`), Fokker–Planck densities and
  partial integro-differential equations for exponential Lévy models;
* two-factor models by ADI: Heston (European and American) and options on
  two assets;
* least-squares Monte Carlo, a bridge to :mod:`spxa.risk` instruments,
  convergence studies, and closed-form references in
  :mod:`spxa.pde.reference`.
"""

from spxa.pde.adi import (
    ADIResult,
    ADISolver,
    Dirichlet2D,
    DiscreteOperator2D,
    Grid2D,
    Linear2D,
    LinearPDE2D,
    Natural2D,
    Neumann2D,
    PDESolution2D,
)
from spxa.pde.american import (
    BrennanSchwartzLCP,
    IkonenToivanenLCP,
    PSORLCP,
    PenaltyLCP,
    complementarity_residual,
    make_lcp,
)
from spxa.pde.asian import (
    AsianPDE,
    vecer_shares,
)
from spxa.pde.black_scholes import (
    BlackScholesPDE,
    DiscreteDividend,
)
from spxa.pde.boundary import (
    BoundaryCondition,
    Dirichlet,
    Linear,
    Natural,
    Neumann,
)
from spxa.pde.convergence import (
    ConvergenceStudy,
    convergence_study,
    observed_order,
    observed_order_from_values,
    richardson_extrapolate,
)
from spxa.pde.fokker_planck import (
    DensitySolution,
    FokkerPlanck1D,
)
from spxa.pde.generator import (
    DiffusionPDE,
    to_pde,
)
from spxa.pde.grid import (
    Grid1D,
    TimeGrid,
)
from spxa.pde.heston import (
    HestonPDE,
    HestonPDEResult,
)
from spxa.pde.instruments import (
    PDEQuote,
    pde_price,
    pde_pricing_fn,
    supported_instruments,
)
from spxa.pde.local_vol import (
    DupirePDE,
    DupireSurface,
    implied_volatility,
    local_vol_from_prices,
)
from spxa.pde.lsm import (
    LSMResult,
    gbm_paths,
    longstaff_schwartz,
    lsm_american,
)
from spxa.pde.operators import (
    DiscreteOperator1D,
    LinearPDE1D,
)
from spxa.pde.payoffs import (
    Payoff,
    asset_or_nothing,
    digital,
    project_payoff,
    vanilla,
)
from spxa.pde.pide import (
    LevyJumpOperator,
    LevyPIDE,
)
from spxa.pde.short_rate import (
    ShortRatePDE,
)
from spxa.pde.solution import (
    ExerciseBoundary,
    PDEPricingResult,
    PDESolution1D,
)
from spxa.pde.theta import (
    LCPStepper,
    ThetaResult,
    ThetaSolver1D,
)
from spxa.pde.two_asset import (
    TwoAssetBlackScholesPDE,
    TwoAssetPDEResult,
)
from spxa.pde import reference

__all__ = [
    # Grids, operators and time stepping
    "Grid1D",
    "TimeGrid",
    "BoundaryCondition",
    "Dirichlet",
    "Neumann",
    "Linear",
    "Natural",
    "LinearPDE1D",
    "DiscreteOperator1D",
    "ThetaSolver1D",
    "ThetaResult",
    "LCPStepper",
    # Payoffs and solutions
    "Payoff",
    "vanilla",
    "digital",
    "asset_or_nothing",
    "project_payoff",
    "PDESolution1D",
    "PDEPricingResult",
    "ExerciseBoundary",
    # Early exercise
    "PenaltyLCP",
    "PSORLCP",
    "BrennanSchwartzLCP",
    "IkonenToivanenLCP",
    "make_lcp",
    "complementarity_residual",
    # One-factor models
    "BlackScholesPDE",
    "DiscreteDividend",
    "AsianPDE",
    "vecer_shares",
    "ShortRatePDE",
    "DupirePDE",
    "DupireSurface",
    "implied_volatility",
    "local_vol_from_prices",
    # Generators, densities and jumps
    "DiffusionPDE",
    "to_pde",
    "FokkerPlanck1D",
    "DensitySolution",
    "LevyJumpOperator",
    "LevyPIDE",
    # Two dimensions (ADI)
    "Grid2D",
    "Dirichlet2D",
    "Neumann2D",
    "Natural2D",
    "Linear2D",
    "LinearPDE2D",
    "DiscreteOperator2D",
    "ADISolver",
    "ADIResult",
    "PDESolution2D",
    "HestonPDE",
    "HestonPDEResult",
    "TwoAssetBlackScholesPDE",
    "TwoAssetPDEResult",
    # Monte Carlo cross-checks, risk bridge and verification
    "longstaff_schwartz",
    "lsm_american",
    "gbm_paths",
    "LSMResult",
    "pde_price",
    "pde_pricing_fn",
    "PDEQuote",
    "supported_instruments",
    "ConvergenceStudy",
    "convergence_study",
    "observed_order",
    "observed_order_from_values",
    "richardson_extrapolate",
    # Closed-form references (submodule)
    "reference",
]
