"""
Kou Double Exponential Jump-Diffusion Process.

Kou (2002) model combines continuous Brownian diffusion with compound
Poisson jump shocks whose sizes follow an asymmetric double exponential distribution:

    X_t = μ t + σ W_t + Σ_{i=1}^{N_t} Y_i

where:
  - W_t is standard Brownian motion with diffusion volatility σ > 0.
  - N_t ~ Poisson(λ t) is a homogeneous Poisson process with arrival intensity λ > 0.
  - Y_i are i.i.d. asymmetric double exponential jump sizes with density:
        f_Y(y) = p η_1 e^{-η_1 y} 𝟙_{y ≥ 0} + (1 - p) η_2 e^{η_2 y} 𝟙_{y < 0}
    with p ∈ [0, 1], η_1 > 1 (required for E[e^Y] < ∞), and η_2 > 0.

The process is an EXACT Lévy process with finite activity jump measure:
    ν(dy) = λ f_Y(y) dy

It provides exact closed-form characteristic functions, analytical cumulants of
all orders, exact simulation without discretization error, and European option pricing
via Gil–Pelaez / Carr–Madan Fourier inversion.

References
----------
Kou, S. G. (2002). A jump-diffusion model for option pricing.
    *Management Science*, 48(8), 1086–1101.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Chapman & Hall/CRC. Section 3.5.

Kou, S. G. & Wang, H. (2004). Option pricing under a double exponential
    jump diffusion model. *Management Science*, 50(9), 1178–1192.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Optional, Union

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_kou
from spxa.story.narrator import CompositionNode, NodeKind
from spxa.zoo.levy.compound_poisson import DoubleExponentialJump, _CompoundPoissonLevyMeasure


class KouJumpDiffusion(Process):
    r"""
    Kou (2002) double exponential jump-diffusion process.

    .. math::

        X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i

    where :math:`W_t` is standard Brownian motion, :math:`N_t \sim \mathrm{Poisson}(\lambda t)`,
    and :math:`Y_i` has an asymmetric double exponential density:

    .. math::

        f_Y(y) = p \eta_1 e^{-\eta_1 y} \mathbb{I}_{y \ge 0} + (1 - p) \eta_2 e^{\eta_2 y} \mathbb{I}_{y < 0}

    Parameters
    ----------
    drift : float, default 0.0
        Continuous drift parameter :math:`\mu \in \mathbb{R}`.
    sigma : float, default 0.2
        Continuous diffusion volatility :math:`\sigma > 0`.
    rate : float, default 1.0
        Jump arrival intensity :math:`\lambda > 0`.
    p : float, default 0.4
        Probability of an upward jump :math:`p \in [0, 1]`.
    eta1 : float, default 10.0
        Decay rate of upward jumps :math:`\eta_1 > 1`.
    eta2 : float, default 5.0
        Decay rate of downward jumps :math:`\eta_2 > 0`.
    """

    def __init__(
        self,
        drift: float = 0.0,
        sigma: float = 0.2,
        rate: float = 1.0,
        p: float = 0.4,
        eta1: float = 10.0,
        eta2: float = 5.0,
        **kwargs,
    ) -> None:
        drift_val = kwargs.pop("mu", drift)
        sigma_val = kwargs.pop("vol", sigma)
        rate_val = kwargs.pop("lam", kwargs.pop("lambda_", kwargs.pop("jump_rate", rate)))
        p_val = kwargs.pop("prob_up", p)
        eta1_val = kwargs.pop("eta_1", eta1)
        eta2_val = kwargs.pop("eta_2", eta2)

        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {list(kwargs.keys())}")

        if sigma_val <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma_val}")
        if rate_val <= 0:
            raise ValueError(f"rate must be strictly positive, got {rate_val}")
        if not (0.0 <= p_val <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p_val}")
        if eta1_val <= 1.0:
            raise ValueError(
                f"eta1 must be strictly greater than 1.0 for finite expectation, got {eta1_val}"
            )
        if eta2_val <= 0:
            raise ValueError(f"eta2 must be strictly positive, got {eta2_val}")

        self.drift = float(drift_val)
        self.sigma = float(sigma_val)
        self.rate = float(rate_val)
        self.p = float(p_val)
        self.eta1 = float(eta1_val)
        self.eta2 = float(eta2_val)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"KouJumpDiffusion(drift={self.drift}, sigma={self.sigma}, rate={self.rate}, p={self.p}, eta1={self.eta1}, eta2={self.eta2})",
            math_note=(
                f"X_t = {self.drift}t + {self.sigma}W_t + Σ Y_i; "
                f"λ={self.rate}, p={self.p}, η_1={self.eta1}, η_2={self.eta2}"
            ),
            reference="Kou, S. G. (2002), Management Science 48(8), 1086–1101",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def __repr__(self) -> str:
        return (
            f"KouJumpDiffusion(drift={self.drift}, sigma={self.sigma}, "
            f"rate={self.rate}, p={self.p}, eta1={self.eta1}, eta2={self.eta2})"
        )

    # ------------------------------------------------------------------
    # Canonical aliases and properties
    # ------------------------------------------------------------------

    @property
    def mu(self) -> float:
        """Alias for continuous drift."""
        return self.drift

    @property
    def lam(self) -> float:
        """Alias for jump intensity rate."""
        return self.rate

    @property
    def jump_rate(self) -> float:
        """Alias for jump arrival intensity."""
        return self.rate

    @property
    def q(self) -> float:
        """Probability of downward jump 1 - p."""
        return 1.0 - self.p

    @property
    def jump_mean(self) -> float:
        r"""Mean jump size E[Y] = p/η_1 - (1-p)/η_2."""
        return self.p / self.eta1 - self.q / self.eta2

    @property
    def jump_var(self) -> float:
        r"""Jump size variance Var(Y) = 2p/η_1² + 2(1-p)/η_2² - (E[Y])²."""
        raw2 = 2.0 * self.p / (self.eta1**2) + 2.0 * self.q / (self.eta2**2)
        return float(raw2 - self.jump_mean**2)

    @property
    def expected_relative_jump(self) -> float:
        r"""
        Expected relative price jump:
        \kappa = E[e^Y - 1] = p \eta_1 / (\eta_1 - 1) + (1-p) \eta_2 / (\eta_2 + 1) - 1.
        """
        term_up = self.p * self.eta1 / (self.eta1 - 1.0)
        term_down = self.q * self.eta2 / (self.eta2 + 1.0)
        return float(term_up + term_down - 1.0)

    # ------------------------------------------------------------------
    # Lévy triplet & properties
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        r"""
        Exact Lévy triplet (b, σ², ν) with truncation cutoff c = 1.

        Truncated drift:
            b = \mu + \lambda \int_{-1}^1 y f_Y(y) dy
              = \mu + \lambda [ \frac{p}{\eta_1} (1 - e^{-\eta_1}(1+\eta_1)) - \frac{1-p}{\eta_2} (1 - e^{-\eta_2}(1+\eta_2)) ]
        """
        int_pos = (self.p / self.eta1) * (1.0 - math.exp(-self.eta1) * (1.0 + self.eta1))
        int_neg = (self.q / self.eta2) * (1.0 - math.exp(-self.eta2) * (1.0 + self.eta2))
        b = self.drift + self.rate * (int_pos - int_neg)

        jump_dist = DoubleExponentialJump(p=self.p, eta1=self.eta1, eta2=self.eta2)
        levy_meas = _CompoundPoissonLevyMeasure(self.rate, jump_dist)
        return LevyTriplet(b=float(b), sigma_sq=float(self.sigma**2), nu=levy_meas)

    def _properties(self) -> ProcessProperties:
        mean_rate = self.drift + self.rate * self.jump_mean
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=bool(abs(mean_rate) < 1e-12),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=float(self.eta1),
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"Kou (2002) jump-diffusion: drift={self.drift}, sigma={self.sigma}, "
                f"rate={self.rate}, p={self.p}, eta1={self.eta1}, eta2={self.eta2}.",
                f"Mean jump size: {self.jump_mean:.4f}, expected relative jump: {self.expected_relative_jump:.4f}.",
                "Asymmetric semi-heavy double exponential tails capturing asymmetric skewness.",
            ],
        )

    # ------------------------------------------------------------------
    # Characteristic exponent & cumulants
    # ------------------------------------------------------------------

    def char_exp(self, u: complex) -> complex:
        r"""
        Characteristic exponent ψ(u) such that E[exp(i u X_t)] = exp(t ψ(u)).

        ψ(u) = i μ u - ½ σ² u² + λ [ p η_1 / (η_1 - i u) + (1-p) η_2 / (η_2 + i u) - 1 ]
        """
        jump_part = (
            (self.p * self.eta1) / (self.eta1 - 1j * u)
            + (self.q * self.eta2) / (self.eta2 + 1j * u)
            - 1.0
        )
        return 1j * self.drift * u - 0.5 * (self.sigma**2) * (u**2) + self.rate * jump_part

    def characteristic_function(self, u: complex, t: float = 1.0) -> complex:
        """Characteristic function E[exp(i u X_t)] = exp(t ψ(u))."""
        if t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return cmath.exp(t * self.char_exp(u))

    def char_func_exact(self, u: float | complex | np.ndarray, t: float = 1.0) -> Any:
        r"""
        Closed-form characteristic function :math:`\varphi(u; t) = e^{t\psi(u)}`.

        Vectorised over ``u``. The exponent is valid for complex arguments with
        :math:`-\eta_1 < \operatorname{Im}(u) < \eta_2`, which includes the
        exponential moment :math:`u = -i` whenever :math:`\eta_1 > 1`.
        """
        if t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        jump_part = (
            self.p * self.eta1 / (self.eta1 - 1j * u_arr)
            + self.q * self.eta2 / (self.eta2 + 1j * u_arr)
            - 1.0
        )
        exponent = 1j * self.drift * u_arr - 0.5 * self.sigma**2 * u_arr**2 + self.rate * jump_part
        result = np.exp(t * exponent)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | complex | np.ndarray, t: float = 1.0) -> Any:
        """Characteristic function, evaluated in closed form (see :meth:`char_func_exact`)."""
        return self.char_func_exact(u=u, t=t)

    def cumulants(self, order: int = 4, t: float = 1.0) -> dict[int, float]:
        r"""
        Exact analytical cumulants κ_n(t) for n = 1, ..., order.

        κ_1(t) = t (μ + λ E[Y])
        κ_2(t) = t (σ² + λ E[Y²])
        κ_n(t) = t λ n! [ p / η_1^n + (-1)^n (1-p) / η_2^n ]  for n ≥ 3.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        if t < 0:
            raise ValueError("t must be >= 0")

        res: dict[int, float] = {}
        res[1] = t * (self.drift + self.rate * self.jump_mean)

        raw2 = 2.0 * self.p / (self.eta1**2) + 2.0 * self.q / (self.eta2**2)
        res[2] = t * (self.sigma**2 + self.rate * raw2)

        for n in range(3, order + 1):
            term_up = self.p / (self.eta1**n)
            term_down = ((-1.0) ** n) * self.q / (self.eta2**n)
            res[n] = t * self.rate * math.factorial(n) * (term_up + term_down)

        return res

    # ------------------------------------------------------------------
    # Analytical European option pricing & implied volatility smile
    # ------------------------------------------------------------------

    def call_price(
        self,
        strikes: Union[float, list[float], np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact European call option pricing under Kou jump-diffusion via Fourier inversion.

        Parameters
        ----------
        strikes : float or array-like
            Strike prices K > 0.
        T : float, default 1.0
            Maturity T > 0.
        S0 : float, default 100.0
            Initial spot price S_0 > 0.
        r : float, default 0.05
            Risk-free rate.
        q : float, default 0.0
            Continuous dividend yield.
        """
        if T <= 0:
            raise ValueError("Maturity T must be strictly positive")
        if S0 <= 0:
            raise ValueError("S0 must be strictly positive")

        is_scalar = np.isscalar(strikes)
        k_arr = np.atleast_1d(np.asarray(strikes, dtype=float))
        if np.any(k_arr <= 0.0):
            raise ValueError("All strikes must be strictly positive")

        # Risk-neutral drift condition: mu_rn = r - q - 0.5*sigma^2 - lambda * kappa
        kappa = self.expected_relative_jump
        mu_rn = r - q - 0.5 * (self.sigma**2) - self.rate * kappa

        def rn_char_func(u: complex) -> complex:
            jump_part = (
                (self.p * self.eta1) / (self.eta1 - 1j * u)
                + (self.q * self.eta2) / (self.eta2 + 1j * u)
                - 1.0
            )
            psi = 1j * mu_rn * u - 0.5 * (self.sigma**2) * (u**2) + self.rate * jump_part
            return cmath.exp(T * psi)

        phi_neg_i = rn_char_func(-1j).real  # Equal to exp((r - q) * T)

        prices = np.empty_like(k_arr)
        for idx, K in enumerate(k_arr):
            k_log = math.log(K / S0)

            def integrand1(u: float) -> float:
                if u == 0.0:
                    return 0.0
                val = (cmath.exp(-1j * u * k_log) * rn_char_func(u - 1j)) / (1j * u * phi_neg_i)
                return val.real

            def integrand2(u: float) -> float:
                if u == 0.0:
                    return 0.0
                val = (cmath.exp(-1j * u * k_log) * rn_char_func(u)) / (1j * u)
                return val.real

            int1, _ = quad(integrand1, 0.0, 150.0, limit=100)
            int2, _ = quad(integrand2, 0.0, 150.0, limit=100)

            pi1 = 0.5 + int1 / math.pi
            pi2 = 0.5 + int2 / math.pi

            call = S0 * math.exp(-q * T) * pi1 - K * math.exp(-r * T) * pi2
            prices[idx] = max(call, 0.0)

        return float(prices[0]) if is_scalar else prices

    def put_price(
        self,
        strikes: Union[float, list[float], np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact European put option pricing via Put-Call parity:
            P(K, T) = C(K, T) - S_0 e^{-q T} + K e^{-r T}.
        """
        calls = self.call_price(strikes, T=T, S0=S0, r=r, q=q)
        k_arr = np.asarray(strikes, dtype=float)
        parity = np.maximum(calls - S0 * math.exp(-q * T) + k_arr * math.exp(-r * T), 0.0)
        return float(parity) if np.isscalar(strikes) else parity

    def implied_vol_smile(
        self,
        strikes: Union[float, list[float], np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""Black–Scholes implied volatility smile corresponding to Kou call prices."""
        calls = np.atleast_1d(self.call_price(strikes, T=T, S0=S0, r=r, q=q))
        k_arr = np.atleast_1d(np.asarray(strikes, dtype=float))
        vols = np.empty_like(calls)

        for i, (K, C) in enumerate(zip(k_arr, calls)):
            intrinsic = max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
            if C <= intrinsic:
                vols[i] = 1e-4
                continue

            def obj(sig: float) -> float:
                d1 = (math.log(S0 / K) + (r - q + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
                d2 = d1 - sig * math.sqrt(T)
                bs = S0 * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(
                    -r * T
                ) * stats.norm.cdf(d2)
                return bs - C

            try:
                vols[i] = brentq(obj, 1e-4, 5.0)
            except Exception:
                vols[i] = np.nan

        return float(vols[0]) if np.isscalar(strikes) else vols

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        T: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate exact sample paths of the Kou jump-diffusion process.
        """
        horizon = T if T is not None else time_horizon
        return sample_kou(
            drift=self.drift,
            sigma=self.sigma,
            rate=self.rate,
            p=self.p,
            eta1=self.eta1,
            eta2=self.eta2,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=horizon,
            rng=rng,
        )


# Aliases
KouProcess = KouJumpDiffusion
Kou72 = KouJumpDiffusion
Kou02 = KouJumpDiffusion
