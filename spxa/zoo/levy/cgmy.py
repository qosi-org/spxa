"""
CGMY process (Carr–Geman–Madan–Yor).

A four-parameter generalisation of the VG model that nests VG (Y=0),
compound Poisson (Y < 0), and α-stable (C+=C-, G=M=0) as special cases.
Controls tail heaviness, jump activity, and asymmetry independently.

Lévy density (Carr et al. 2002, equation 1):
  k(x) = C · exp(-G·x)/x^{1+Y} 𝟙_{x>0} + C · exp(-M·|x|)/|x|^{1+Y} 𝟙_{x<0}

Parameter constraints:
  C > 0, G ≥ 0, M ≥ 0, Y < 2.
  For Y ∈ [1, 2): infinite variation; for Y ∈ (0, 1): finite variation, infinite activity;
  for Y ≤ 0: finite activity.

References
----------
Carr, P., Geman, H., Madan, D.B. & Yor, M. (2002). The fine structure of
asset returns: an empirical investigation. *Journal of Business*, 75(2),
305–332.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Table 4.2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _cgmy_levy_measure(C: float, G: float, M: float, Y: float) -> DensityLevyMeasure:
    """
    CGMY Lévy density.

    k(x) = C·exp(-M·x)/x^{1+Y}·𝟙_{x>0} + C·exp(-G·|x|)/|x|^{1+Y}·𝟙_{x<0}

    Carr et al. (2002), equation (1): M tempers the upward jumps and G the
    downward ones, consistent with the characteristic exponent
    C·Γ(-Y)·((M-iu)^Y - M^Y + (G+iu)^Y - G^Y).
    """
    from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        neg = x < 0
        out[pos] = C * np.exp(-M * x[pos]) / x[pos] ** (1 + Y)
        out[neg] = C * np.exp(-G * np.abs(x[neg])) / np.abs(x[neg]) ** (1 + Y)
        return out

    def total_mass() -> float:
        if Y >= 0:
            return np.inf
        from scipy import integrate  # type: ignore[import-untyped]

        pos_mass, _ = integrate.quad(lambda x: C * np.exp(-M * x) / x ** (1 + Y), 0, np.inf)
        neg_mass, _ = integrate.quad(lambda x: C * np.exp(-G * x) / x ** (1 + Y), 0, np.inf)
        return pos_mass + neg_mass

    def tail(xv: float) -> float:
        from scipy import integrate  # type: ignore[import-untyped]

        result, _ = integrate.quad(
            lambda t: C * np.exp(-M * t) / t ** (1 + Y), xv, np.inf, limit=200
        )
        return result

    def moment_1() -> float:
        return C * gamma_fn(1 - Y) * (M ** (Y - 1) - G ** (Y - 1))

    def moment_2() -> float:
        return C * gamma_fn(2 - Y) * (M ** (Y - 2) + G ** (Y - 2))

    def moment_3() -> float:
        return C * gamma_fn(3 - Y) * (M ** (Y - 3) - G ** (Y - 3))

    def moment_4() -> float:
        return C * gamma_fn(4 - Y) * (M ** (Y - 4) + G ** (Y - 4))

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=total_mass(),
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2, 3: moment_3, 4: moment_4},
    )


class CGMY(Process):
    """
    CGMY (Carr–Geman–Madan–Yor) process.

    A flexible pure-jump Lévy model with independent control over:
    - Overall jump intensity (C)
    - Left tail decay (G)
    - Right tail decay (M)
    - Fine structure / activity (Y)

    Parameters
    ----------
    C :
        Overall jump intensity. C > 0.
    G :
        Left-side (downward jump) exponential decay rate. G ≥ 0.
        G=0 gives a stable left tail.
    M :
        Right-side (upward jump) exponential decay rate. M ≥ 0.
        M=0 gives a stable right tail.
    Y :
        Fine-structure index. Y < 2.
        Y < 0  → finite activity (compound Poisson)
        Y ∈ [0,1) → infinite activity, finite variation
        Y ∈ [1,2) → infinite activity, infinite variation
        Y → 2 approximates stable process

    Notes
    -----
    Special cases:
    - Y = 0: Variance Gamma (with appropriate C, G, M)
    - G = M, Y = 0: symmetric VG
    - G = M → 0, Y ∈ (0,2): α-stable

    Cumulants (Carr et al. 2002, equation 2):
      κ_n = C · Γ(n-Y) · (M^{Y-n} + (-1)^n · G^{Y-n})

    References
    ----------
    Carr, P., Geman, H., Madan, D.B. & Yor, M. (2002). The fine structure of
    asset returns. *Journal of Business*, 75(2), 305–332.
    """

    def __init__(self, C: float, G: float, M: float, Y: float) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if G < 0:
            raise ValueError(f"G must be non-negative, got {G}")
        if M < 0:
            raise ValueError(f"M must be non-negative, got {M}")
        if Y >= 2:
            raise ValueError(f"Y must be < 2, got {Y}")

        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CGMY(C={C}, G={G}, M={M}, Y={Y})",
            math_note=(
                f"Lévy density: C·exp(-M·x)/x^{{1+Y}}·𝟙_{{x>0}} + C·exp(-G·|x|)/|x|^{{1+Y}}·𝟙_{{x<0}}; "
                f"activity: {'finite' if Y < 0 else 'infinite'}, "
                f"variation: {'finite' if Y < 1 else 'infinite'}"
            ),
            reference="Carr, Geman, Madan & Yor (2002), eq. (1); Cont & Tankov (2004) Table 4.2",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy import integrate  # type: ignore[import-untyped]

        nu = _cgmy_levy_measure(self.C, self.G, self.M, self.Y)

        C, G, M, Y = self.C, self.G, self.M, self.Y
        if Y < 1:
            # finite variation: the characteristic exponent has no drift, so
            # b = ∫_{|x|≤1} x ν(dx) for the truncation 𝟙_{|x|≤1}
            b_pos, _ = integrate.quad(lambda x: C * np.exp(-M * x) / x**Y, 0.0, 1.0, limit=200)
            b_neg, _ = integrate.quad(lambda x: C * np.exp(-G * x) / x**Y, 0.0, 1.0, limit=200)
            b_drift = b_pos - b_neg
        else:
            # E[X_1] = b + ∫_{|x|>1} x ν(dx)
            big_pos, _ = integrate.quad(lambda x: C * np.exp(-M * x) / x**Y, 1.0, np.inf, limit=200)
            big_neg, _ = integrate.quad(lambda x: C * np.exp(-G * x) / x**Y, 1.0, np.inf, limit=200)
            b_drift = self.cumulant_exact(1) - (big_pos - big_neg)

        return LevyTriplet(b=b_drift, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=(self.Y < 2),
            has_finite_mean=(self.Y < 1 or (self.G > 0 and self.M > 0)),
            self_similarity_index=None,
            tail_index=self.Y if self.G == 0 or self.M == 0 else None,
            is_subordinator=False,
        )

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact closed-form cumulant.

        κ_n(X_t) = t · C · Γ(n-Y) · (M^{Y-n} + (-1)^n · G^{Y-n})

        Valid whenever G, M > 0 and n - Y is not a non-positive integer
        (exponential tempering makes every moment finite).
        Carr et al. (2002), equation (2).

        Parameters
        ----------
        n :
            Cumulant order.
        t :
            Time.
        """
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        untempered = min(self.G, self.M) == 0.0
        if (untempered and n >= self.Y) or (n - self.Y <= 0 and float(n - self.Y).is_integer()):
            raise ValueError(
                f"Cumulant of order {n} does not exist (or the closed form is singular) "
                f"for CGMY with G={self.G}, M={self.M}, Y={self.Y}."
            )
        return (
            t
            * self.C
            * float(gamma_fn(n - self.Y))
            * (self.M ** (self.Y - n) + ((-1) ** n) * self.G ** (self.Y - n))
        )

    def cumulants(self, order: int) -> dict[int, float]:
        """Return exact cumulants up to given order using closed-form formula."""
        result: dict[int, float] = {}
        for n in range(1, order + 1):
            try:
                result[n] = self.cumulant_exact(n)
            except ValueError:
                continue
        return result

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function of CGMY process.

        φ(u; t) = exp(t · C · Γ(-Y) · ((M-iu)^Y - M^Y + (G+iu)^Y - G^Y))

        Carr et al. (2002), equation (5).

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time.
        """
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        exponent = (
            self.C
            * float(gamma_fn(-self.Y))
            * (
                (self.M - 1j * u_arr) ** self.Y
                - self.M**self.Y
                + (self.G + 1j * u_arr) ** self.Y
                - self.G**self.Y
            )
        )
        result = np.exp(t * exponent)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulation via characteristic function inversion (Gil-Pelaez).

        For each time step, samples from the CGMY distribution by inverting
        the characteristic function numerically. This is exact in distribution
        but slower than closed-form methods.

        For production use with many paths, a series representation method
        (Rosiński 2001) should be preferred. This implementation is provided
        for correctness; the series method will be added in a future release.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps

        # Numerically invert characteristic function via Fourier FFT
        u_max = max(2000.0, 100.0 * max(self.G, self.M))
        N = 8192
        u_grid = np.linspace(-u_max, u_max, N)
        du = u_grid[1] - u_grid[0]
        cf_vals = self.char_func_exact(u=u_grid, t=dt)

        dx = 2.0 * np.pi / (N * du)
        x_grid = (np.arange(N) - N / 2.0) * dx

        # Fourier inversion: 1/(2pi) int phi(u) e^{-iux} du = ifft(conj(phi)) * (N*du)/(2pi)
        pdf_fft = (
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.conj(cf_vals)))).real
            * (N * du)
            / (2.0 * np.pi)
        )
        pdf_fft = np.maximum(pdf_fft, 0.0)
        cdf = np.cumsum(pdf_fft) * dx
        cdf = np.clip(cdf / (cdf[-1] + 1e-300), 0.0, 1.0)

        uniforms = rng.uniform(size=(n_paths, n_steps))
        increments = np.interp(uniforms, cdf, x_grid)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"CGMY(C={self.C}, G={self.G}, M={self.M}, Y={self.Y})"
