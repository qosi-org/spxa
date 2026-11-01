"""Model linear complementarity problems shared by the LCP solver tests."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


def model_lcp(n=60, convection=0.0, seed=0):
    """Implicit heat-equation step with a put-like obstacle (exercise on the left)."""
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    dt = 0.01
    main = np.full(n, 1.0 + 2.0 * dt / h**2)
    off_lo = np.full(n - 1, -dt / h**2 + convection * dt / (2 * h))
    off_up = np.full(n - 1, -dt / h**2 - convection * dt / (2 * h))
    M = sp.diags([off_lo, main, off_up], [-1, 0, 1], format="lil")
    M[0, :] = 0.0
    M[0, 0] = 1.0
    M[n - 1, :] = 0.0
    M[n - 1, n - 1] = 1.0
    M = M.tocsr()
    g = np.maximum(0.6 - x, 0.0)
    rng = np.random.default_rng(seed)
    u_prev = 0.9 * g + 0.03 * np.sin(np.pi * x) + 0.01 * rng.random(n)
    rhs = u_prev.copy()
    rhs[0] = g[0]
    rhs[-1] = 0.0
    free = np.ones(n, dtype=bool)
    free[[0, -1]] = False
    return M, rhs, g, u_prev, dt, free


def qp_solution(M, rhs, g, free):
    """Independent reference: for symmetric M the LCP is a bound-constrained QP."""
    A = M.toarray()
    n = rhs.size
    inner = np.where(free)[0]
    fixed = np.where(~free)[0]
    u_fixed = np.linalg.solve(A[np.ix_(fixed, fixed)], rhs[fixed])
    Ai = A[np.ix_(inner, inner)]
    bi = rhs[inner] - A[np.ix_(inner, fixed)] @ u_fixed

    def f(v):
        return 0.5 * v @ Ai @ v - bi @ v, Ai @ v - bi

    res = minimize(
        f,
        np.maximum(g[inner], 0.0),
        jac=True,
        method="L-BFGS-B",
        bounds=[(gi, None) for gi in g[inner]],
        options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 10000},
    )
    u = np.empty(n)
    u[inner] = res.x
    u[fixed] = u_fixed
    return u
