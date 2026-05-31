"""Cox-Ross-Rubinstein binomial tree (European + American).

Setup:
    Time horizon T split into n steps of dt = T/n.
    At each step the price either goes up by factor u or down by d = 1/u.
        u = exp(sigma sqrt(dt))
        d = 1/u
    Risk-neutral up-probability (matches first moment of GBM):
        p = (exp((r - q) dt) - d) / (u - d)
    Discount factor per step: exp(-r dt).

Why this works:
    As n -> infinity the tree's terminal distribution converges to the
    lognormal of GBM (CLT on log returns) and the prices converge to BS.
    Convergence is oscillatory (odd vs even n) — a textbook fix is to
    average pairs (n, n+1) or use the leisen-reimer scheme. We keep
    plain CRR for clarity.

Why a tree at all when we have BS?
    American options: at each node we can compare continuation value
    against immediate exercise value, picking the max. BS can't do that.
"""

from __future__ import annotations

import numpy as np


def _crr_setup(T, r, q, sigma, n_steps):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral prob {p} out of (0,1); "
            "try smaller dt (more steps) or check parameters."
        )
    disc = np.exp(-r * dt)
    return dt, u, d, p, disc


def binomial_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    exercise: str = "european",
    n_steps: int = 500,
) -> float:
    """Vectorised CRR price.

    Parameters
    ----------
    exercise : 'european' or 'american'

    Implementation note (memory-efficient backward induction):
        We never store the whole tree (would be O(n^2)). At each step k,
        the option values live on a vector of length k+1 — we update it
        in place going backwards.
    """
    dt, u, d, p, disc = _crr_setup(T, r, q, sigma, n_steps)
    exercise = exercise.lower()
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")

    # Terminal asset prices S_n[j] = S0 * u^(n - j) * d^j, j = 0..n
    j = np.arange(n_steps + 1)
    S_T = S0 * (u ** (n_steps - j)) * (d ** j)

    # Terminal payoffs
    if option == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        # Continuation value at each node of this step
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if exercise == "american":
            # Asset prices at this layer
            j = np.arange(step + 1)
            S = S0 * (u ** (step - j)) * (d ** j)
            intrinsic = (S - K) if option == "call" else (K - S)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def binomial_american_early_exercise_boundary(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "put",
    n_steps: int = 500,
) -> np.ndarray:
    """Extract the early-exercise boundary S*(t) for an American option.

    Returns an array of length n_steps+1: at time-step k (i.e. time k*dt),
    S_boundary[k] is the asset price below which it's optimal to exercise
    immediately (for a put). NaN where the boundary is undefined (no
    exercise region at that step).

    Useful for plotting and for intuition: shows why American puts are
    worth more than European puts (the boundary cuts off bad scenarios).
    """
    dt, u, d, p, disc = _crr_setup(T, r, q, sigma, n_steps)
    j = np.arange(n_steps + 1)
    S_T = S0 * (u ** (n_steps - j)) * (d ** j)
    intrinsic_T = (S_T - K) if option == "call" else (K - S_T)
    V = np.maximum(intrinsic_T, 0.0)
    boundary = np.full(n_steps + 1, np.nan)

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        j = np.arange(step + 1)
        S = S0 * (u ** (step - j)) * (d ** j)
        intrinsic = (S - K) if option == "call" else (K - S)
        exercise_now = intrinsic >= V
        V = np.where(exercise_now, intrinsic, V)
        if exercise_now.any():
            # For an American put, exercise region is S <= S*; pick the
            # largest S in the exercise region.
            if option == "put":
                boundary[step] = S[exercise_now].max()
            else:
                boundary[step] = S[exercise_now].min()

    return boundary
