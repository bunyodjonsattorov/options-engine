"""Longstaff-Schwartz Least-Squares Monte Carlo for American options.

The trick (2001 paper):
    At each exercise date, working backwards, we have a sample of paths
    and we know the CASH-FLOW (if we follow the optimal policy from this
    date onwards) for each path. We want to compare immediate-exercise
    value against the EXPECTED continuation value given S_t.

    Estimate that conditional expectation by REGRESSING the discounted
    future cash flow on a basis of functions of S_t, using only the
    in-the-money paths (the rest don't exercise anyway and including them
    just adds noise / biases the regression).

    Then for each ITM path: if exercise > regressed continuation, exercise
    now and overwrite cash flow with exercise value at this time.

Bias considerations (good interview material):
    * LSM gives a LOWER bound on the true American price: we're using an
      estimated, suboptimal stopping rule.
    * Upper bound via Andersen-Broadie duality exists (not implemented).
    * Variance scales with number of basis functions; too many => overfit
      the regression on each step => noisy continuation estimate.
    * Standard basis: 1, S, S^2 (polynomial) or Laguerre polynomials.

References:
    Longstaff & Schwartz, "Valuing American Options by Simulation: A
    Simple Least-Squares Approach", Review of Financial Studies, 2001.
"""

from __future__ import annotations

import numpy as np

from models.gbm import simulate_gbm_paths


def _basis(S: np.ndarray, kind: str = "poly") -> np.ndarray:
    """Return regression design matrix of shape (n_paths, n_basis)."""
    if kind == "poly":
        return np.column_stack([np.ones_like(S), S, S * S])
    if kind == "laguerre":
        # First three Laguerre polynomials evaluated at S/100 (scaling
        # matters numerically — bare S^2 with S~100 produces a huge
        # condition number).
        x = S / 100.0
        L0 = np.ones_like(x)
        L1 = 1.0 - x
        L2 = 1.0 - 2.0 * x + 0.5 * x * x
        return np.column_stack([L0, L1, L2])
    raise ValueError("kind must be 'poly' or 'laguerre'")


def lsm_american_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 50,
    basis: str = "poly",
    seed: int | None = None,
) -> tuple[float, float]:
    """Longstaff-Schwartz price of an American put. Returns (price, stderr).

    Note on stderr: this is the stderr OF THE LSM ESTIMATOR (variance
    across paths), not the bias from suboptimal policy. The true
    American price is >= reported price (within stderr).
    """
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_paths, n_steps, rng=rng)
    dt = T / n_steps
    disc_step = np.exp(-r * dt)

    # Cash flow per path if we exercise at the OPTIMAL time we've found
    # so far. Initialise at terminal payoff (held-to-expiry policy).
    payoff = np.maximum(K - paths[:, -1], 0.0)
    # Time-step index of the cash flow (used to discount back when needed)
    exercise_step = np.full(n_paths, n_steps, dtype=int)

    # Walk backwards from step n-1 to 1; step 0 is decision today (no
    # regression — we just compare to today's continuation value).
    for step in range(n_steps - 1, 0, -1):
        S = paths[:, step]
        itm = K > S        # American put: only ITM paths exercise
        if not itm.any():
            continue

        # Discount each path's current cash flow back to THIS step.
        steps_to_cf = exercise_step[itm] - step
        disc_cf = payoff[itm] * (disc_step ** steps_to_cf)

        X = _basis(S[itm], kind=basis)
        # Least-squares fit of continuation value
        coef, *_ = np.linalg.lstsq(X, disc_cf, rcond=None)
        continuation = X @ coef

        exercise_value = K - S[itm]
        # Where exercise beats continuation, take the exercise
        exercise_now = exercise_value > continuation
        # Update cash flow vector
        idx = np.where(itm)[0][exercise_now]
        payoff[idx] = exercise_value[exercise_now]
        exercise_step[idx] = step

    # Discount each path's cash flow back to t=0 and average
    disc_to_0 = payoff * np.exp(-r * dt * exercise_step)
    price = float(disc_to_0.mean())
    stderr = float(disc_to_0.std(ddof=1) / np.sqrt(n_paths))
    return price, stderr
