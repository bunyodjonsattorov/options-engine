"""Geometric Brownian Motion simulators (the BS world).

Risk-neutral dynamics:
    dS_t = (r - q) S_t dt + sigma S_t dW_t^Q

Exact discretisation (works because GBM has a closed-form solution):
    S_{t+dt} = S_t * exp( (r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z )
with Z ~ N(0, 1) i.i.d. across (path, time-step).

Why exact (not Euler) by default:
    Euler on GBM (S_{t+dt} = S_t + (r-q) S_t dt + sigma S_t sqrt(dt) Z)
    introduces a discretisation bias of order dt and CAN go negative.
    Log-Euler / exact is unbiased and stays positive — free wins, use it.

We expose two entry points:
    * simulate_gbm_terminal: only S_T  (for path-independent payoffs).
      Single jump => no per-step loop, fastest.
    * simulate_gbm_paths: full path  (for Asian / barrier / lookback /
      American-via-LSM).
"""

from __future__ import annotations

import numpy as np


def simulate_gbm_terminal(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    antithetic: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw S_T for n_paths terminal samples.

    Returns (S_T, Z) — S_T is the terminal prices, Z is the underlying
    standard-normal draws (kept for likelihood-ratio Greeks).

    If antithetic=True, the second half of paths uses -Z of the first
    half. n_paths must then be even.
    """
    if rng is None:
        rng = np.random.default_rng()

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        Z_half = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)
    return S_T, Z


def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    *,
    antithetic: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate full GBM paths, shape (n_paths, n_steps + 1).

    paths[:, 0] = S0 for all paths.
    Vectorised: one big draw of shape (n_paths, n_steps), then cumprod.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        Z_half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion
    log_S = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_S)
    return paths
