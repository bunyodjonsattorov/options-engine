"""Finite-difference Greeks — model-agnostic, used to verify analytical Greeks.

Idea: any Greek is a partial derivative of price. Approximate it by
bumping the input and revaluing. Central differences have O(h^2) error
vs O(h) for forward differences — slightly more expensive (2 evals
instead of 1) but much more accurate, so we use them by default.

Tradeoff to know for interview:
    * Bump size h: too small => numerical noise dominates; too large =>
      truncation error dominates. Rule of thumb: h ~ x * sqrt(eps_machine)
      for first derivatives, h ~ x * eps_machine^(1/3) for second.
    * For MC pricers, bumping with a *fresh* set of random numbers each
      eval is a disaster (the MC noise swamps the small price difference).
      Common-random-numbers (reuse the same draws across S, S+h, S-h) is
      mandatory. Even better: pathwise / likelihood-ratio Greeks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def fd_greeks(
    price_fn: Callable[..., float],
    S: float,
    sigma: float,
    T: float,
    r: float,
    *,
    h_S: float = 1e-2,
    h_sigma: float = 1e-4,
    h_T: float = 1.0 / 365.0,
    h_r: float = 1e-4,
    **kwargs,
) -> dict[str, float]:
    """Central-difference Greeks for any pricer with signature
    ``price_fn(S, K, T, r, sigma, q=..., option=...)``.

    Bump sizes default to values that work well for typical equity option
    parameters (S ~ 100, sigma ~ 0.2). Override per call if you're pricing
    very different scales (e.g. FX, rates).
    """
    def f(**overrides):
        args = dict(S=S, sigma=sigma, T=T, r=r, **kwargs)
        args.update(overrides)
        return float(price_fn(**args))

    # Delta = dV/dS
    delta = (f(S=S + h_S) - f(S=S - h_S)) / (2 * h_S)
    # Gamma = d2V/dS2
    gamma = (f(S=S + h_S) - 2 * f() + f(S=S - h_S)) / (h_S**2)
    # Vega = dV/dsigma
    vega = (f(sigma=sigma + h_sigma) - f(sigma=sigma - h_sigma)) / (2 * h_sigma)
    # Theta = dV/dt = -dV/dT  (convention: theta is wrt calendar time)
    theta = -(f(T=T + h_T) - f(T=T - h_T)) / (2 * h_T)
    # Rho = dV/dr
    rho = (f(r=r + h_r) - f(r=r - h_r)) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
