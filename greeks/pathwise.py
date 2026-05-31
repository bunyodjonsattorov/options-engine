"""Pathwise and likelihood-ratio Greeks for MC pricers.

Two ways to get a Greek out of MC without bumping:

PATHWISE (a.k.a. infinitesimal perturbation analysis):
    Swap d/dtheta and E. Differentiate the payoff w.r.t. theta along
    each path.  Works iff payoff is a.s. differentiable in theta (Lipschitz).
    FAILS for discontinuous payoffs (digital, barrier-touch) because
    d/dtheta of an indicator is a Dirac delta.

    For a European call on GBM:
        V(S_0) = e^{-rT} E[ max(S_T - K, 0) ]
        S_T = S_0 * exp((r-q-sigma^2/2) T + sigma sqrt(T) Z)
        => dS_T/dS_0 = S_T / S_0
    Pathwise delta estimator (one sample):
        Delta_i = e^{-rT} * I(S_T,i > K) * (S_T,i / S_0)

LIKELIHOOD RATIO (score function method):
    Differentiate the density f(x; theta) instead of the payoff.
        dV/dtheta = E[ payoff(X) * d log f / dtheta ]
    Works for any payoff (including discontinuous) but typically higher
    variance than pathwise when both apply.

    For GBM terminal price, S_T is log-normal with
        log S_T ~ N( log S_0 + (r-q-sigma^2/2)T,  sigma^2 T )
    Score for S_0:
        d log f / d S_0  = Z / (S_0 sigma sqrt(T))
    where Z is the standard normal draw that produced S_T. Hence:
        Delta_LR_i = e^{-rT} * payoff_i * Z_i / (S_0 sigma sqrt(T))

Compare: pathwise has lower variance for vanilla; LR is your only option
for digitals. Real production code often uses a mix (Malliavin calculus
generalises this).
"""

from __future__ import annotations

import numpy as np

from models.gbm import simulate_gbm_terminal


def _payoff_and_indicator(S_T, K, option):
    if option == "call":
        return np.maximum(S_T - K, 0.0), (S_T > K).astype(float)
    if option == "put":
        return np.maximum(K - S_T, 0.0), (S_T < K).astype(float)
    raise ValueError("option must be 'call' or 'put'")


def pathwise_delta_european(
    S0, K, T, r, sigma, q=0.0, option="call",
    n_paths=200_000, seed=None,
) -> tuple[float, float]:
    """Pathwise delta of European call/put. Returns (delta, stderr)."""
    rng = np.random.default_rng(seed)
    S_T, _ = simulate_gbm_terminal(S0, r, q, sigma, T, n_paths, rng=rng)
    _, indicator = _payoff_and_indicator(S_T, K, option)
    sign = 1.0 if option == "call" else -1.0
    sample = np.exp(-r * T) * sign * indicator * (S_T / S0)
    return float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n_paths))


def likelihood_ratio_delta_european(
    S0, K, T, r, sigma, q=0.0, option="call",
    n_paths=200_000, seed=None,
) -> tuple[float, float]:
    """Likelihood-ratio delta of European call/put. Returns (delta, stderr)."""
    rng = np.random.default_rng(seed)
    S_T, Z = simulate_gbm_terminal(S0, r, q, sigma, T, n_paths, rng=rng)
    payoff, _ = _payoff_and_indicator(S_T, K, option)
    score = Z / (S0 * sigma * np.sqrt(T))
    sample = np.exp(-r * T) * payoff * score
    return float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n_paths))


def likelihood_ratio_delta_digital(
    S0, K, T, r, sigma, q=0.0, option="call",
    n_paths=200_000, seed=None,
) -> tuple[float, float]:
    """Likelihood-ratio delta of a digital (cash-or-nothing) call/put.

    Digital payoff:  I(S_T > K)  (call) or I(S_T < K) (put), pays $1.
    Pathwise FAILS here (derivative of an indicator). LR is the standard
    fix — demonstrating you know the difference is a textbook interview win.
    """
    rng = np.random.default_rng(seed)
    S_T, Z = simulate_gbm_terminal(S0, r, q, sigma, T, n_paths, rng=rng)
    if option == "call":
        payoff = (S_T > K).astype(float)
    elif option == "put":
        payoff = (S_T < K).astype(float)
    else:
        raise ValueError("option must be 'call' or 'put'")
    score = Z / (S0 * sigma * np.sqrt(T))
    sample = np.exp(-r * T) * payoff * score
    return float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n_paths))
