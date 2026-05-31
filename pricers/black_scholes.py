"""Black-Scholes-Merton closed-form pricing and analytical Greeks.

Assumptions (all break in reality — know them cold for interviews):
    1. Underlying S_t follows geometric Brownian motion under the
       risk-neutral measure Q:
            dS_t = (r - q) S_t dt + sigma S_t dW_t^Q
       => constant drift, constant vol.
    2. No arbitrage, frictionless market (no taxes / transaction costs).
    3. Continuous trading, continuous hedging possible.
    4. Constant risk-free rate r and constant dividend yield q.
    5. Log-returns are normal => terminal price S_T is log-normal.
    6. European exercise only (no early exercise — pricing American with
       this formula is wrong; see pricers.binomial / longstaff_schwartz).

Pricing formula:
    Under Q, an option's price today is the discounted expected payoff:
        V_0 = exp(-rT) * E^Q[ payoff(S_T) ]
    For a call payoff max(S_T - K, 0) under log-normal S_T this integral
    has a closed form:
        C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)        (by put-call parity)
    where
        d1 = [ ln(S/K) + (r - q + 0.5 sigma^2) T ] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)
        N(.) = standard normal CDF.

Put-call parity (model-free; only assumes no-arbitrage + same K, T):
        C - P = S e^{-qT} - K e^{-rT}
We use this both as a sanity test and (later) to convert between call IVs
and put IVs in the calibration module.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (d1, d2) used in every BS formula.

    Vectorised over S, K, T, sigma so the same code prices a whole option
    chain at once.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Avoid division-by-zero at expiry or zero vol; we handle the limit
    # explicitly in bs_price/bs_greeks rather than letting NaNs propagate.
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    vol_sqrtT = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrtT
        d2 = d1 - vol_sqrtT
    return d1, d2


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option: str = "call",
) -> np.ndarray:
    """Black-Scholes European call/put price.

    Parameters
    ----------
    S : spot price.
    K : strike.
    T : time to expiry in years (e.g. 30/365 for a one-month option).
    r : continuously compounded risk-free rate.
    sigma : volatility (annualised, in decimal — 0.20 means 20%).
    q : continuous dividend yield.
    option : 'call' or 'put'.

    At-expiry behaviour: returns intrinsic value max(S-K, 0) or max(K-S, 0).
    """
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    # Intrinsic at/after expiry — bypass d1/d2 to avoid 0/0.
    intrinsic = np.where(
        option == "call",
        np.maximum(S_arr - K_arr, 0.0),
        np.maximum(K_arr - S_arr, 0.0),
    )

    d1, d2 = d1_d2(S_arr, K_arr, T_arr, r, sigma, q)
    disc_K = K_arr * np.exp(-r * T_arr)
    disc_S = S_arr * np.exp(-q * T_arr)
    if option == "call":
        price = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        price = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)

    return np.where(T_arr <= 0.0, intrinsic, price)


def bs_greeks(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option: str = "call",
) -> dict[str, np.ndarray]:
    """First-order Greeks (Delta, Gamma, Vega, Theta, Rho).

    Conventions used here:
        * Vega is per unit vol (NOT per vol point). Multiply by 0.01 to
          get the price change per 1% vol move. We keep the raw form so
          composition with other formulas is clean.
        * Theta is per year. Divide by 365 for a calendar-day theta.
        * Greeks for a put are derived from the call Greeks using parity;
          Gamma and Vega are identical for calls and puts.

    Why know these by heart:
        Delta = dV/dS    -- hedge ratio (shares of stock per option)
        Gamma = d2V/dS2  -- how much delta moves; long gamma => convexity
        Vega  = dV/dsigma -- exposure to vol level
        Theta = dV/dt    -- time decay (negative for long options usually)
        Rho   = dV/dr    -- rate sensitivity (small for short-dated equity opt)
    """
    option = option.lower()
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    pdf_d1 = norm.pdf(d1)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    # Gamma & Vega — identical for call and put.
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    elif option == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def put_call_parity_residual(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Return C - P - (S e^{-qT} - K e^{-rT}).

    Should be ~0 for arbitrage-free prices. Used in tests and as a
    sanity check on observed market quotes (a large residual usually
    means stale quotes, not free money).
    """
    return call_price - put_price - (S * np.exp(-q * T) - K * np.exp(-r * T))
