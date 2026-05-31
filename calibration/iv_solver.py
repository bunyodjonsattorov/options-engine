"""Implied volatility: invert BS for sigma given an observed price.

Method: Brent's method (scipy.optimize.brentq) — combines bisection
(guaranteed convergence) with secant/inverse quadratic for speedup.

Subtleties (good interview answer if asked "how would you compute IV?"):
    1. No-arb bounds: option price must be inside [intrinsic, S e^{-qT}]
       (call) or [intrinsic, K e^{-rT}] (put). Outside => no real IV
       exists; we return NaN rather than raise (one bad quote shouldn't
       kill a chain scan).
    2. Initial bracket: [1e-6, 5.0] vol covers any realistic case.
    3. For very deep ITM/OTM, vega -> 0 and Brent can be slow / unstable.
       Filter |delta| in (0.05, 0.95) before fitting to a smile.
    4. American options: BS-implied vol of an American option is biased
       (vol that fits a European BS to an American price). For short-
       dated ATM equity options the error is small. For deep ITM American
       puts with dividends — large. We accept the bias here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pricers.black_scholes import bs_price


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option: str = "call",
    *,
    tol: float = 1e-8,
    sigma_range: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Return sigma_imp such that bs_price(...) ~ price. NaN if no IV."""
    if T <= 0:
        return float("nan")
    intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        if option == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    )
    upper = (
        S * np.exp(-q * T) if option == "call" else K * np.exp(-r * T)
    )
    if not (intrinsic - 1e-10 < price < upper + 1e-10):
        return float("nan")

    def f(sigma):
        return float(bs_price(S, K, T, r, sigma, q, option)) - price

    f_lo = f(sigma_range[0])
    f_hi = f(sigma_range[1])
    if f_lo * f_hi > 0:
        return float("nan")
    try:
        return float(brentq(f, sigma_range[0], sigma_range[1], xtol=tol))
    except (ValueError, RuntimeError):
        return float("nan")


def add_iv_column(chain: pd.DataFrame, price_col: str = "mid") -> pd.DataFrame:
    """Append an 'iv' column to a chain DataFrame (NaN where no IV exists)."""
    out = chain.copy()
    ivs = []
    for row in chain.itertuples(index=False):
        ivs.append(implied_vol(
            getattr(row, price_col), row.spot, row.strike, row.T,
            row.r, row.q, row.option,
        ))
    out["iv"] = ivs
    return out
