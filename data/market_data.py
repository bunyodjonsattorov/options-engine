"""Pull option chain data — live (yfinance) or synthetic (for offline dev).

Real-world note (interview answer): option chain mid-prices look clean
but are often stale. NBBO bid/ask spreads are wide for far-OTM strikes
and short-dated options. Best practice:
    * Filter out zero-bid quotes.
    * Filter out quotes where (bid + ask)/2 < intrinsic (arbitrage relic
      of stale quotes).
    * Use the mid as your "market price" but track the half-spread for
      weighting in calibration (cheap quotes get less weight).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptionQuote:
    """One row of an option chain after cleaning."""
    expiry: pd.Timestamp
    T: float                  # time-to-expiry in years
    strike: float
    option: str               # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    spot: float
    r: float                  # risk-free for this maturity (we use one flat r)
    q: float                  # dividend yield


def get_option_chain(
    ticker: str = "SPY",
    expiries: int | None = 4,
    r: float = 0.05,
    q: float = 0.015,
) -> pd.DataFrame:
    """Pull live option chain from yfinance. Falls back to a clear error.

    Returns a tidy DataFrame with columns:
        expiry, T, strike, option, bid, ask, mid, spot, r, q.

    Parameters
    ----------
    ticker : underlying symbol.
    expiries : take the first N expiry dates (None = all).
    r : flat risk-free rate (could be a curve in production).
    q : flat dividend yield.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance not installed. `pip install yfinance` to use this, "
            "or call synthetic_chain() for an offline test chain."
        ) from e

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)

    exp_list = list(tk.options)
    if expiries is not None:
        exp_list = exp_list[:expiries]

    rows: list[dict] = []
    for exp in exp_list:
        expiry_ts = pd.Timestamp(exp)
        T = max((expiry_ts - today).days / 365.0, 1.0 / 365.0)
        chain = tk.option_chain(exp)
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df[(df.bid > 0) & (df.ask > 0)].copy()
            df["mid"] = 0.5 * (df.bid + df.ask)
            for _, row in df.iterrows():
                rows.append(dict(
                    expiry=expiry_ts, T=T, strike=float(row.strike),
                    option=opt_type, bid=float(row.bid), ask=float(row.ask),
                    mid=float(row.mid), spot=spot, r=r, q=q,
                ))
    return pd.DataFrame(rows)


def synthetic_chain(
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    expiries_years: tuple[float, ...] = (1 / 12, 3 / 12, 6 / 12, 1.0),
    n_strikes: int = 21,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic chain with a realistic equity-style smile.

    Vol surface used:
        sigma_imp(k, T) = sigma_atm(T) * (1 - skew*k + curv*k^2)
        sigma_atm(T)    = 0.18 + 0.04 / sqrt(T)   (short-dated higher)
        skew, curv      = small positive (downward skew, smile curvature)

    Then we add a tiny ($0.01) bid/ask spread around the BS price so it
    looks like an actual cleaned chain.
    """
    from pricers.black_scholes import bs_price
    rng = np.random.default_rng(seed)
    strikes = spot * np.linspace(*moneyness_range, n_strikes)

    rows = []
    today = pd.Timestamp("2026-05-22")
    for T in expiries_years:
        sigma_atm = 0.18 + 0.04 / np.sqrt(T)
        F = spot * np.exp((r - q) * T)
        for K in strikes:
            k = np.log(K / F)
            sigma = sigma_atm * (1.0 - 0.35 * k + 0.6 * k * k)
            sigma = float(max(0.05, sigma))
            for opt in ("call", "put"):
                mid = float(bs_price(spot, K, T, r, sigma, q, opt))
                # tiny bid/ask spread (1-3 cents typical for liquid SPY)
                spread = max(0.01, 0.002 * mid)
                bid, ask = mid - spread / 2, mid + spread / 2
                rows.append(dict(
                    expiry=today + pd.Timedelta(days=int(T * 365)),
                    T=T, strike=float(K), option=opt,
                    bid=bid, ask=ask, mid=mid, spot=spot, r=r, q=q,
                ))
    return pd.DataFrame(rows)
