"""Short-straddle delta-hedge simulation + full PnL decomposition.

Strategy:
    At t=0 we SELL one ATM call and one ATM put expiring at T (a short
    straddle). We collect the premium. Then daily we re-hedge: compute
    the total option-position delta and trade enough stock to keep the
    portfolio delta-flat. At expiry we settle the options at intrinsic.

The trade thesis (Jane Street will probe this):
    sigma_implied at the sell is the *market's expected realized vol*.
    If realized_vol < implied_vol, the straddle was overpriced and we
    profit. If realized > implied, we lose. Delta-hedging strips out
    direction so we isolate the vol bet. Cleanly.

PnL decomposition (one-step, continuous time approximation):
    Position = -V (short option) + Delta * S (hedge)  (+ cash)
    From Ito + BS PDE, for a delta-hedged position the PnL between
    rebalances is approximately:

        dPnL_hedge  ~  -0.5 * Gamma_pos * [ dS^2  -  sigma_imp^2 * S^2 * dt ]
                       - Vega_pos * d(sigma_imp)
                       - transaction costs

    where Gamma_pos and Vega_pos are the position-level Greeks (negative
    for a short straddle => gamma PnL is negative when stock moves
    *more* than implied, positive when it stays still).
    Theta is encoded in the relation above: BS PDE links Theta to Gamma.

We compute three contributions explicitly:
    * gamma_pnl  = -0.5 * Gamma_pos * (dS_actual^2 - sigma_imp^2 * S^2 * dt)
    * vega_pnl   = -Vega_pos * d(sigma_imp)   (zero if IV held constant)
    * tx_costs   = -|d(shares)| * S * tc_rate
    * residual   = total - sum(above)  (captures higher-order terms and
                   the discrete-hedge gap)

We also report the gamma-implied vs realized vol gap, which is the
*economic* explanation of the PnL.

References:
    Wilmott, "Paul Wilmott on Quantitative Finance", Ch. 7 (delta hedging).
    Bouchaud & Potters, "Theory of Financial Risk and Derivative Pricing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pricers.black_scholes import bs_price, bs_greeks
from models.gbm import simulate_gbm_paths


@dataclass
class DeltaHedgeConfig:
    S0: float = 100.0
    K: float | None = None         # default ATM at start
    T: float = 30 / 365            # 30 calendar days
    r: float = 0.05
    q: float = 0.0
    sigma_imp: float = 0.20        # implied vol used to price + hedge
    sigma_real: float = 0.20       # realized vol of the simulated path
    n_steps: int = 30              # rebalance frequency (steps within T)
    tc_per_share: float = 0.0      # $ cost per share traded
    bid_ask_bps: float = 0.0       # half-spread on the stock in basis points
    contract_size: float = 1.0     # one straddle (1 call + 1 put)


@dataclass
class DeltaHedgeResult:
    times: np.ndarray
    spot: np.ndarray
    call_price: np.ndarray
    put_price: np.ndarray
    position_value: np.ndarray     # full mark-to-market PnL series, cumulative
    gamma_pnl: np.ndarray
    vega_pnl: np.ndarray
    tx_costs: np.ndarray
    delta_position: np.ndarray
    shares_held: np.ndarray
    cash: np.ndarray
    final_pnl: float = 0.0
    pnl_decomposition: dict = field(default_factory=dict)


def _straddle_price(S, K, T, r, sigma, q):
    """Total ATM call+put premium (one of each)."""
    c = float(bs_price(S, K, T, r, sigma, q, "call"))
    p = float(bs_price(S, K, T, r, sigma, q, "put"))
    return c, p


def _straddle_greeks(S, K, T, r, sigma, q):
    g_c = bs_greeks(S, K, T, r, sigma, q, "call")
    g_p = bs_greeks(S, K, T, r, sigma, q, "put")
    # Position-level (a SHORT straddle => multiply by -1 outside this fn)
    return {name: float(g_c[name] + g_p[name]) for name in g_c}


def simulate_short_straddle_hedged(
    cfg: DeltaHedgeConfig,
    *,
    path: np.ndarray | None = None,
    rng_seed: int | None = None,
    sigma_imp_path: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> DeltaHedgeResult:
    """Run one path of the short-straddle delta-hedge strategy.

    Parameters
    ----------
    path : optional, shape (n_steps+1,). If None we simulate GBM with
        sigma_real. Pass a real historical SPY path to backtest on data.
    sigma_imp_path : optional callable (t, S) -> sigma_imp. Default holds
        IV constant at cfg.sigma_imp (no vol-of-vol). Override to bring
        in a stochastic IV process and exercise Vega PnL.
    """
    K = cfg.S0 if cfg.K is None else cfg.K
    dt = cfg.T / cfg.n_steps

    if path is None:
        rng = np.random.default_rng(rng_seed)
        path = simulate_gbm_paths(
            cfg.S0, cfg.r, cfg.q, cfg.sigma_real, cfg.T, 1, cfg.n_steps, rng=rng,
        )[0]
    assert path.shape == (cfg.n_steps + 1,)

    times = np.linspace(0.0, cfg.T, cfg.n_steps + 1)

    # IV at each step (default flat)
    if sigma_imp_path is None:
        iv = np.full(cfg.n_steps + 1, cfg.sigma_imp)
    else:
        iv = np.asarray(sigma_imp_path(times, path), dtype=float)

    # Allocate result arrays
    call_p = np.zeros(cfg.n_steps + 1)
    put_p = np.zeros(cfg.n_steps + 1)
    delta_pos = np.zeros(cfg.n_steps + 1)
    shares = np.zeros(cfg.n_steps + 1)
    cash = np.zeros(cfg.n_steps + 1)
    pos_val = np.zeros(cfg.n_steps + 1)
    gamma_pnl_step = np.zeros(cfg.n_steps + 1)
    vega_pnl_step = np.zeros(cfg.n_steps + 1)
    tx_cost_step = np.zeros(cfg.n_steps + 1)

    # ---- t = 0: sell the straddle, hedge to delta-neutral ----
    tau = cfg.T
    c0, p0 = _straddle_price(path[0], K, tau, cfg.r, iv[0], cfg.q)
    premium = (c0 + p0) * cfg.contract_size
    call_p[0], put_p[0] = c0, p0

    g0 = _straddle_greeks(path[0], K, tau, cfg.r, iv[0], cfg.q)
    short_delta = -g0["delta"] * cfg.contract_size  # we are short the straddle
    # Hedge: buy/sell stock so total delta (short_delta + shares) = 0
    shares_to_buy = -short_delta
    shares[0] = shares_to_buy
    tx0 = abs(shares_to_buy) * path[0] * (cfg.bid_ask_bps / 10_000.0) \
          + abs(shares_to_buy) * cfg.tc_per_share
    cash[0] = premium - shares_to_buy * path[0] - tx0
    tx_cost_step[0] = -tx0
    delta_pos[0] = short_delta + shares[0]  # ~0 by construction

    # ---- step loop ----
    for i in range(1, cfg.n_steps + 1):
        tau = cfg.T - times[i]
        S_prev, S_now = path[i - 1], path[i]
        dS = S_now - S_prev

        # gamma PnL of the option position (we are SHORT => negate)
        gp = _straddle_greeks(S_prev, K, cfg.T - times[i - 1], cfg.r, iv[i - 1], cfg.q)
        gamma_pos = -gp["gamma"] * cfg.contract_size
        vega_pos = -gp["vega"] * cfg.contract_size

        # PnL decomposition (theoretical contributions over [t_{i-1}, t_i])
        gamma_pnl_step[i] = 0.5 * gamma_pos * (
            dS**2 - (iv[i - 1] ** 2) * (S_prev**2) * dt
        )
        vega_pnl_step[i] = vega_pos * (iv[i] - iv[i - 1])

        # Cash accrues at r over the step (continuous compounding)
        cash[i] = cash[i - 1] * np.exp(cfg.r * dt)

        # Re-price options at the new spot, time, and (possibly new) IV
        if tau <= 1e-10:
            # at-expiry settle: intrinsic
            call_p[i] = max(S_now - K, 0.0) * cfg.contract_size
            put_p[i] = max(K - S_now, 0.0) * cfg.contract_size
            # short position pays intrinsic out of cash
            cash[i] -= (call_p[i] + put_p[i])
            # liquidate hedge
            cash[i] += shares[i - 1] * S_now \
                       - abs(shares[i - 1]) * S_now * (cfg.bid_ask_bps / 10_000.0)
            tx_cost_step[i] = -abs(shares[i - 1]) * S_now * (cfg.bid_ask_bps / 10_000.0)
            shares[i] = 0.0
            delta_pos[i] = 0.0
        else:
            c, p = _straddle_price(S_now, K, tau, cfg.r, iv[i], cfg.q)
            call_p[i], put_p[i] = c * cfg.contract_size, p * cfg.contract_size
            # New delta target
            g = _straddle_greeks(S_now, K, tau, cfg.r, iv[i], cfg.q)
            short_delta_new = -g["delta"] * cfg.contract_size
            target_shares = -short_delta_new
            d_shares = target_shares - shares[i - 1]
            tx = abs(d_shares) * S_now * (cfg.bid_ask_bps / 10_000.0) \
                 + abs(d_shares) * cfg.tc_per_share
            shares[i] = target_shares
            cash[i] += -d_shares * S_now - tx
            tx_cost_step[i] = -tx
            delta_pos[i] = short_delta_new + shares[i]

        # Mark-to-market position value.
        # After expiry settlement, everything has converted to cash
        # (intrinsic was paid out, hedge was liquidated), so pos_val is
        # just the cash balance — using the MTM formula here would
        # double-count the already-paid intrinsic.
        if tau <= 1e-10:
            pos_val[i] = cash[i]
        else:
            liability = call_p[i] + put_p[i]  # what we owe (MTM value)
            pos_val[i] = cash[i] + shares[i] * S_now - liability

    # PnL relative to initial cash injection (which is 0 — we got premium
    # then bought hedge with it). Starting position value at t=0:
    liab0 = call_p[0] + put_p[0]
    pos_val[0] = cash[0] + shares[0] * path[0] - liab0

    final_pnl = float(pos_val[-1] - pos_val[0])

    decomp = {
        "gamma_pnl_total": float(gamma_pnl_step.sum()),
        "vega_pnl_total": float(vega_pnl_step.sum()),
        "tx_costs_total": float(tx_cost_step.sum()),
        "residual": float(
            final_pnl - gamma_pnl_step.sum() - vega_pnl_step.sum() - tx_cost_step.sum()
        ),
    }

    return DeltaHedgeResult(
        times=times, spot=path,
        call_price=call_p, put_price=put_p,
        position_value=pos_val,
        gamma_pnl=gamma_pnl_step, vega_pnl=vega_pnl_step, tx_costs=tx_cost_step,
        delta_position=delta_pos, shares_held=shares, cash=cash,
        final_pnl=final_pnl, pnl_decomposition=decomp,
    )


def summarise_runs(pnls: np.ndarray) -> dict:
    """Distribution statistics over many simulation runs."""
    pnls = np.asarray(pnls)
    return {
        "mean": float(pnls.mean()),
        "std": float(pnls.std(ddof=1)),
        "median": float(np.median(pnls)),
        "p5": float(np.percentile(pnls, 5)),
        "p95": float(np.percentile(pnls, 95)),
        "min": float(pnls.min()),
        "max": float(pnls.max()),
        "sharpe_per_run": float(pnls.mean() / pnls.std(ddof=1)) if pnls.std(ddof=1) > 0 else float("nan"),
        "win_rate": float((pnls > 0).mean()),
        "n_runs": int(pnls.size),
    }
