"""Cross-validation tests across pricers + parity checks.

Why these tests matter (not just "we have tests"):
    * put_call_parity: model-free, must hold to numerical precision.
    * bs_vs_fd: analytical Greeks must match finite-difference Greeks
      (catches sign errors and dropped terms in the closed-form).
    * mc_converges_to_bs: a sanity check on the GBM simulator AND on
      the BS pricer. Both wrong in the same direction would be very
      unlucky.
    * binomial_converges_to_bs: ditto for the tree.
"""

from __future__ import annotations

import numpy as np

from pricers.black_scholes import (
    bs_price, bs_greeks, put_call_parity_residual,
)
from greeks.finite_difference import fd_greeks


# ---- baseline params (ATM, 3-month, 20% vol, 5% rate) ----
S0, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.05, 0.20, 0.00


def test_put_call_parity():
    c = float(bs_price(S0, K, T, r, sigma, q, "call"))
    p = float(bs_price(S0, K, T, r, sigma, q, "put"))
    residual = put_call_parity_residual(c, p, S0, K, T, r, q)
    assert abs(residual) < 1e-10, f"parity residual = {residual}"


def test_intrinsic_at_expiry():
    # ITM call at T=0 should equal S - K, OTM should equal 0.
    assert float(bs_price(120, 100, 0.0, r, sigma, q, "call")) == 20.0
    assert float(bs_price(80, 100, 0.0, r, sigma, q, "call")) == 0.0
    assert float(bs_price(80, 100, 0.0, r, sigma, q, "put")) == 20.0


def test_analytical_vs_fd_greeks_call():
    g_an = bs_greeks(S0, K, T, r, sigma, q, "call")
    g_fd = fd_greeks(bs_price, S=S0, sigma=sigma, T=T, r=r, K=K, q=q, option="call")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert np.isclose(g_an[name], g_fd[name], rtol=1e-3, atol=1e-4), (
            f"{name}: analytic={g_an[name]:.6f}, fd={g_fd[name]:.6f}"
        )


def test_analytical_vs_fd_greeks_put():
    g_an = bs_greeks(S0, K, T, r, sigma, q, "put")
    g_fd = fd_greeks(bs_price, S=S0, sigma=sigma, T=T, r=r, K=K, q=q, option="put")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert np.isclose(g_an[name], g_fd[name], rtol=1e-3, atol=1e-4), (
            f"{name}: analytic={g_an[name]:.6f}, fd={g_fd[name]:.6f}"
        )


def test_call_delta_bounds():
    # Deep ITM call delta -> 1 (or e^{-qT} with dividends); deep OTM -> 0.
    deep_itm = float(bs_greeks(1000, 100, T, r, sigma, q, "call")["delta"])
    deep_otm = float(bs_greeks(10, 100, T, r, sigma, q, "call")["delta"])
    assert deep_itm > 0.99
    assert deep_otm < 0.01


def test_vectorised_pricing():
    # Vectorisation: a single call should price 5 strikes simultaneously.
    strikes = np.array([80, 90, 100, 110, 120])
    prices = bs_price(S0, strikes, T, r, sigma, q, "call")
    assert prices.shape == (5,)
    # Monotonic decreasing in strike for a call.
    assert np.all(np.diff(prices) < 0)


# ---- Phase 2: Monte Carlo ----
def test_mc_european_within_3se_of_bs():
    from pricers.monte_carlo import mc_european
    bs = float(bs_price(S0, K, T, r, sigma, q, "call"))
    p, se = mc_european(S0, K, T, r, sigma, q, "call", n_paths=200_000, seed=1)
    assert abs(p - bs) < 3 * se, f"MC {p:.4f} vs BS {bs:.4f}; SE={se:.4f}"


def test_mc_antithetic_cuts_variance():
    from pricers.monte_carlo import mc_european
    _, se_plain = mc_european(S0, K, T, r, sigma, q, "call", 50_000, seed=2)
    _, se_anti = mc_european(S0, K, T, r, sigma, q, "call", 50_000,
                             antithetic=True, seed=2)
    # Antithetic should help for vanilla call (monotonic payoff in Z).
    assert se_anti < se_plain * 0.85


def test_pathwise_delta_matches_bs():
    from greeks.pathwise import pathwise_delta_european
    from pricers.black_scholes import bs_greeks
    true_delta = float(bs_greeks(S0, K, T, r, sigma, q, "call")["delta"])
    d, se = pathwise_delta_european(S0, K, T, r, sigma, q, "call",
                                    n_paths=200_000, seed=3)
    assert abs(d - true_delta) < 3 * se


# ---- Phase 3: Binomial / LSM ----
def test_binomial_converges_to_bs():
    from pricers.binomial import binomial_price
    bs_p = float(bs_price(S0, K, T, r, sigma, q, "put"))
    err_50 = abs(binomial_price(S0, K, T, r, sigma, q, "put", "european", 50) - bs_p)
    err_500 = abs(binomial_price(S0, K, T, r, sigma, q, "put", "european", 500) - bs_p)
    assert err_500 < err_50  # convergence


def test_american_put_premium_over_european():
    from pricers.binomial import binomial_price
    eu = binomial_price(S0, K, T, r, sigma, q, "put", "european", 500)
    am = binomial_price(S0, K, T, r, sigma, q, "put", "american", 500)
    assert am >= eu


def test_american_call_equals_european_when_no_dividend():
    """Merton (1973): never optimal to early-exercise an American call on
    a non-dividend-paying stock."""
    from pricers.binomial import binomial_price
    eu = binomial_price(S0, K, T, r, sigma, 0.0, "call", "european", 500)
    am = binomial_price(S0, K, T, r, sigma, 0.0, "call", "american", 500)
    assert abs(am - eu) < 1e-6


def test_lsm_matches_binomial_american_put():
    from pricers.binomial import binomial_price
    from pricers.longstaff_schwartz import lsm_american_put
    am_bin = binomial_price(S0, K, 1.0, 0.06, sigma, 0.0, "put", "american", 500)
    am_lsm, se = lsm_american_put(S0, K, 1.0, 0.06, sigma, 0.0,
                                  n_paths=50_000, n_steps=50, seed=4)
    assert abs(am_lsm - am_bin) < 4 * se


# ---- Phase 4: IV solver / Heston ----
def test_iv_round_trip():
    from calibration.iv_solver import implied_vol
    sigma_true = 0.27
    price = float(bs_price(S0, 110.0, 0.5, r, sigma_true, q, "call"))
    sigma_back = implied_vol(price, S0, 110.0, 0.5, r, q, "call")
    assert abs(sigma_back - sigma_true) < 1e-6


def test_heston_bs_limit():
    """As sigma_v -> 0 and v0 = theta = const, Heston reduces to BS."""
    from models.heston import heston_price
    sigma_bs = 0.20
    bs_c = float(bs_price(100, 100, 1.0, 0.05, sigma_bs, 0.0, "call"))
    h_c = heston_price(100, 100, 1.0, 0.05, 0.0,
                       kappa=5.0, theta=sigma_bs**2, sigma_v=1e-3,
                       rho=0.0, v0=sigma_bs**2, option="call")
    assert abs(h_c - bs_c) < 1e-3


# ---- Phase 5: Delta hedge ----
def test_delta_hedge_realized_equals_implied_zero_mean():
    """If realized vol = implied vol, mean PnL of delta-hedged short
    straddle is ~0 (small noise from discrete hedging)."""
    from strategies.delta_hedge import (
        DeltaHedgeConfig, simulate_short_straddle_hedged,
    )
    cfg = DeltaHedgeConfig(S0=100, T=30 / 365, sigma_imp=0.20,
                          sigma_real=0.20, n_steps=30)
    pnls = np.array([simulate_short_straddle_hedged(cfg, rng_seed=s).final_pnl
                     for s in range(500)])
    assert abs(pnls.mean()) < 0.2, f"mean PnL = {pnls.mean()}"


def test_delta_hedge_decomposition_sums_to_total():
    """gamma + vega + tx + residual must sum to total final PnL."""
    from strategies.delta_hedge import (
        DeltaHedgeConfig, simulate_short_straddle_hedged,
    )
    cfg = DeltaHedgeConfig(S0=100, T=30 / 365, sigma_imp=0.25,
                          sigma_real=0.15, n_steps=30)
    r = simulate_short_straddle_hedged(cfg, rng_seed=42)
    d = r.pnl_decomposition
    total = d["gamma_pnl_total"] + d["vega_pnl_total"] + d["tx_costs_total"] + d["residual"]
    assert abs(total - r.final_pnl) < 1e-10
