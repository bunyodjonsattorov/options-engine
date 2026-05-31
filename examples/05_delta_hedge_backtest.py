"""Example 5: delta-hedged short straddle — backtest + PnL decomposition.

Demonstrates the central options-MM intuition:
    PnL_hedged ~ 0.5 * Gamma_pos * (realized_var - implied_var) * dt
So a short-vol trade wins iff realized < implied.

Run:  python3 examples/05_delta_hedge_backtest.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from strategies.delta_hedge import (
    DeltaHedgeConfig,
    simulate_short_straddle_hedged,
    summarise_runs,
)


def run_dist(label, sigma_imp, sigma_real, n_runs=1000, **kw):
    cfg = DeltaHedgeConfig(
        S0=100, T=30 / 365, r=0.05, q=0.0,
        sigma_imp=sigma_imp, sigma_real=sigma_real, n_steps=30, **kw,
    )
    pnls = np.array([simulate_short_straddle_hedged(cfg, rng_seed=s).final_pnl
                     for s in range(n_runs)])
    stats = summarise_runs(pnls)
    print(f"  {label:<45} mean={stats['mean']:+7.4f}  std={stats['std']:.3f}  "
          f"win={stats['win_rate']:>5.1%}  p5={stats['p5']:+.3f}")
    return stats


def main():
    print("=== Short straddle delta-hedged — 1000 paths each ===\n")
    print("  Vol regime                                    "
          "Mean PnL    Std    Win    5%-ile")
    print("  " + "-" * 80)

    run_dist("Realized = implied (20%)",        0.20, 0.20)
    run_dist("Vol RICH  (sold 25% imp, real 15%)", 0.25, 0.15)
    run_dist("Vol CHEAP (sold 15% imp, real 25%)", 0.15, 0.25)
    run_dist("Same as RICH + 5bp half-spread on stock", 0.25, 0.15, bid_ask_bps=5.0)
    run_dist("Same as RICH + 1c per-share tx cost", 0.25, 0.15, tc_per_share=0.01)

    print("\n=== Single-path decomposition (vol rich, seed=42) ===")
    cfg = DeltaHedgeConfig(S0=100, T=30 / 365, r=0.05, q=0.0,
                          sigma_imp=0.25, sigma_real=0.15, n_steps=30)
    r = simulate_short_straddle_hedged(cfg, rng_seed=42)
    d = r.pnl_decomposition
    print(f"  total final PnL                = {r.final_pnl:+.4f}")
    print(f"    gamma component (vs implied) = {d['gamma_pnl_total']:+.4f}")
    print(f"    vega component (IV moves)    = {d['vega_pnl_total']:+.4f}")
    print(f"    transaction costs            = {d['tx_costs_total']:+.4f}")
    print(f"    residual (discrete hedge)    = {d['residual']:+.4f}")
    print(f"    --- sum ---                  = "
          f"{d['gamma_pnl_total'] + d['vega_pnl_total'] + d['tx_costs_total'] + d['residual']:+.4f}")

    print("\n=== Realized-vs-implied sweep ===")
    print(f"  {'sigma_real':>10}  {'mean PnL':>10}  {'win rate':>10}")
    for sigma_real in np.linspace(0.10, 0.30, 9):
        cfg = DeltaHedgeConfig(S0=100, T=30 / 365, sigma_imp=0.20,
                              sigma_real=sigma_real, n_steps=30)
        pnls = np.array([simulate_short_straddle_hedged(cfg, rng_seed=s).final_pnl
                         for s in range(500)])
        print(f"  {sigma_real:>10.2%}  {pnls.mean():>+10.4f}  {(pnls > 0).mean():>10.1%}")


if __name__ == "__main__":
    main()
