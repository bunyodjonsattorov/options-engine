"""Example 2: Monte Carlo convergence + variance reduction + exotics.

Demonstrates:
    * MC error ~ 1/sqrt(N) (the textbook rate)
    * Antithetic variates cut variance ~2x for vanilla calls
    * Control variate on S_T cuts ~5x
    * Geometric-Asian control variate cuts ~1000x for arithmetic Asians

Run:  python3 examples/02_mc_convergence.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from pricers.black_scholes import bs_price
from pricers.monte_carlo import (
    mc_european,
    mc_european_with_cv,
    mc_asian_arithmetic,
    mc_barrier_up_and_out_call,
    mc_lookback_floating_call,
)
from greeks.pathwise import (
    pathwise_delta_european,
    likelihood_ratio_delta_european,
    likelihood_ratio_delta_digital,
)


def main():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.05, 0.20, 0.0
    bs = float(bs_price(S, K, T, r, sigma, q, "call"))
    print(f"=== MC convergence (call, BS = {bs:.4f}) ===\n")

    print(f"  {'N':>8}  {'MC plain':>14}  {'1.96·SE':>10}  {'err vs BS':>10}")
    for N in (1_000, 10_000, 100_000, 1_000_000):
        p, se = mc_european(S, K, T, r, sigma, q, "call", n_paths=N, seed=1)
        print(f"  {N:>8d}  {p:>14.6f}  {1.96*se:>10.4f}  {p - bs:>+10.4f}")

    print("\n=== Variance reduction (N = 100k) ===")
    p_pl, se_pl = mc_european(S, K, T, r, sigma, q, "call", 100_000, seed=2)
    p_at, se_at = mc_european(S, K, T, r, sigma, q, "call", 100_000, antithetic=True, seed=2)
    p_cv, se_cv = mc_european_with_cv(S, K, T, r, sigma, q, "call", 100_000, seed=2)
    print(f"  plain        SE = {se_pl:.5f}")
    print(f"  antithetic   SE = {se_at:.5f}  ({(se_pl/se_at)**2:.2f}x variance cut)")
    print(f"  CV on S_T    SE = {se_cv:.5f}  ({(se_pl/se_cv)**2:.2f}x variance cut)")

    print("\n=== Exotic options (no closed form, MC required) ===")
    p_as_p, se_as_p = mc_asian_arithmetic(S, K, T, r, sigma, q, "call",
                                          50_000, 50, antithetic=False,
                                          control_variate=False, seed=3)
    p_as_cv, se_as_cv = mc_asian_arithmetic(S, K, T, r, sigma, q, "call",
                                            50_000, 50, antithetic=False,
                                            control_variate=True, seed=3)
    print(f"  Asian arith MC plain : {p_as_p:.4f} +/- {1.96*se_as_p:.4f}")
    print(f"  Asian arith MC + CV  : {p_as_cv:.4f} +/- {1.96*se_as_cv:.4f}  "
          f"({(se_as_p/se_as_cv)**2:.0f}x variance cut)")
    p_bar, _ = mc_barrier_up_and_out_call(S, K, T, r, sigma, 120.0, q, 50_000, 252, seed=4)
    print(f"  Up-and-out call B=120: {p_bar:.4f}  (cf vanilla {bs:.4f}, "
          f"barrier reduces value)")
    p_lb, _ = mc_lookback_floating_call(S, T, r, sigma, q, 50_000, 252, seed=5)
    print(f"  Floating lookback    : {p_lb:.4f}  (always positive, "
          "biggest possible payoff)")

    print("\n=== Pathwise vs LR Greeks ===")
    from pricers.black_scholes import bs_greeks
    true_delta = float(bs_greeks(S, K, T, r, sigma, q, "call")["delta"])
    d_pw, se_pw = pathwise_delta_european(S, K, T, r, sigma, q, "call", 200_000, seed=6)
    d_lr, se_lr = likelihood_ratio_delta_european(S, K, T, r, sigma, q, "call", 200_000, seed=6)
    print(f"  Delta BS (true)   : {true_delta:.6f}")
    print(f"  Delta pathwise    : {d_pw:.6f} +/- {1.96*se_pw:.5f}")
    print(f"  Delta LR          : {d_lr:.6f} +/- {1.96*se_lr:.5f}  "
          f"(higher SE — LR's noisier when both apply)")

    print("\n=== LR delta for a DIGITAL (where pathwise fails) ===")
    d_dig, se_dig = likelihood_ratio_delta_digital(S, K, T, r, sigma, q, "call", 200_000, seed=7)
    print(f"  LR delta of digital call: {d_dig:.6f} +/- {1.96*se_dig:.5f}")
    print("  (pathwise can't differentiate I(S_T > K); LR can.)")


if __name__ == "__main__":
    main()
