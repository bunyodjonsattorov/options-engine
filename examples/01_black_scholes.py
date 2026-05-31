"""Example 1: Black-Scholes pricing, Greeks, put-call parity.

Run:  python3 examples/01_black_scholes.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from pricers.black_scholes import bs_price, bs_greeks, put_call_parity_residual
from greeks.finite_difference import fd_greeks


def main():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.05, 0.20, 0.0

    print(f"=== Black-Scholes example ({K=}, {T=}y, {sigma=:.0%}, {r=:.0%}) ===\n")

    c = float(bs_price(S, K, T, r, sigma, q, "call"))
    p = float(bs_price(S, K, T, r, sigma, q, "put"))
    print(f"Call price: {c:.4f}")
    print(f"Put price:  {p:.4f}")

    residual = put_call_parity_residual(c, p, S, K, T, r, q)
    print(f"Put-call parity residual: {residual:.2e}  (should be ~0)\n")

    print("Analytical vs finite-difference Greeks (call):")
    g_an = bs_greeks(S, K, T, r, sigma, q, "call")
    g_fd = fd_greeks(bs_price, S=S, sigma=sigma, T=T, r=r, K=K, q=q, option="call")
    print(f"  {'Greek':<8} {'analytic':>12} {'fin-diff':>12} {'abs err':>10}")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        err = abs(g_an[name] - g_fd[name])
        print(f"  {name:<8} {float(g_an[name]):>12.6f} {g_fd[name]:>12.6f} {err:>10.2e}")

    # Vol smile sweep — vega is highest near ATM.
    print("\nVega by strike (ATM has the most vega):")
    strikes = np.linspace(80, 120, 9)
    vegas = bs_greeks(S, strikes, T, r, sigma, q, "call")["vega"]
    for K_, v in zip(strikes, vegas):
        bar = "#" * int(v * 4)
        print(f"  K={K_:>5.0f}  vega={float(v):.3f}  {bar}")


if __name__ == "__main__":
    main()
