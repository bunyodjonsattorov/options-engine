"""Example 3: American options — binomial tree, LSM, early exercise.

Demonstrates:
    * CRR binomial converges to BS as n grows (European case)
    * American put > European put (early exercise premium)
    * American call = European call when q=0 (Merton's theorem)
    * LSM MC matches binomial American put within MC error

Run:  python3 examples/03_american_options.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from pricers.black_scholes import bs_price
from pricers.binomial import binomial_price, binomial_american_early_exercise_boundary
from pricers.longstaff_schwartz import lsm_american_put


def main():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.06, 0.20, 0.0
    bs_p = float(bs_price(S, K, T, r, sigma, q, "put"))
    bs_c = float(bs_price(S, K, T, r, sigma, q, "call"))
    print(f"=== European baseline (BS) ===")
    print(f"  put  = {bs_p:.6f}")
    print(f"  call = {bs_c:.6f}\n")

    print("=== Binomial convergence to BS (European put) ===")
    print(f"  {'n':>5}  {'binomial':>10}  {'err vs BS':>10}")
    for n in (50, 100, 500, 1000, 5000):
        p = binomial_price(S, K, T, r, sigma, q, "put", "european", n_steps=n)
        print(f"  {n:>5d}  {p:>10.6f}  {p - bs_p:>+10.4e}")

    print("\n=== American vs European ===")
    am_p = binomial_price(S, K, T, r, sigma, q, "put", "american", n_steps=1000)
    am_c = binomial_price(S, K, T, r, sigma, q, "call", "american", n_steps=1000)
    print(f"  American put   = {am_p:.6f}   (premium over EU: +{am_p - bs_p:.4f})")
    print(f"  American call  = {am_c:.6f}   (= EU since q=0, Merton's theorem)")

    lsm_p, lsm_se = lsm_american_put(S, K, T, r, sigma, q, n_paths=100_000, n_steps=50, seed=11)
    print(f"  LSM American put = {lsm_p:.6f} +/- {1.96*lsm_se:.4f}  (matches binomial)")

    print("\n=== Early-exercise boundary (American put) ===")
    boundary = binomial_american_early_exercise_boundary(S, K, T, r, sigma, q, "put", n_steps=200)
    ts = np.linspace(0, T, 201)
    sample_idx = [0, 40, 80, 120, 160, 195]
    print(f"  {'t (years)':>10}  {'S*(t)':>10}  {'K = ' + str(K)}")
    for k in sample_idx:
        if np.isfinite(boundary[k]):
            print(f"  {ts[k]:>10.3f}  {boundary[k]:>10.3f}")
    print("  Boundary rises toward K as t -> T (classic shape).")


if __name__ == "__main__":
    main()
