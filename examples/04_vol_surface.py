"""Example 4: implied vol, SVI smile fit, Heston calibration.

Uses synthetic chain by default. Swap synthetic_chain() for
get_option_chain('SPY') if yfinance is installed and you want real data.

Run:  python3 examples/04_vol_surface.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.market_data import synthetic_chain
from calibration.iv_solver import add_iv_column
from calibration.svi import fit_svi_slice, svi_implied_vol
from calibration.heston_calibration import calibrate_heston


def main():
    print("=== Building synthetic option chain ===")
    chain = synthetic_chain(spot=100.0, r=0.05, q=0.0)
    chain = add_iv_column(chain, "mid")
    print(f"  {len(chain)} quotes  ({chain['T'].nunique()} maturities, "
          f"{chain['strike'].nunique()} strikes per side)\n")

    print("=== SVI smile fit (per maturity, calls only) ===")
    print(f"  {'T (months)':>11}  {'#quotes':>8}  {'RMSE (vol pts)':>16}  "
          f"{'a':>7}  {'b':>7}  {'rho':>7}")
    for T in sorted(chain["T"].unique()):
        sl = chain[(np.isclose(chain["T"], T)) & (chain["option"] == "call")]
        F = 100 * np.exp(0.05 * T)
        k = np.log(sl["strike"].values / F)
        iv = sl["iv"].values
        fit = fit_svi_slice(k, iv, T)
        a, b, rho, m, sigma_ = fit["params"]
        print(f"  {T * 12:>11.1f}  {len(sl):>8d}  {fit['rmse_in_vol'] * 100:>15.4f}%  "
              f"{a:>7.4f}  {b:>7.4f}  {rho:>7.4f}")

    print("\n=== Heston calibration (full surface, calls) ===")
    print("  ...this takes ~15 seconds for ~30 quotes...")
    subset = chain[chain["option"] == "call"].iloc[::3].reset_index(drop=True)
    result = calibrate_heston(subset)
    p = result["params"]
    print(f"  fit RMSE        : {result['rmse_in_vol'] * 100:.3f}% vol")
    print(f"  Feller satisfied: {result['feller_ok']}  (2 kappa theta > sigma_v^2?)")
    print(f"  kappa   = {p['kappa']:.4f}   (mean-reversion speed)")
    print(f"  theta   = {p['theta']:.4f}   (long-run variance, i.e. (~{np.sqrt(p['theta']) * 100:.1f}% vol)^2)")
    print(f"  sigma_v = {p['sigma_v']:.4f}   (vol of vol)")
    print(f"  rho     = {p['rho']:.4f}    (price-vol correlation; negative = equity leverage effect)")
    print(f"  v0      = {p['v0']:.4f}     (initial variance, ~{np.sqrt(p['v0']) * 100:.1f}% vol)")

    print("\n  Note: kappa often hits its upper bound — Heston calibration")
    print("  is famously ill-posed. Many (kappa, theta) pairs give nearly")
    print("  identical surfaces. Real production use: multi-start, fix")
    print("  one parameter from term structure, or regularise to a prior.")


if __name__ == "__main__":
    main()
